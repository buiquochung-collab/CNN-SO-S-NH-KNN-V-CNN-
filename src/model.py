import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from collections import OrderedDict

from torchvision.models.detection.rpn import AnchorGenerator

class MultiScaleContextBlock(nn.Module):
    """
    Block trích xuất đặc trưng đa tỉ lệ (Multi-scale Feature Extraction).
    Sử dụng nhiều nhánh Convolution với kernel size khác nhau.
    Tương tự Inception Module nhưng nhẹ hơn và tối ưu cho vật thể nhỏ.
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleContextBlock, self).__init__()
        # Nhánh 1: 1x1 Conv — bắt đặc trưng cục bộ tức thì
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Nhánh 2: 3x3 Conv — bắt đặc trưng tầm trung
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Nhánh 3: 5x5 equiv (2x 3x3) — bắt đặc trưng tầm xa, ít tham số hơn 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Nhánh 4: MaxPooling — bắt đặc trưng bất biến với dịch chuyển
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Feature Fusion — kết hợp 4 nhánh
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        out = self.fusion(out)
        return out

class CustomBackboneWithFPN(nn.Module):
    """
    Backbone tích hợp Multi-scale Context Block và FPN với LastLevelMaxPool.
    FPN sinh ra 5 cấp độ đặc trưng: '0', '1', '2', '3', 'pool'
    phù hợp với tiêu chuẩn Faster R-CNN của torchvision.
    """
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super(CustomBackboneWithFPN, self).__init__()
        
        if backbone_name == 'resnet50':
            resnet = torchvision.models.resnet50(weights='DEFAULT' if pretrained else None)
            self.body = nn.ModuleDict({
                '0': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1), # C2: 256ch
                '1': resnet.layer2, # C3: 512ch
                '2': resnet.layer3, # C4: 1024ch
                '3': resnet.layer4  # C5: 2048ch
            })
            in_channels_list = [256, 512, 1024, 2048]
        elif backbone_name == 'mobilenet_v2':
            mobilenet = torchvision.models.mobilenet_v2(weights='DEFAULT' if pretrained else None).features
            self.body = nn.ModuleDict({
                '0': mobilenet[0:4],   # C2: 24ch
                '1': mobilenet[4:7],   # C3: 32ch
                '2': mobilenet[7:14],  # C4: 96ch
                '3': mobilenet[14:]    # C5: 1280ch
            })
            in_channels_list = [24, 32, 96, 1280]
        else:
            raise ValueError("Chỉ hỗ trợ 'resnet50' hoặc 'mobilenet_v2'")

        # MultiScaleContextBlock áp dụng lên feature map cuối (C5) để tăng cường đặc trưng
        self.multi_scale_block = MultiScaleContextBlock(in_channels_list[-1], in_channels_list[-1])
        
        self.out_channels = 256

        # FIX: Thêm LastLevelMaxPool để FPN sinh ra 5 levels ('0','1','2','3','pool')
        # Điều này đồng bộ với AnchorGenerator 5-tuple và MultiScaleRoIAlign
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool()
        )

    def forward(self, x):
        # FIX: Đổi tên biến trung gian để tránh ghi đè biến đầu vào x
        feature_maps = OrderedDict()
        feat = x
        for k, v in self.body.items():
            feat = v(feat)
            # Áp dụng MultiScaleContextBlock tại feature map cuối (C5)
            if k == '3':
                feat = self.multi_scale_block(feat)
            feature_maps[k] = feat
        out = self.fpn(feature_maps)
        return out

def get_multi_scale_detector(num_classes, backbone_name='resnet50'):
    """
    Khởi tạo mô hình Faster R-CNN với:
    - Anchor sizes siêu nhỏ (8-128px) cho vật thể vài pixel
    - FPN 5 levels với LastLevelMaxPool
    - MultiScaleRoIAlign trên tất cả 5 FPN levels
    """
    backbone = CustomBackboneWithFPN(backbone_name=backbone_name)
    
    # FIX: 5 anchor tuples khớp với 5 FPN output levels ('0','1','2','3','pool')
    anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )
    
    # FIX: Thêm MultiScaleRoIAlign để tận dụng tất cả 5 FPN feature maps
    # Thay vì chỉ dùng level '0' (default của FasterRCNN)
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3', 'pool'],
        output_size=7,
        sampling_ratio=2
    )
    
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        box_regression_loss_type='giou'
    )
    
    return model

if __name__ == "__main__":
    # Kiểm tra mô hình nhanh
    print("Đang khởi tạo mô hình...")
    model = get_multi_scale_detector(num_classes=2)
    print(f"Khởi tạo thành công. Số tham số: {sum(p.numel() for p in model.parameters()):,}")
    model.eval()
    x = torch.randn(1, 3, 800, 800)
    with torch.no_grad():
        out = model(x)
    print(f"Mô hình tạo ra {len(out[0]['boxes'])} bounding boxes trên ảnh thử nghiệm.")
    print("✅ model.py hoạt động bình thường!")
