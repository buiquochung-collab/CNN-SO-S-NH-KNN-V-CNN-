import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict

from torchvision.models.detection.rpn import AnchorGenerator

class MultiScaleContextBlock(nn.Module):
    """
    Block trích xuất đặc trưng đa tỉ lệ (Multi-scale Feature Extraction).
    Sử dụng nhiều nhánh Convolution với kernel size khác nhau.
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleContextBlock, self).__init__()
        # Nhánh 1: 1x1 Conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Nhánh 2: 3x3 Conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Nhánh 3: 5x5 Conv (thay bằng 2 lớp 3x3)
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
        
        # Nhánh 4: MaxPooling
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Feature Fusion
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
    Tạo Backbone có tích hợp Multi-scale Context Block và FPN.
    """
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super(CustomBackboneWithFPN, self).__init__()
        
        if backbone_name == 'resnet50':
            resnet = torchvision.models.resnet50(weights='DEFAULT' if pretrained else None)
            self.body = nn.ModuleDict({
                '0': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1), # C2
                '1': resnet.layer2, # C3
                '2': resnet.layer3, # C4
                '3': resnet.layer4  # C5
            })
            in_channels_list = [256, 512, 1024, 2048]
        elif backbone_name == 'mobilenet_v2':
            mobilenet = torchvision.models.mobilenet_v2(weights='DEFAULT' if pretrained else None).features
            self.body = nn.ModuleDict({
                '0': mobilenet[0:4],   # C2
                '1': mobilenet[4:7],   # C3
                '2': mobilenet[7:14],  # C4
                '3': mobilenet[14:]    # C5
            })
            in_channels_list = [24, 32, 96, 1280]
        else:
            raise ValueError("Chỉ hỗ trợ 'resnet50' hoặc 'mobilenet_v2'")

        self.multi_scale_block = MultiScaleContextBlock(in_channels_list[-1], in_channels_list[-1])
        
        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=self.out_channels
        )

    def forward(self, x):
        x_dict = OrderedDict()
        for k, v in self.body.items():
            x = v(x)
            if k == '3':
                x = self.multi_scale_block(x)
            x_dict[k] = x
        out = self.fpn(x_dict)
        return out

def get_multi_scale_detector(num_classes, backbone_name='resnet50'):
    """
    Khởi tạo mô hình Faster R-CNN với Anchor Generator tối ưu cho vật thể SIÊU NHỎ (Ảnh vệ tinh).
    """
    backbone = CustomBackboneWithFPN(backbone_name=backbone_name)
    
    # Anchor sizes siêu nhỏ: 8, 16, 32, 64 dành cho vật thể chỉ vài pixel
    anchor_sizes = ((8,), (16,), (32,), (64,)) 
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )
    
    # Khởi tạo Faster R-CNN
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator
    )
    
    return model

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_standard_detector(num_classes):
    """
    Khởi tạo mô hình Faster R-CNN nguyên bản (Standard CNN) để so sánh.
    Sử dụng ResNet50 + FPN chuẩn của PyTorch.
    """
    # Load model pre-trained trên COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Lấy số lượng input features của bộ phân loại
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Thay thế phần head để dự đoán số class của bài toán hiện tại
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    # Test mô hình nhanh
    model = get_multi_scale_detector(num_classes=10)
    print("Khởi tạo mô hình thành công.")
    x = torch.randn(1, 3, 800, 800)
    # Cần đặt ở chế độ eval để test output type
    model.eval()
    out = model(x)
    print(f"Mô hình tạo ra dự đoán với {len(out[0]['boxes'])} bounding boxes.")
