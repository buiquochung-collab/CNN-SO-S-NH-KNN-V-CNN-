import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import CustomObjectDetectionDataset, get_transforms, collate_fn
from src.model import get_multi_scale_detector
from src.utils import save_checkpoint, plot_loss_metrics

def train_one_epoch(model, optimizer, data_loader, device, epoch, writer=None):
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)
    for images, targets in progress_bar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Loss computation (FasterRCNN trong PyTorch trả về dict các loss khi ở chế độ train)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        running_loss += losses.item()
        progress_bar.set_postfix({'loss': f"{losses.item():.4f}"})
        
    avg_loss = running_loss / len(data_loader)
    
    if writer:
        writer.add_scalar('Training Loss', avg_loss, epoch)
        # Log các loại loss cụ thể của Faster R-CNN
        # writer.add_scalar('Loss/classifier', loss_dict['loss_classifier'].item(), epoch)
        # writer.add_scalar('Loss/box_reg', loss_dict['loss_box_reg'].item(), epoch)
        # writer.add_scalar('Loss/objectness', loss_dict['loss_objectness'].item(), epoch)
        # writer.add_scalar('Loss/rpn_box_reg', loss_dict['loss_rpn_box_reg'].item(), epoch)
        
    return avg_loss

def main():
    # Cấu hình siêu tham số
    NUM_CLASSES = 2 # 1 class (car) + 1 background
    BATCH_SIZE = 2 # Giảm batch size cho máy yếu hoặc ít ảnh
    NUM_EPOCHS = 10 # Chạy thử 10 epoch
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {DEVICE}")
    
    # Cấu hình đường dẫn
    TRAIN_IMG_DIR = "dataset/train/images"
    TRAIN_ANN_FILE = "dataset/train/annotations.json"
    
    # Khởi tạo mô hình Multi-scale CNN với FPN
    model = get_multi_scale_detector(num_classes=NUM_CLASSES, backbone_name='resnet50')
    model.to(DEVICE)
    
    # Khởi tạo Optimizer và Learning Rate Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Early Stopping params
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Tensorboard Writer
    writer = SummaryWriter(log_dir="runs/multi_scale_cnn")
    
    # KÍCH HOẠT HUẤN LUYỆN
    train_dataset = CustomObjectDetectionDataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE, transforms=get_transforms(train=True))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0) # num_workers=0 để tránh lỗi đa luồng trên Windows
    
    train_losses = []
    
    print("Bắt đầu huấn luyện...")
    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, writer)
        train_losses.append(loss)
        
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] - Average Loss: {loss:.4f}")
        lr_scheduler.step()
        
        # Early Stopping & Model Checkpoint
        if loss < best_loss:
            best_loss = loss
            save_checkpoint(model, optimizer, epoch, loss, "models/best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping kích hoạt!")
                break
                
        # Lưu checkpoint định kỳ
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, loss, f"models/model_epoch_{epoch}.pth")
            
    # Vẽ và lưu đồ thị Loss
    plot_loss_metrics(train_losses, [], "models/")

    print("Pipeline huấn luyện đã được thiết lập. Hãy thêm dữ liệu vào 'dataset/' và bỏ comment đoạn mã trên để bắt đầu train.")

if __name__ == "__main__":
    main()
