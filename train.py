import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import CustomObjectDetectionDataset, get_transforms, collate_fn
from src.model import get_multi_scale_detector
from src.utils import save_checkpoint, plot_loss_metrics

# ============================================================
# Hằng số cấu hình — đồng bộ với compare_models.py và app.py
# ============================================================
BACKBONE_NAME = 'resnet50'   # ResNet50 cho độ chính xác cao hơn
NUM_CLASSES   = 2            # 1 class (car) + 1 background
BATCH_SIZE    = 2            # Giảm batch size cho máy CPU hoặc GPU nhỏ
NUM_EPOCHS    = 10
LEARNING_RATE = 0.001

def train_one_epoch(model, optimizer, data_loader, device, epoch, writer=None):
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}", leave=False)
    for images, targets in progress_bar:
        images  = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # FasterRCNN trả về dict các loss khi ở chế độ train
        loss_dict = model(images, targets)
        losses    = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        running_loss += losses.item()
        progress_bar.set_postfix({'loss': f"{losses.item():.4f}"})
        
    avg_loss = running_loss / len(data_loader)
    
    if writer:
        writer.add_scalar('Loss/train_total', avg_loss, epoch)
        # Log từng thành phần loss của Faster R-CNN
        for loss_name, loss_val in loss_dict.items():
            writer.add_scalar(f'Loss/{loss_name}', loss_val.item(), epoch)
        
    return avg_loss

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {DEVICE}")
    print(f"Backbone: {BACKBONE_NAME} | Batch size: {BATCH_SIZE} | Epochs: {NUM_EPOCHS}")
    
    # Đường dẫn dữ liệu
    TRAIN_IMG_DIR  = "dataset/train/images"
    TRAIN_ANN_FILE = "dataset/train/annotations.json"
    VAL_IMG_DIR    = "dataset/val/images"
    VAL_ANN_FILE   = "dataset/val/annotations.json"
    
    # FIX: Dùng BACKBONE_NAME nhất quán để đảm bảo app.py load đúng weights
    model = get_multi_scale_detector(num_classes=NUM_CLASSES, backbone_name=BACKBONE_NAME)
    model.to(DEVICE)
    
    # Optimizer và Scheduler
    params       = [p for p in model.parameters() if p.requires_grad]
    optimizer    = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Early Stopping
    best_loss        = float('inf')
    patience         = 5
    patience_counter = 0
    
    # Tensorboard
    import time as time_module
    writer = SummaryWriter(log_dir=f"tensorboard_logs/multi_scale_cnn_{int(time_module.time())}")
    
    # Dataset và DataLoader
    train_dataset = CustomObjectDetectionDataset(
        TRAIN_IMG_DIR, TRAIN_ANN_FILE, transforms=get_transforms(train=True))
    train_loader  = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0)  # num_workers=0 cho Windows
    
    val_dataset = CustomObjectDetectionDataset(
        VAL_IMG_DIR, VAL_ANN_FILE, transforms=get_transforms(train=False))
    val_loader  = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0)
    
    train_losses = []
    val_losses   = []
    
    print(f"\nBắt đầu huấn luyện CNN Multiscale...")
    os.makedirs("models", exist_ok=True)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, writer)
        train_losses.append(train_loss)
        
        # Validation loss
        model.train()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images  = list(image.to(DEVICE) for image in images)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if writer:
            writer.add_scalar('Loss/val_total', val_loss, epoch)
        
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        lr_scheduler.step()
        
        # Early Stopping & Checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, "models/best_model.pth")
            # FIX: Lưu thêm backbone_name vào checkpoint để app.py load đúng
            checkpoint = torch.load("models/best_model.pth")
            checkpoint['backbone_name'] = BACKBONE_NAME
            torch.save(checkpoint, "models/best_model.pth")
            patience_counter = 0
            print(f"  ✅ Model tốt nhất được lưu (val_loss={best_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping kích hoạt!")
                break
                
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, train_loss, f"models/model_epoch_{epoch}.pth")
            
    # Lưu đồ thị Loss
    plot_loss_metrics(train_losses, val_losses, "models/")
    writer.close()
    print(f"\nHuấn luyện hoàn tất! Best Val Loss: {best_loss:.4f}")
    print("Trọng số tốt nhất: models/best_model.pth")

if __name__ == "__main__":
    main()
