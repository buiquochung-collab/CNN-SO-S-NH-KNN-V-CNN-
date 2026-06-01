import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def draw_bounding_boxes(image, boxes, labels, scores=None, class_names=None, threshold=0.5):
    """
    Vẽ bounding boxes lên ảnh.
    """
    if isinstance(image, torch.Tensor):
        # Chuyển đổi tensor [C, H, W] về numpy [H, W, C]
        image = image.permute(1, 2, 0).cpu().numpy()
        # Denormalize (nếu đã normalize với ImageNet)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
    else:
        image = image.copy()

    for i in range(len(boxes)):
        if scores is not None and scores[i] < threshold:
            continue
            
        box = boxes[i].cpu().numpy().astype(int)
        label = int(labels[i].item())
        score = scores[i].item() if scores is not None else 1.0
        
        # Vẽ HCN
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # Text
        name = class_names[label] if class_names else str(label)
        text = f"{name}: {score:.2f}"
        cv2.putText(image, text, (box[0], max(0, box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    return image

def plot_loss_metrics(train_losses, val_losses, save_dir):
    """
    Vẽ biểu đồ Loss của quá trình huấn luyện.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

def save_checkpoint(model, optimizer, epoch, loss, filename):
    """Lưu trữ checkpoint mô hình."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename, model, optimizer=None):
    """Load checkpoint cho mô hình."""
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Loaded checkpoint '{filename}' (epoch {epoch})")
        return epoch, loss
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, 0
