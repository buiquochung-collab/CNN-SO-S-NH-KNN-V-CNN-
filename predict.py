import torch
import cv2
import os
import sys
import time
from src.model import get_multi_scale_detector
from src.utils import draw_bounding_boxes
from torchvision.transforms import functional as F

def predict(image_path, threshold=0.3):
    """
    Chạy inference trên một ảnh với CNN Multiscale và lưu kết quả.
    
    Args:
        image_path (str): Đường dẫn ảnh đầu vào.
        threshold (float): Ngưỡng confidence để lọc detection.
    """
    DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES = 2
    CLASS_NAMES = ["Background", "Car"]
    CHECKPOINT_PATH = "models/best_model.pth"

    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy file ảnh tại {image_path}")
        return

    if not os.path.exists(CHECKPOINT_PATH):
        print("Lỗi: Không tìm thấy trọng số tại models/best_model.pth. Hãy train trước!")
        return

    # Load model — đọc backbone từ checkpoint để tránh mismatch
    print(f"Đang tải mô hình lên {DEVICE}...")
    checkpoint    = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    backbone_name = checkpoint.get('backbone_name', 'resnet50')
    
    model = get_multi_scale_detector(num_classes=NUM_CLASSES, backbone_name=backbone_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    print(f"Đã tải model (backbone: {backbone_name}, epoch: {checkpoint.get('epoch', 'N/A')})")

    # Xử lý ảnh
    image_cv  = cv2.imread(image_path)
    if image_cv is None:
        print(f"Lỗi: Không đọc được ảnh từ {image_path}")
        return
    image_rgb    = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(DEVICE)

    # Dự đoán và đo thời gian inference
    print(f"Đang dự đoán: {image_path} (threshold={threshold})...")
    
    # Warm-up (1 lần)
    with torch.no_grad():
        _ = model(image_tensor)
    
    # Đo thực tế
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    inf_ms = (time.perf_counter() - t0) * 1000

    # In kết quả
    boxes  = prediction['boxes']
    scores = prediction['scores']
    labels = prediction['labels']
    
    num_detected = (scores > threshold).sum().item()
    print(f"\n--- Kết quả Dự đoán ---")
    print(f"  Thời gian inference: {inf_ms:.1f} ms")
    print(f"  Số xe phát hiện (score > {threshold}): {num_detected}")
    
    if num_detected > 0:
        valid_mask = scores > threshold
        top_score  = float(scores[valid_mask].max())
        print(f"  Score cao nhất: {top_score:.4f}")
    
    # Vẽ kết quả
    output_image = draw_bounding_boxes(
        image_rgb, boxes, labels, scores,
        class_names=CLASS_NAMES, threshold=threshold
    )

    # Lưu kết quả với tên rõ ràng
    base_name   = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"result_{base_name}_thr{threshold}.png"
    cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    print(f"\n✅ Kết quả đã được lưu: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        predict(sys.argv[1], float(sys.argv[2]))
    elif len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        path   = input("Nhập đường dẫn file ảnh: ").strip()
        thresh = input("Nhập ngưỡng threshold (Enter để dùng 0.3): ").strip()
        predict(path, float(thresh) if thresh else 0.3)
