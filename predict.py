import torch
import cv2
import os
import sys
from src.model import get_multi_scale_detector
from src.utils import draw_bounding_boxes
from torchvision.transforms import functional as F

def predict(image_path, threshold=0.3):
    # Cấu hình
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES = 2
    CLASS_NAMES = ["Background", "Car"]
    CHECKPOINT_PATH = "models/best_model.pth"

    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy file ảnh tại {image_path}")
        return

    if not os.path.exists(CHECKPOINT_PATH):
        print("Lỗi: Không tìm thấy trọng số mô hình tại models/best_model.pth. Hãy train trước!")
        return

    # Load model
    print(f"Đang tải mô hình lên {DEVICE}...")
    model = get_multi_scale_detector(num_classes=NUM_CLASSES)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    # Xử lý ảnh
    image_cv = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(DEVICE)

    # Dự đoán
    print(f"Đang dự đoán ảnh: {image_path} với ngưỡng {threshold}...")
    with torch.no_grad():
        prediction = model(image_tensor)[0]

    # In ra số lượng box tìm thấy
    num_detected = (prediction['scores'] > threshold).sum().item()
    print(f"Tìm thấy {num_detected} xe có độ tự tin trên {threshold}.")

    # Vẽ kết quả
    output_image = draw_bounding_boxes(
        image_rgb, 
        prediction['boxes'], 
        prediction['labels'], 
        prediction['scores'], 
        class_names=CLASS_NAMES, 
        threshold=threshold
    )

    # Lưu kết quả
    output_path = "result_prediction.png"
    cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    print(f"Xong! Kết quả đã được lưu tại: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        predict(sys.argv[1], float(sys.argv[2]))
    elif len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        path = input("Nhập đường dẫn file ảnh: ")
        thresh = input("Nhập ngưỡng threshold (mặc định 0.3): ")
        if thresh:
            predict(path, float(thresh))
        else:
            predict(path)
