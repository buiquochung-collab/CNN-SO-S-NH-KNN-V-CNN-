import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from src.model import get_multi_scale_detector
from src.utils import draw_bounding_boxes
from torchvision.transforms import functional as F
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# pycocotools cho mAP thực tế
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_AVAILABLE = True
except ImportError:
    COCO_AVAILABLE = False

def test_single_image(model, image_path, device, class_names, threshold=0.5, save_path=None):
    """Test model trên một ảnh duy nhất và lưu kết quả."""
    model.eval()
    
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return
        
    image_rgb    = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(image_tensor)[0]
        
    boxes  = prediction['boxes']
    labels = prediction['labels']
    scores = prediction['scores']
    
    num_detected = (scores > threshold).sum().item()
    print(f"  → Tìm thấy {num_detected} đối tượng trên ngưỡng {threshold}")
    
    output_image = draw_bounding_boxes(
        image_rgb, boxes, labels, scores,
        class_names=class_names, threshold=threshold
    )
    
    plt.figure(figsize=(10, 8))
    plt.imshow(output_image)
    plt.axis('off')
    plt.title(f"Dự đoán: {os.path.basename(image_path)} ({num_detected} xe)")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"  → Đã lưu kết quả: {save_path}")
    
    plt.close()
    return num_detected

def evaluate_map_on_test(model, ann_file, img_dir, device, score_threshold=0.05):
    """
    Đánh giá mAP thực tế trên tập test bằng pycocotools.
    """
    if not COCO_AVAILABLE:
        print("  ⚠️  pycocotools chưa cài. Bỏ qua mAP. Chạy: pip install pycocotools")
        return None
    
    print(f"\n  Đánh giá mAP trên {ann_file}...")
    model.eval()
    coco_gt  = COCO(ann_file)
    img_ids  = list(coco_gt.imgs.keys())
    
    from tqdm import tqdm
    coco_results = []
    
    with torch.no_grad():
        for img_id in tqdm(img_ids, desc="  Chạy inference (test set)"):
            img_info = coco_gt.imgs[img_id]
            img_path = os.path.join(img_dir, img_info['file_name'])
            image    = cv2.imread(img_path)
            if image is None:
                continue
            image_rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            prediction = model(img_tensor)[0]
            
            boxes  = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            
            for box, score, label in zip(boxes, scores, labels):
                if score < score_threshold:
                    continue
                x1, y1, x2, y2 = box
                coco_results.append({
                    'image_id':    int(img_id),
                    'category_id': int(label),
                    'bbox':        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'score':       float(score)
                })
    
    if not coco_results:
        print("  → Không có detection nào vượt ngưỡng. mAP = 0.0")
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0, "mAP_small": 0.0}
    
    coco_dt   = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    stats = coco_eval.stats
    return {
        "mAP":       float(stats[0]),
        "mAP_50":    float(stats[1]),
        "mAP_75":    float(stats[2]),
        "mAP_small": float(stats[3]),
    }

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="models/confusion_matrix.png"):
    """Vẽ và lưu Confusion Matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.title('Confusion Matrix — CNN Multiscale')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Đã lưu Confusion Matrix: {save_path}")

def main():
    DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES = 2
    CLASS_NAMES = ["Background", "Car"]
    
    print(f"Thiết bị: {DEVICE}")
    
    # Nạp model từ checkpoint
    checkpoint_path = "models/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print("CẢNH BÁO: Không tìm thấy trọng số. Hãy chạy train.py hoặc compare_models.py trước.")
        return
    
    checkpoint    = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    backbone_name = checkpoint.get('backbone_name', 'resnet50')  # Đọc backbone từ checkpoint
    
    model = get_multi_scale_detector(num_classes=NUM_CLASSES, backbone_name=backbone_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    print(f"Đã tải trọng số từ {checkpoint_path} (Epoch {checkpoint['epoch']}, backbone: {backbone_name})")
    
    # ---- 1. Test ảnh mẫu ----
    test_img_dir = "dataset/test/images"
    if os.path.exists(test_img_dir) and os.listdir(test_img_dir):
        first_img = sorted(os.listdir(test_img_dir))[0]
        test_img  = os.path.join(test_img_dir, first_img)
        print(f"\n--- Test ảnh đơn: {test_img} ---")
        test_single_image(model, test_img, DEVICE, CLASS_NAMES,
                          threshold=0.5, save_path="models/result_sample.png")
    else:
        print(f"\nKhông tìm thấy ảnh test trong {test_img_dir}")
    
    # ---- 2. Đánh giá mAP trên tập test ----
    test_ann  = "dataset/test/annotations.json"
    test_imgs = "dataset/test/images"
    if os.path.exists(test_ann):
        print("\n--- Đánh giá mAP trên tập Test ---")
        map_results = evaluate_map_on_test(model, test_ann, test_imgs, DEVICE)
        if map_results:
            print(f"\n  ✅ mAP@0.5:0.95 = {map_results['mAP']:.4f}")
            print(f"  ✅ mAP@0.50     = {map_results['mAP_50']:.4f}")
            print(f"  ✅ mAP@0.75     = {map_results['mAP_75']:.4f}")
            print(f"  ✅ mAP_small    = {map_results['mAP_small']:.4f}")
            
            os.makedirs("results", exist_ok=True)
            with open("results/test_map_results.json", "w") as f:
                json.dump(map_results, f, indent=2)
            print("  → Đã lưu kết quả test: results/test_map_results.json")
    else:
        print(f"\n  ⚠️  Không có {test_ann}. Bỏ qua đánh giá mAP trên test set.")
    
    # ---- 3. Confusion Matrix (dùng dữ liệu val nếu có) ----
    val_ann  = "dataset/val/annotations.json"
    val_imgs = "dataset/val/images"
    if os.path.exists(val_ann) and COCO_AVAILABLE:
        print("\n--- Tạo Confusion Matrix trên tập Validation ---")
        from pycocotools.coco import COCO
        from tqdm import tqdm
        model.eval()
        coco_gt  = COCO(val_ann)
        img_ids  = list(coco_gt.imgs.keys())[:50]  # Lấy 50 ảnh mẫu
        
        y_true, y_pred = [], []
        with torch.no_grad():
            for img_id in tqdm(img_ids, desc="  Thu thập dự đoán"):
                img_info  = coco_gt.imgs[img_id]
                img_path  = os.path.join(val_imgs, img_info['file_name'])
                image     = cv2.imread(img_path)
                if image is None:
                    continue
                image_rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
                
                prediction = model(img_tensor)[0]
                scores     = prediction['scores'].cpu().numpy()
                labels     = prediction['labels'].cpu().numpy()
                
                # Ground truth: ảnh này có xe không?
                gt_anns    = coco_gt.getAnnIds(imgIds=[img_id])
                has_car    = 1 if len(gt_anns) > 0 else 0
                
                # Dự đoán: model có phát hiện xe nào score > 0.5 không?
                pred_car   = 1 if any((scores > 0.5) & (labels == 1)) else 0
                
                y_true.append(has_car)
                y_pred.append(pred_car)
        
        if y_true:
            plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)
            print("\n  Classification Report:")
            print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    print("\n✅ test.py hoàn tất!")

if __name__ == "__main__":
    main()
