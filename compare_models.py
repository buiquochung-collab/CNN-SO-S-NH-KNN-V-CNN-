import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import os
import time
import json
import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Không cần GUI window, lưu file trực tiếp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# pycocotools cho đánh giá mAP thực tế
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_AVAILABLE = True
except ImportError:
    print("CẢNH BÁO: pycocotools chưa được cài đặt. mAP sẽ không được tính.")
    print("Chạy: pip install pycocotools")
    COCO_AVAILABLE = False

from src.dataset import CustomObjectDetectionDataset, get_transforms, collate_fn
from src.model import get_multi_scale_detector
from src.utils import save_checkpoint

# ============================================================
# HẰNG SỐ CẤU HÌNH (đồng bộ toàn dự án)
# ============================================================
BACKBONE_NAME = 'mobilenet_v2'   # Dùng thống nhất trong train, compare, app
NUM_CLASSES = 2              # 1 class (car) + 1 background

# ==========================================
# 1. Định nghĩa Dataset phân loại (Classification)
# ==========================================
class ClassificationDataset(Dataset):
    """
    Dataset dùng để crop các ảnh 64x64 từ ảnh gốc làm dữ liệu phân loại.
    Lớp 1: Xe (crop từ bounding box).
    Lớp 0: Nền (crop ngẫu nhiên từ vùng không có xe).
    """
    def __init__(self, img_dir, ann_file, size=64, max_samples_per_image=15):
        self.img_dir = img_dir
        self.size = size
        
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
            
        self.images = {img['id']: img for img in coco_data['images']}
        self.annotations = coco_data['annotations']
        
        # Nhóm annotations theo image_id
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        self.samples = []
        
        print(f"Đang chuẩn bị dữ liệu phân loại từ {ann_file}...")
        for img_id, img_info in tqdm(self.images.items(), desc="Đang trích xuất ảnh crop"):
            img_path = os.path.join(self.img_dir, img_info['file_name'])
            image = cv2.imread(img_path)
            if image is None:
                continue
            h_img, w_img, _ = image.shape
            
            anns = self.img_to_anns.get(img_id, [])
            
            # Trích xuất mẫu xe (Lớp 1)
            num_cars = min(len(anns), max_samples_per_image)
            if len(anns) > 0:
                indices = np.random.choice(len(anns), num_cars, replace=False)
                for idx in indices:
                    ann = anns[idx]
                    x, y, w, h = ann['bbox']
                    x1, y1 = max(0, int(x)), max(0, int(y))
                    x2, y2 = min(w_img, int(x + w)), min(h_img, int(y + h))
                    if (x2 - x1) > 5 and (y2 - y1) > 5:
                        patch = image[y1:y2, x1:x2]
                        patch = cv2.resize(patch, (self.size, self.size))
                        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                        self.samples.append((patch, 1))
                        
            # Trích xuất mẫu nền (Lớp 0)
            num_bg = num_cars
            bg_taken = 0
            attempts = 0
            while bg_taken < num_bg and attempts < 100:
                attempts += 1
                if w_img <= self.size or h_img <= self.size:
                    break
                bx1 = np.random.randint(0, w_img - self.size)
                by1 = np.random.randint(0, h_img - self.size)
                bx2, by2 = bx1 + self.size, by1 + self.size
                
                # Kiểm tra giao nhau với các xe
                overlap = False
                for ann in anns:
                    ax, ay, aw, ah = ann['bbox']
                    ax1, ay1, ax2, ay2 = int(ax), int(ay), int(ax + aw), int(ay + ah)
                    if not (bx2 <= ax1 or bx1 >= ax2 or by2 <= ay1 or by1 >= ay2):
                        overlap = True
                        break
                        
                if not overlap:
                    patch = image[by1:by2, bx1:bx2]
                    patch = cv2.resize(patch, (self.size, self.size))
                    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                    self.samples.append((patch, 0))
                    bg_taken += 1
        
        n_car = len([s for s in self.samples if s[1] == 1])
        n_bg  = len([s for s in self.samples if s[1] == 0])
        print(f"Hoàn thành! Tạo ra {len(self.samples)} ảnh phân loại (Car: {n_car}, Background: {n_bg})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch, label = self.samples[idx]
        patch_t = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        return patch_t, label

# ==========================================
# 2. Định nghĩa mô hình CNN Base (Classification)
# ==========================================
class CNNBase(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNBase, self).__init__()
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==========================================
# 3. Hàm trích xuất đặc trưng kết hợp (HOG + HSV Color Histogram) cho KNN
# ==========================================
def extract_features_for_knn(patch):
    """
    Trích xuất đặc trưng kết hợp cho KNN:
    - HOG features (hình dạng, hướng gradient)
    - HSV Color Histogram (màu sắc của xe)
    """
    # 1. HOG features (grayscale)
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    hog_feat = hog(gray, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    
    # 2. Color Histogram (HSV space)
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    
    color_feat = np.concatenate([h_hist, s_hist, v_hist])
    color_feat /= (color_feat.sum() + 1e-6) # L1 normalize
    
    # Ghép hai đặc trưng lại thành 1 vector
    return np.concatenate([hog_feat, color_feat])


def extract_hog_dataset(dataset):
    """Trích xuất tập đặc trưng kết hợp cho toàn bộ dataset."""
    features = []
    labels = []
    for patch, label in tqdm(dataset.samples, desc="Trích xuất đặc trưng KNN (HOG + Color)"):
        feat = extract_features_for_knn(patch)
        features.append(feat)
        labels.append(label)
    return np.array(features), np.array(labels)

# ==========================================
# 4. Đo thời gian Inference (ms) — THỰC TẾ
# ==========================================
def measure_inference_time_cnn(model_fn, input_tensor, device, n_runs=30):
    """
    Đo inference time trung bình (ms) cho model PyTorch.
    Loại bỏ 5 lần warm-up đầu để kết quả ổn định.
    """
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model_fn(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    times = times[5:]  # Bỏ warm-up
    return float(np.mean(times)), float(np.std(times))

def measure_inference_time_knn(knn_model, hog_feat, n_runs=30):
    """Đo inference time (ms) cho KNN + HOG pipeline."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        knn_model.predict(hog_feat)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    times = times[5:]
    return float(np.mean(times)), float(np.std(times))

# ==========================================
# 5. Đánh giá mAP thực tế bằng pycocotools
# ==========================================
def evaluate_map_multiscale(model, ann_file, img_dir, device, score_threshold=0.05):
    """
    Đánh giá mAP@0.5:0.95 cho CNN Multiscale (Faster R-CNN) bằng pycocotools.
    Trả về dict chứa các chỉ số: mAP, mAP_50, mAP_75, mAP_small.
    """
    if not COCO_AVAILABLE:
        print("  → pycocotools không có. Bỏ qua đánh giá mAP.")
        return {"mAP": None, "mAP_50": None, "mAP_75": None, "mAP_small": None}

    print(f"\n  Đánh giá mAP trên {ann_file}...")
    model.eval()
    coco_gt = COCO(ann_file)
    img_ids = list(coco_gt.imgs.keys())
    
    coco_results = []
    with torch.no_grad():
        for img_id in tqdm(img_ids, desc="  Chạy inference để tính mAP"):
            img_info = coco_gt.imgs[img_id]
            img_path = os.path.join(img_dir, img_info['file_name'])
            image = cv2.imread(img_path)
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
                    'image_id':   int(img_id),
                    'category_id': int(label),
                    'bbox':        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'score':       float(score)
                })
    
    if len(coco_results) == 0:
        print("  → Không có detection nào vượt ngưỡng score. mAP = 0.0")
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0, "mAP_small": 0.0}
    
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    stats = coco_eval.stats
    return {
        "mAP":       float(stats[0]),   # mAP@0.5:0.95
        "mAP_50":    float(stats[1]),   # mAP@0.50
        "mAP_75":    float(stats[2]),   # mAP@0.75
        "mAP_small": float(stats[3]),   # mAP cho small objects (< 32^2 px)
    }

def evaluate_image_level_accuracy_multiscale(model, ann_file, img_dir, device, score_threshold=0.3):
    """
    Tính Accuracy % ở cấp độ ảnh (image-level) cho CNN Multiscale.

    Quy tắc chuyển đổi Detector → Classifier:
      - Nếu model phát hiện >=1 box với score > threshold và label==1  → dự đoán "Có xe" (1)
      - Ngược lại                                                         → dự đoán "Không xe" (0)
      - Ground truth: ảnh có ít nhất 1 annotation → label 1, không có → label 0

    Cho phép so sánh accuracy % trực tiếp với KNN và CNN Base.
    """
    if not COCO_AVAILABLE:
        print("  → pycocotools chưa có. Bỏ qua image-level accuracy.")
        return {"accuracy": None, "precision": None, "recall": None, "f1": None}

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    print(f"  Tính image-level accuracy trên {ann_file}...")
    model.eval()
    coco_gt  = COCO(ann_file)
    img_ids  = list(coco_gt.imgs.keys())

    y_true, y_pred = [], []

    with torch.no_grad():
        for img_id in tqdm(img_ids, desc="  Image-level accuracy CNN Multiscale"):
            img_info = coco_gt.imgs[img_id]
            img_path = os.path.join(img_dir, img_info['file_name'])
            image    = cv2.imread(img_path)
            if image is None:
                continue

            # Ground truth: ảnh có xe không?
            gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
            has_car_gt = 1 if len(gt_ann_ids) > 0 else 0

            # Dự đoán: model phát hiện xe không?
            image_rgb  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)

            prediction = model(img_tensor)[0]
            scores = prediction['scores'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()

            has_car_pred = 1 if any((scores > score_threshold) & (labels == 1)) else 0

            y_true.append(has_car_gt)
            y_pred.append(has_car_pred)

    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "y_true": [], "y_pred": []}

    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "y_true":    y_true,
        "y_pred":    y_pred,
    }


def evaluate_map_classifier(model_or_knn, val_dataset, device, model_type='cnn', feature_scaler=None):
    """
    Đánh giá Precision/Recall/F1/Accuracy cho CNN Base và KNN (patch classifier).
    feature_scaler: StandardScaler đã fit trên train set (chỉ dùng cho KNN).
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    all_preds  = []
    all_labels = []

    if model_type == 'knn':
        for patch, label in tqdm(val_dataset.samples, desc="  Đánh giá KNN"):
            feat = extract_features_for_knn(patch).reshape(1, -1)
            if feature_scaler is not None:
                feat = feature_scaler.transform(feat)  # Chuẩn hoá đúng như lúc train
            pred = int(model_or_knn.predict(feat)[0])
            all_preds.append(pred)
            all_labels.append(int(label))
    else:
        model_or_knn.eval()
        loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="  Đánh giá CNN Base"):
                images = images.to(device)
                outputs = model_or_knn(images)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.numpy().tolist())

    return {
        "accuracy":  float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall":    float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1":        float(f1_score(all_labels, all_preds, zero_division=0)),
        "y_true":    all_labels,
        "y_pred":    all_preds,
    }

# ==========================================
# 6. Tính Validation Loss cho CNN Multiscale
# ==========================================
def compute_val_loss_multiscale(model, data_loader, device):
    """
    Tính validation loss cho Faster R-CNN.
    Dùng train mode để FasterRCNN trả về loss dict, kết hợp với torch.no_grad().
    """
    model.train()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
    return val_loss / len(data_loader)

# ==========================================
# 7. Huấn luyện và đánh giá CNN Base
# ==========================================
def train_cnn_base(train_dataset, val_dataset, device, num_epochs=10, batch_size=16, patience=3):
    print("\n--- Đang huấn luyện CNN Base ---")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    
    model     = CNNBase(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    metrics = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc": [],
        "early_stop_epoch": num_epochs
    }
    
    best_val_loss  = float('inf')
    best_val_acc   = 0.0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss  += loss.item() * images.size(0)
            _, predicted   = outputs.max(1)
            total_train   += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc  = correct_train / total_train
        
        # ---- Validate ----
        model.eval()
        running_val_loss = 0.0
        correct_val, total_val = 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted      = outputs.max(1)
                total_val        += labels.size(0)
                correct_val      += predicted.eq(labels).sum().item()
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc  = correct_val / total_val
        
        metrics["train_loss"].append(epoch_train_loss)
        metrics["val_loss"].append(epoch_val_loss)
        metrics["train_acc"].append(epoch_train_acc)
        metrics["val_acc"].append(epoch_val_acc)
        
        print(f"  Epoch [{epoch}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.2f}%"
              f" | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:.2f}%")
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss    = epoch_val_loss
            best_val_acc     = epoch_val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping kích hoạt tại epoch {epoch}!")
                metrics["early_stop_epoch"] = epoch
                break
                
    os.makedirs("models", exist_ok=True)
    if best_model_state is not None:
        torch.save(best_model_state, "models/cnn_base_best.pth")
        print("  Đã lưu trọng số CNN Base tốt nhất → models/cnn_base_best.pth")
        model.load_state_dict(best_model_state)
        
    return metrics, best_val_acc, best_val_loss, model

# ==========================================
# 8. Huấn luyện và đánh giá CNN Multiscale
# ==========================================
def train_cnn_multiscale(device, num_epochs=10, batch_size=2, patience=5):
    print("\n--- Đang huấn luyện CNN Multiscale (Faster R-CNN) ---")
    TRAIN_IMG_DIR = "dataset/train/images"
    TRAIN_ANN_FILE = "dataset/train/annotations.json"
    VAL_IMG_DIR   = "dataset/val/images"
    VAL_ANN_FILE   = "dataset/val/annotations.json"
    
    train_dataset = CustomObjectDetectionDataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE, transforms=get_transforms(train=True))
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    val_dataset = CustomObjectDetectionDataset(VAL_IMG_DIR, VAL_ANN_FILE, transforms=get_transforms(train=False))
    val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # FIX: dùng BACKBONE_NAME để đồng bộ với app.py và train.py
    model = get_multi_scale_detector(num_classes=NUM_CLASSES, backbone_name=BACKBONE_NAME)
    model.to(device)
    
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)
    
    metrics = {
        "train_loss": [], "val_loss": [],
        "early_stop_epoch": num_epochs
    }
    
    best_val_loss    = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        
        for images, targets in tqdm(train_loader, desc=f"  Epoch {epoch}/{num_epochs}", leave=False):
            images  = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses    = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            running_loss += losses.item()
            
        epoch_train_loss = running_loss / len(train_loader)
        epoch_val_loss   = compute_val_loss_multiscale(model, val_loader, device)
        
        metrics["train_loss"].append(epoch_train_loss)
        metrics["val_loss"].append(epoch_val_loss)
        
        print(f"  Epoch [{epoch}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss    = epoch_val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping kích hoạt tại epoch {epoch}!")
                metrics["early_stop_epoch"] = epoch
                break
                
    os.makedirs("models", exist_ok=True)
    if best_model_state is not None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
            'backbone_name': BACKBONE_NAME   # Lưu backbone để load_state_dict đúng
        }, "models/best_model.pth")
        print("  Đã lưu trọng số CNN Multiscale tốt nhất → models/best_model.pth")
        model.load_state_dict(best_model_state)
        
    return metrics, best_val_loss, model


# ==========================================
# 8.5. Đánh giá CNN Multiscale trên các Patch
# ==========================================
def evaluate_multiscale_on_patches(model, val_dataset, device, score_threshold=0.3):
    """
    Đánh giá CNN Multiscale trên các patch của ClassificationDataset bằng cách
    nhúng patch 64x64 vào trung tâm canvas 800x800 để mô hình detect đúng tỷ lệ đã học.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for patch, label in tqdm(val_dataset.samples, desc="  Đánh giá CNN Multiscale trên Patch"):
            # Nhúng patch vào canvas 800x800 để khớp anchor sizes
            canvas = np.zeros((800, 800, 3), dtype=np.uint8) + 127
            canvas[368:432, 368:432] = patch
            
            # Chuyển thành tensor (numpy RGB -> PyTorch tensor)
            img_tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            prediction = model(img_tensor)[0]
            scores = prediction['scores'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            
            # Dự đoán có xe nếu phát hiện ra bất kỳ xe nào có score > threshold
            has_car_pred = 1 if any((scores > score_threshold) & (labels == 1)) else 0
            all_preds.append(has_car_pred)
            all_labels.append(int(label))
            
    return {
        "y_true": all_labels,
        "y_pred": all_preds
    }


# ==========================================
# 8.6. Vẽ Sơ Đồ Nhầm Lẫn và Tương Quan
# ==========================================
def plot_confusion_and_correlation(knn_eval, cnn_base_eval, cnn_ms_patch_eval, cnn_ms_image_eval, save_dir="results"):
    """
    Vẽ và lưu sơ đồ nhầm lẫn (confusion matrix) và sơ đồ tương quan (correlation matrix) của 3 mô hình.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Lấy ground truth và predictions
    y_true_patch = knn_eval.get('y_true', [])
    y_pred_knn = knn_eval.get('y_pred', [])
    y_pred_base = cnn_base_eval.get('y_pred', [])
    y_pred_ms_patch = cnn_ms_patch_eval.get('y_pred', [])
    
    y_true_img = cnn_ms_image_eval.get('y_true', [])
    y_pred_ms_img = cnn_ms_image_eval.get('y_pred', [])
    
    # Đảm bảo dữ liệu không rỗng
    if not y_true_patch or not y_pred_knn or not y_pred_base or not y_pred_ms_patch:
        print("  ⚠️ Không đủ dữ liệu test để vẽ ma trận nhầm lẫn / tương quan.")
        return None, None

    classes = ['Không Xe', 'Có Xe']
    
    # --------------------------------------------------------
    # 1. Vẽ FIGURE: Confusion Matrices (Ma trận nhầm lẫn)
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Ma Trận Nhầm Lẫn (Confusion Matrix) của các Mô Hình trên Tập Test', 
                 fontsize=14, fontweight='bold', y=1.05)
    
    # KNN (Patch-level)
    cm_knn = confusion_matrix(y_true_patch, y_pred_knn)
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', ax=axes[0],
                xticklabels=classes, yticklabels=classes, cbar=False, annot_kws={"size": 12, "weight": "bold"})
    axes[0].set_title('KNN + HOG\n(Cấp độ Patch 64x64)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Dự đoán', fontsize=10)
    axes[0].set_ylabel('Thực tế', fontsize=10)
    
    # CNN Base (Patch-level)
    cm_base = confusion_matrix(y_true_patch, y_pred_base)
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=classes, yticklabels=classes, cbar=False, annot_kws={"size": 12, "weight": "bold"})
    axes[1].set_title('CNN Base\n(Cấp độ Patch 64x64)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Dự đoán', fontsize=10)
    axes[1].set_ylabel('Thực tế', fontsize=10)
    
    # CNN Multiscale (Patch-level)
    cm_ms_patch = confusion_matrix(y_true_patch, y_pred_ms_patch)
    sns.heatmap(cm_ms_patch, annot=True, fmt='d', cmap='Oranges', ax=axes[2],
                xticklabels=classes, yticklabels=classes, cbar=False, annot_kws={"size": 12, "weight": "bold"})
    axes[2].set_title('CNN Multiscale\n(Cấp độ Patch 64x64)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Dự đoán', fontsize=10)
    axes[2].set_ylabel('Thực tế', fontsize=10)
    
    # CNN Multiscale (Image-level)
    if len(y_true_img) > 0 and len(y_pred_ms_img) > 0:
        cm_ms_img = confusion_matrix(y_true_img, y_pred_ms_img)
        sns.heatmap(cm_ms_img, annot=True, fmt='d', cmap='Purples', ax=axes[3],
                    xticklabels=classes, yticklabels=classes, cbar=False, annot_kws={"size": 12, "weight": "bold"})
        axes[3].set_title('CNN Multiscale (Detector)\n(Cấp độ Ảnh 800x800)', fontsize=12, fontweight='bold')
        axes[3].set_xlabel('Dự đoán', fontsize=10)
        axes[3].set_ylabel('Thực tế', fontsize=10)
    else:
        axes[3].text(0.5, 0.5, 'Không có dữ liệu\nImage-level', ha='center', va='center', fontsize=12)
        axes[3].set_title('CNN Multiscale (Detector)\n(Cấp độ Ảnh 800x800)', fontsize=12, fontweight='bold')
        axes[3].axis('off')
    
    plt.tight_layout()
    path_cm = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(path_cm, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Đã lưu ma trận nhầm lẫn: {path_cm}")
    
    # --------------------------------------------------------
    # 2. Vẽ FIGURE: Correlation Matrix (Ma trận tương quan dự đoán)
    # --------------------------------------------------------
    df_preds = pd.DataFrame({
        'KNN + HOG': y_pred_knn,
        'CNN Base': y_pred_base,
        'CNN Multiscale': y_pred_ms_patch
    })
    corr_matrix = df_preds.corr(method='pearson')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1, vmax=1,
                annot_kws={"size": 12, "weight": "bold"}, square=True, linewidths=.5)
    plt.title('Biểu Đồ Tương Quan Dự Đoán (Prediction Correlation Heatmap)\ngiữa 3 Mô Hình trên tập Test (Patch-level)', 
              fontsize=12, fontweight='bold', pad=15)
    
    plt.tight_layout()
    path_corr = os.path.join(save_dir, 'correlation_matrix.png')
    plt.savefig(path_corr, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Đã lưu biểu đồ tương quan: {path_corr}")
    
    return path_cm, path_corr


# ==========================================
# 9. Vẽ Biểu Đồ Training Curves
# ==========================================
def plot_training_curves(cnn_base_metrics, cnn_multiscale_metrics, knn_metrics, save_dir="results"):
    """
    Vẽ và lưu biểu đồ training curves cho cả 3 mô hình.

    Figure 1 — CNN Base + CNN Multiscale (chung):
      - Subplot trái : Loss (train_loss + val_loss của cả 2 model, 4 đường)
      - Subplot phải : Accuracy (train_acc + val_acc của CNN Base;
                       CNN Multiscale không có epoch accuracy → ghi chú)
      - Đường đứt dọc: điểm Early Stopping của từng model

    Figure 2 — KNN:
      - Subplot trái : train_acc + val_acc qua các simulated epoch
      - Subplot phải : CV k-selection — accuracy ± std theo từng k
    """
    os.makedirs(save_dir, exist_ok=True)
    STYLE = {
        'base_train': dict(color='#2196F3', linestyle='-',  marker='o', markersize=5, linewidth=2),
        'base_val':   dict(color='#2196F3', linestyle='--', marker='s', markersize=5, linewidth=2),
        'ms_train':   dict(color='#F44336', linestyle='-',  marker='o', markersize=5, linewidth=2),
        'ms_val':     dict(color='#F44336', linestyle='--', marker='s', markersize=5, linewidth=2),
        'knn_train':  dict(color='#4CAF50', linestyle='-',  marker='o', markersize=6, linewidth=2),
        'knn_val':    dict(color='#4CAF50', linestyle='--', marker='s', markersize=6, linewidth=2),
    }

    # ─────────────────────────────────────────────────────────────────
    # FIGURE 1: CNN Base + CNN Multiscale
    # ─────────────────────────────────────────────────────────────────
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training Curves — CNN Base & CNN Multiscale\n(Early Stopping giúp tránh Overfitting)',
                 fontsize=14, fontweight='bold', y=1.02)

    # ---------- Subplot 1: LOSS ----------
    base_epochs = list(range(1, len(cnn_base_metrics['train_loss']) + 1))
    ms_epochs   = list(range(1, len(cnn_multiscale_metrics['train_loss']) + 1))
    base_stop   = cnn_base_metrics.get('early_stop_epoch', len(base_epochs))
    ms_stop     = cnn_multiscale_metrics.get('early_stop_epoch', len(ms_epochs))

    ax_loss.plot(base_epochs, cnn_base_metrics['train_loss'], label='CNN Base — Train Loss', **STYLE['base_train'])
    ax_loss.plot(base_epochs, cnn_base_metrics['val_loss'],   label='CNN Base — Val Loss',   **STYLE['base_val'])
    ax_loss.plot(ms_epochs,   cnn_multiscale_metrics['train_loss'], label='CNN Multiscale — Train Loss', **STYLE['ms_train'])
    ax_loss.plot(ms_epochs,   cnn_multiscale_metrics['val_loss'],   label='CNN Multiscale — Val Loss',   **STYLE['ms_val'])

    # Đường đứt dọc Early Stopping
    if base_stop <= len(base_epochs):
        ax_loss.axvline(x=base_stop, color='#2196F3', linestyle=':', linewidth=2.5, alpha=0.8)
        ax_loss.annotate(f'Early Stop\nCNN Base\n(Ep.{base_stop})',
                         xy=(base_stop, ax_loss.get_ylim()[1] if ax_loss.get_ylim()[1] != 1 else 1),
                         xytext=(base_stop + 0.3, 0),
                         textcoords=('data', 'axes fraction'),
                         fontsize=8, color='#2196F3',
                         bbox=dict(boxstyle='round,pad=0.3', fc='#E3F2FD', alpha=0.8))
    if ms_stop <= len(ms_epochs):
        ax_loss.axvline(x=ms_stop, color='#F44336', linestyle=':', linewidth=2.5, alpha=0.8)
        ax_loss.annotate(f'Early Stop\nMultiscale\n(Ep.{ms_stop})',
                         xy=(ms_stop, 0),
                         xytext=(ms_stop + 0.3, 0.55),
                         textcoords=('data', 'axes fraction'),
                         fontsize=8, color='#F44336',
                         bbox=dict(boxstyle='round,pad=0.3', fc='#FFEBEE', alpha=0.8))

    ax_loss.set_xlabel('Epoch', fontsize=11)
    ax_loss.set_ylabel('Loss', fontsize=11)
    ax_loss.set_title('Loss — Train vs Validation', fontsize=12, fontweight='bold')
    ax_loss.legend(fontsize=9, loc='upper right')
    ax_loss.grid(True, alpha=0.3, linestyle='--')
    ax_loss.set_facecolor('#FAFAFA')

    # ---------- Subplot 2: ACCURACY ----------
    # CNN Base có train_acc + val_acc theo epoch
    base_train_acc = [a * 100 for a in cnn_base_metrics['train_acc']]
    base_val_acc   = [a * 100 for a in cnn_base_metrics['val_acc']]

    ax_acc.plot(base_epochs, base_train_acc, label='CNN Base — Train Acc', **STYLE['base_train'])
    ax_acc.plot(base_epochs, base_val_acc,   label='CNN Base — Val Acc',   **STYLE['base_val'])

    # Điểm Early Stopping CNN Base
    if base_stop <= len(base_epochs):
        ax_acc.axvline(x=base_stop, color='#2196F3', linestyle=':', linewidth=2.5, alpha=0.8)
        stop_val = base_val_acc[base_stop - 1] if base_stop <= len(base_val_acc) else base_val_acc[-1]
        ax_acc.scatter([base_stop], [stop_val], color='#2196F3', s=120, zorder=5,
                       marker='*', label=f'Best Val Acc CNN Base ({stop_val:.1f}%)')

    # CNN Multiscale không có epoch accuracy — thêm hộp chú thích
    ms_final_acc = cnn_multiscale_metrics.get('eval_image_level', {})
    ms_acc_val   = ms_final_acc.get('accuracy', None)
    note_text    = (f"CNN Multiscale:\nKhông có Epoch Accuracy\n"
                    f"(Full Detector — Faster R-CNN)\n\n"
                    f"Image-level Acc (Val):\n"
                    f"{ms_acc_val*100:.1f}%" if ms_acc_val is not None
                    else "CNN Multiscale:\nKhông có Epoch Accuracy\n(Full Detector)")
    ax_acc.text(0.97, 0.05, note_text,
                transform=ax_acc.transAxes, fontsize=8.5,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0',
                          edgecolor='#F44336', linewidth=1.5, alpha=0.9))

    # Thêm điểm accuracy cuối cùng của CNN Multiscale (image-level)
    if ms_acc_val is not None:
        ax_acc.axhline(y=ms_acc_val * 100, color='#F44336', linestyle='-.', linewidth=1.8, alpha=0.7,
                       label=f'CNN Multiscale — Val Acc (image-level) {ms_acc_val*100:.1f}%')

    ax_acc.set_xlabel('Epoch', fontsize=11)
    ax_acc.set_ylabel('Accuracy (%)', fontsize=11)
    ax_acc.set_title('Accuracy — Train vs Validation', fontsize=12, fontweight='bold')
    ax_acc.legend(fontsize=9, loc='lower right')
    ax_acc.grid(True, alpha=0.3, linestyle='--')
    ax_acc.set_facecolor('#FAFAFA')
    ax_acc.set_ylim(0, 105)

    plt.tight_layout()
    path1 = os.path.join(save_dir, 'cnn_training_curves.png')
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✅ Đã lưu biểu đồ CNN: {path1}")

    # ─────────────────────────────────────────────────────────────────
    # FIGURE 2: KNN
    # ─────────────────────────────────────────────────────────────────
    has_cv = ('cv_k_values' in knn_metrics and 'cv_mean_scores' in knn_metrics)
    n_cols = 2 if has_cv else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 6))
    if n_cols == 1:
        axes = [axes]
    fig.suptitle('KNN — Simulated Epoch Accuracy & Cross-Validation k Selection\n'
                 '(Cross-Validation thay thế Early Stopping để chống Overfitting)',
                 fontsize=13, fontweight='bold', y=1.02)

    # ---------- Subplot 1: Accuracy qua simulated epochs ----------
    ax_knn = axes[0]
    sim_epochs   = knn_metrics['epochs']
    knn_train_acc = [a * 100 for a in knn_metrics['train_acc']]
    knn_val_acc   = [a * 100 for a in knn_metrics['val_acc']]
    best_k_used   = knn_metrics.get('best_k', 5)

    ax_knn.plot(sim_epochs, knn_train_acc, label=f'KNN (k={best_k_used}) — Train Acc', **STYLE['knn_train'])
    ax_knn.plot(sim_epochs, knn_val_acc,   label=f'KNN (k={best_k_used}) — Val Acc',   **STYLE['knn_val'])

    # Điểm tốt nhất trên val
    best_idx     = int(np.argmax(knn_val_acc))
    best_ep      = sim_epochs[best_idx]
    best_val_val = knn_val_acc[best_idx]
    ax_knn.scatter([best_ep], [best_val_val], color='#4CAF50', s=150, zorder=5,
                   marker='*', label=f'Best Val Acc ({best_val_val:.1f}% tại ep.{best_ep})')

    # Ghi chú không có early stopping
    ax_knn.text(0.03, 0.03,
                f'KNN là Lazy Learner:\nKhông có gradient descent\n→ Không cần Early Stopping\n'
                f'Chống Overfit bằng CV chọn k={best_k_used}',
                transform=ax_knn.transAxes, fontsize=9,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9',
                          edgecolor='#4CAF50', linewidth=1.5, alpha=0.9))

    # X-axis labels: % dữ liệu
    ax_knn.set_xticks(sim_epochs)
    ax_knn.set_xticklabels([f'{p*10}%' for p in range(1, 11)], rotation=30, ha='right')
    ax_knn.set_xlabel('Simulated Epoch (% dữ liệu train được dùng)', fontsize=11)
    ax_knn.set_ylabel('Accuracy (%)', fontsize=11)
    ax_knn.set_title(f'KNN (k={best_k_used}) — Train vs Val Accuracy', fontsize=12, fontweight='bold')
    ax_knn.legend(fontsize=9)
    ax_knn.grid(True, alpha=0.3, linestyle='--')
    ax_knn.set_facecolor('#FAFAFA')
    ax_knn.set_ylim(0, 105)

    # ---------- Subplot 2: Cross-Validation k selection ----------
    if has_cv:
        ax_cv = axes[1]
        k_vals     = knn_metrics['cv_k_values']
        cv_means   = [s * 100 for s in knn_metrics['cv_mean_scores']]
        cv_stds    = [s * 100 for s in knn_metrics['cv_std_scores']]
        best_k_idx = k_vals.index(best_k_used) if best_k_used in k_vals else int(np.argmax(cv_means))

        colors = ['#66BB6A' if i == best_k_idx else '#B0BEC5' for i in range(len(k_vals))]
        bars   = ax_cv.bar(range(len(k_vals)), cv_means, color=colors,
                           width=0.6, edgecolor='white', linewidth=1.2, alpha=0.9)
        ax_cv.errorbar(range(len(k_vals)), cv_means, yerr=cv_stds,
                       fmt='none', color='#546E7A', capsize=4, linewidth=1.5, alpha=0.8)

        # Highlight best k
        ax_cv.bar(best_k_idx, cv_means[best_k_idx], color='#2E7D32',
                  width=0.6, edgecolor='#1B5E20', linewidth=2, alpha=1.0,
                  label=f'k={best_k_used} tối ưu ({cv_means[best_k_idx]:.1f}%)')
        ax_cv.axhline(y=cv_means[best_k_idx], color='#2E7D32',
                      linestyle='--', linewidth=1.5, alpha=0.5)

        # Nhãn giá trị trên từng bar
        for i, (bar, val) in enumerate(zip(bars, cv_means)):
            ax_cv.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                       f'{val:.1f}%', ha='center', va='bottom',
                       fontsize=7.5, color='#212121' if i == best_k_idx else '#546E7A',
                       fontweight='bold' if i == best_k_idx else 'normal')

        ax_cv.set_xticks(range(len(k_vals)))
        ax_cv.set_xticklabels([f'k={k}' for k in k_vals], rotation=30, ha='right', fontsize=9)
        ax_cv.set_xlabel('Giá trị k (số lân cận)', fontsize=11)
        ax_cv.set_ylabel('CV Accuracy (%) ± std', fontsize=11)
        ax_cv.set_title('5-fold Cross-Validation: Chọn k tối ưu\n'
                        '(k nhỏ = overfit | k lớn = underfit)', fontsize=12, fontweight='bold')
        ax_cv.legend(fontsize=9)
        ax_cv.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax_cv.set_facecolor('#FAFAFA')

        # Vùng k nhỏ (overfit) và k lớn (underfit)
        ax_cv.axvspan(-0.5, 1, alpha=0.07, color='red',   label='_Vùng overfit (k quá nhỏ)')
        ax_cv.axvspan(len(k_vals) - 2, len(k_vals) - 0.5,
                      alpha=0.07, color='orange', label='_Vùng underfit (k quá lớn)')
        ax_cv.text(0.3,  2, 'Overfit\n(k nhỏ)',  fontsize=8, color='red',    alpha=0.7, ha='center')
        ax_cv.text(len(k_vals) - 1.5, 2, 'Underfit\n(k lớn)', fontsize=8, color='darkorange', alpha=0.7, ha='center')

    plt.tight_layout()
    path2 = os.path.join(save_dir, 'knn_training_curves.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Đã lưu biểu đồ KNN: {path2}")
    return path1, path2


# ==========================================
# 10. Hàm Main — Điều Phối Toàn Bộ Pipeline
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print(f"  SO SÁNH 3 MÔ HÌNH — SMALL OBJECT DETECTION")
    print(f"  Thiết bị: {device}")
    print(f"{'='*60}")
    
    # --------------------------------------------------------
    # Bước 1: Chuẩn bị dataset phân loại (cho KNN + CNN Base)
    # --------------------------------------------------------
    train_class_dataset = ClassificationDataset(
        "dataset/train/images", "dataset/train/annotations.json", size=64)
    val_class_dataset = ClassificationDataset(
        "dataset/val/images", "dataset/val/annotations.json", size=64)
    
    # --------------------------------------------------------
    # MODEL 1: KNN + HOG
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("  MODEL 1: KNN + HOG")
    print("="*60)
    X_train_hog, y_train = extract_hog_dataset(train_class_dataset)
    X_val_hog,   y_val   = extract_hog_dataset(val_class_dataset)
    
    knn_metrics = {"train_acc": [], "val_acc": [], "epochs": list(range(1, 11))}

    # ----------------------------------------------------------------
    # CHỐNG OVERFITTING CHO KNN: Cross-Validation chọn k tối ưu
    # ----------------------------------------------------------------
    # KNN không có tham số học (lazy learner) nên không có gradient descent
    # → Không cần early stopping.
    # Thay vào đó dùng k-fold Cross-Validation để chọn k tốt nhất:
    #   - k nhỏ (k=1): bias thấp, variance cao → overfit trên train
    #   - k lớn (k=N): bias cao, variance thấp → underfit
    #   - Cross-validation tìm k cân bằng bias-variance tối ưu
    print("  Đang chọn k tối ưu bằng 5-fold Cross-Validation...")
    cv_k_candidates = list(range(1, 22, 2))  # k = 1, 3, 5, 7, ..., 21
    pipeline = Pipeline([
        ('scaler', StandardScaler()),          # Chuẩn hoá HOG features
        ('knn',    KNeighborsClassifier())
    ])
    param_grid = {'knn__n_neighbors': cv_k_candidates}
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipeline, param_grid,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train_hog, y_train)
    best_k   = grid_search.best_params_['knn__n_neighbors']
    best_cv_acc = grid_search.best_score_
    print(f"  → k tối ưu = {best_k} (CV accuracy trung bình = {best_cv_acc*100:.2f}%)")
    knn_metrics["best_k"]   = best_k
    knn_metrics["cv_score"] = best_cv_acc
    knn_metrics["anti_overfit_method"] = f"5-fold StratifiedCV trên k∈{cv_k_candidates}"
    # Lưu CV results để vẽ biểu đồ k-selection
    cv_results = grid_search.cv_results_
    knn_metrics["cv_k_values"]     = [p['knn__n_neighbors'] for p in cv_results['params']]
    knn_metrics["cv_mean_scores"]  = cv_results['mean_test_score'].tolist()
    knn_metrics["cv_std_scores"]   = cv_results['std_test_score'].tolist()

    # Giả lập "epoch" bằng cách tăng dần % dữ liệu — dùng best_k đã chọn
    num_samples = len(X_train_hog)
    # Dùng pipeline có StandardScaler để nhất quán với cross-validation
    scaler    = StandardScaler().fit(X_train_hog)
    X_tr_sc   = scaler.transform(X_train_hog)
    X_val_sc  = scaler.transform(X_val_hog)

    final_knn = None
    for pct in range(10, 110, 10):
        limit = max(best_k + 1, int(num_samples * (pct / 100.0)))  # cần ít nhất best_k+1 mẫu
        X_sub, y_sub = X_tr_sc[:limit], y_train[:limit]

        knn = KNeighborsClassifier(n_neighbors=best_k)
        knn.fit(X_sub, y_sub)

        train_acc = knn.score(X_sub, y_sub)
        val_acc   = knn.score(X_val_sc, y_val)

        knn_metrics["train_acc"].append(train_acc)
        knn_metrics["val_acc"].append(val_acc)
        print(f"  Simulated Epoch [{pct//10}/10] ({pct}% data, k={best_k}) "
              f"- Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%")
        final_knn = knn

    # Lưu cả scaler lẫn knn để app.py dùng đúng
    os.makedirs("models", exist_ok=True)
    knn_bundle = {'knn': final_knn, 'scaler': scaler, 'best_k': best_k}
    with open("models/knn_model.pkl", "wb") as f:
        pickle.dump(knn_bundle, f)
    print("  Đã lưu KNN model + scaler → models/knn_model.pkl")

    # Đánh giá KNN: Precision/Recall/F1 (dùng X_val đã scale)
    print("\n  Đánh giá chi tiết KNN...")
    # Truyền thẳng val_class_dataset nhưng dùng scaler đã fit
    knn_eval = evaluate_map_classifier(
        final_knn, val_class_dataset, device,
        model_type='knn', feature_scaler=scaler
    )
    knn_metrics["eval"] = knn_eval
    print(f"  KNN — Accuracy: {knn_eval['accuracy']*100:.2f}% | Precision: {knn_eval['precision']:.4f}"
          f" | Recall: {knn_eval['recall']:.4f} | F1: {knn_eval['f1']:.4f}")

    # Đo inference time KNN (bao gồm StandardScaler transform)
    sample_raw = X_val_hog[0].reshape(1, -1)
    sample_sc  = scaler.transform(sample_raw)
    knn_time_mean, knn_time_std = measure_inference_time_knn(final_knn, sample_sc)
    knn_metrics["inference_time_ms"]     = knn_time_mean
    knn_metrics["inference_time_std_ms"] = knn_time_std
    print(f"  KNN — Inference Time: {knn_time_mean:.3f} ± {knn_time_std:.3f} ms/patch")
    
    # --------------------------------------------------------
    # MODEL 2: CNN Base
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("  MODEL 2: CNN Base (Patch Classifier)")
    print("="*60)
    cnn_base_metrics, best_cnn_base_acc, best_cnn_base_loss, trained_cnn_base = train_cnn_base(
        train_class_dataset, val_class_dataset, device, num_epochs=10, batch_size=16, patience=3
    )
    
    # Đánh giá CNN Base: Precision/Recall/F1
    print("\n  Đánh giá chi tiết CNN Base...")
    cnn_base_eval = evaluate_map_classifier(trained_cnn_base, val_class_dataset, device, model_type='cnn')
    cnn_base_metrics["eval"] = cnn_base_eval
    print(f"  CNN Base — Accuracy: {cnn_base_eval['accuracy']*100:.2f}% | Precision: {cnn_base_eval['precision']:.4f}"
          f" | Recall: {cnn_base_eval['recall']:.4f} | F1: {cnn_base_eval['f1']:.4f}")
    
    # Đo inference time CNN Base
    trained_cnn_base.eval()
    sample_cnn = torch.randn(1, 3, 64, 64).to(device)
    cnn_base_time_mean, cnn_base_time_std = measure_inference_time_cnn(
        lambda x: trained_cnn_base(x), sample_cnn, device
    )
    cnn_base_metrics["inference_time_ms"] = cnn_base_time_mean
    cnn_base_metrics["inference_time_std_ms"] = cnn_base_time_std
    print(f"  CNN Base — Inference Time: {cnn_base_time_mean:.3f} ± {cnn_base_time_std:.3f} ms/patch")
    
    # --------------------------------------------------------
    # MODEL 3: CNN Multiscale (Faster R-CNN)
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("  MODEL 3: CNN Multiscale (Faster R-CNN + FPN + MultiScaleContextBlock)")
    print("="*60)
    cnn_multiscale_metrics, best_multiscale_loss, trained_multiscale = train_cnn_multiscale(
        device, num_epochs=10, batch_size=2, patience=5
    )
    
    # Đánh giá CNN Multiscale: mAP thực tế
    print("\n  Đánh giá mAP cho CNN Multiscale...")
    multiscale_map = evaluate_map_multiscale(
        trained_multiscale,
        ann_file="dataset/val/annotations.json",
        img_dir="dataset/val/images",
        device=device
    )
    cnn_multiscale_metrics["eval_map"] = multiscale_map
    if multiscale_map["mAP"] is not None:
        print(f"  CNN Multiscale — mAP@0.5:0.95: {multiscale_map['mAP']:.4f} | mAP@0.50: {multiscale_map['mAP_50']:.4f}"
              f" | mAP_small: {multiscale_map['mAP_small']:.4f}")

    # Đánh giá CNN Multiscale: Image-level Accuracy % (để so sánh trực tiếp với KNN/CNN-Base)
    print("\n  Tính Image-level Accuracy cho CNN Multiscale (has-car / no-car)...")
    multiscale_img_eval = evaluate_image_level_accuracy_multiscale(
        trained_multiscale,
        ann_file="dataset/val/annotations.json",
        img_dir="dataset/val/images",
        device=device,
        score_threshold=0.3
    )
    cnn_multiscale_metrics["eval_image_level"] = multiscale_img_eval
    if multiscale_img_eval["accuracy"] is not None:
        print(f"  CNN Multiscale — Image-level Accuracy: {multiscale_img_eval['accuracy']*100:.2f}%"
              f" | Precision: {multiscale_img_eval['precision']:.4f}"
              f" | Recall: {multiscale_img_eval['recall']:.4f} | F1: {multiscale_img_eval['f1']:.4f}")
    
    # --------------------------------------------------------
    # ĐÁNH GIÁ TRÊN TẬP TEST (chưa bao giờ dùng khi train/val)
    # --------------------------------------------------------
    print(f"\n{'='*60}")
    print("  ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST (Held-out Set)")
    print(f"{'='*60}")
    print("  ⚠️  Test set KHÔNG được dùng trong training hoặc early stopping.")
    print("  → Đây là kết quả khách quan nhất của từng mô hình.\n")

    TEST_IMG_DIR  = "dataset/test/images"
    TEST_ANN_FILE = "dataset/test/annotations.json"

    # --- Test: KNN ---
    test_class_dataset = ClassificationDataset(TEST_IMG_DIR, TEST_ANN_FILE, size=64)
    print("  KNN — Test set evaluation:")
    knn_test_eval = evaluate_map_classifier(
        final_knn, test_class_dataset, device,
        model_type='knn', feature_scaler=scaler
    )
    knn_metrics["test_eval"] = knn_test_eval
    print(f"    Accuracy: {knn_test_eval['accuracy']*100:.2f}% | "
          f"Precision: {knn_test_eval['precision']:.4f} | "
          f"Recall: {knn_test_eval['recall']:.4f} | "
          f"F1: {knn_test_eval['f1']:.4f}")

    # --- Test: CNN Base ---
    print("\n  CNN Base — Test set evaluation:")
    cnn_base_test_eval = evaluate_map_classifier(
        trained_cnn_base, test_class_dataset, device, model_type='cnn'
    )
    cnn_base_metrics["test_eval"] = cnn_base_test_eval
    print(f"    Accuracy: {cnn_base_test_eval['accuracy']*100:.2f}% | "
          f"Precision: {cnn_base_test_eval['precision']:.4f} | "
          f"Recall: {cnn_base_test_eval['recall']:.4f} | "
          f"F1: {cnn_base_test_eval['f1']:.4f}")

    # --- Test: CNN Multiscale — mAP trên test set ---
    print("\n  CNN Multiscale — Test set evaluation (mAP + Image-level Acc):")
    if os.path.exists(TEST_ANN_FILE):
        multiscale_test_map = evaluate_map_multiscale(
            trained_multiscale,
            ann_file=TEST_ANN_FILE,
            img_dir=TEST_IMG_DIR,
            device=device
        )
        multiscale_test_img = evaluate_image_level_accuracy_multiscale(
            trained_multiscale,
            ann_file=TEST_ANN_FILE,
            img_dir=TEST_IMG_DIR,
            device=device,
            score_threshold=0.3
        )
        cnn_multiscale_metrics["test_eval_map"]         = multiscale_test_map
        cnn_multiscale_metrics["test_eval_image_level"] = multiscale_test_img
        if multiscale_test_map["mAP"] is not None:
            print(f"    mAP@0.5:0.95 = {multiscale_test_map['mAP']:.4f} | "
                  f"mAP@0.50 = {multiscale_test_map['mAP_50']:.4f} | "
                  f"mAP_small = {multiscale_test_map['mAP_small']:.4f}")
        if multiscale_test_img["accuracy"] is not None:
            print(f"    Image-level Accuracy: {multiscale_test_img['accuracy']*100:.2f}% | "
                  f"F1: {multiscale_test_img['f1']:.4f}")
    else:
        print(f"    ⚠️  Không có {TEST_ANN_FILE}. Bỏ qua test set evaluation.")
        multiscale_test_map = multiscale_map   # fallback dùng val
        multiscale_test_img = multiscale_img_eval
    
    # Đo inference time CNN Multiscale
    trained_multiscale.eval()
    sample_det = [torch.randn(3, 800, 800).to(device)]
    det_time_mean, det_time_std = measure_inference_time_cnn(
        lambda x: trained_multiscale(x), sample_det, device
    )
    cnn_multiscale_metrics["inference_time_ms"] = det_time_mean
    cnn_multiscale_metrics["inference_time_std_ms"] = det_time_std
    print(f"  CNN Multiscale — Inference Time: {det_time_mean:.1f} ± {det_time_std:.1f} ms/image (800×800)")
    # --- Đánh giá CNN Multiscale trên các Patch test (để vẽ Confusion/Correlation) ---
    print("\n  Đánh giá CNN Multiscale trên Patch test set...")
    cnn_ms_test_patch = evaluate_multiscale_on_patches(
        trained_multiscale,
        test_class_dataset,
        device=device,
        score_threshold=0.3
    )

    # --- Vẽ các biểu đồ kết quả ---
    print("\n  Đang vẽ biểu đồ training curves...")
    plot_training_curves(cnn_base_metrics, cnn_multiscale_metrics, knn_metrics, save_dir="results")

    print("\n  Đang vẽ sơ đồ nhầm lẫn và tương quan...")
    plot_confusion_and_correlation(
        knn_test_eval,
        cnn_base_test_eval,
        cnn_ms_test_patch,
        multiscale_test_img,
        save_dir="results"
    )

    # --------------------------------------------------------
    # Lưu toàn bộ kết quả
    # --------------------------------------------------------
    results = {
        "knn":            knn_metrics,
        "cnn_base":       cnn_base_metrics,
        "cnn_multiscale": cnn_multiscale_metrics
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/comparison_results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n  Đã lưu toàn bộ kết quả → results/comparison_results.json")
    
    # --------------------------------------------------------
    # In bảng so sánh cuối cùng
    # --------------------------------------------------------
    print(f"\n{'='*80}")
    print("  BẢNG SO SÁNH MÔ HÌNH — KẾT QUẢ THỰC TẾ")
    print(f"{'='*80}")
    
    best_knn_acc = knn_metrics["val_acc"][-1] if knn_metrics["val_acc"] else 0.0
    best_k_used  = knn_metrics.get("best_k", 5)

    # Early-stop display
    base_stop     = cnn_base_metrics["early_stop_epoch"]
    base_stop_str = f"Epoch {base_stop}" if base_stop < 10 else "Không dừng sớm"

    multi_stop     = cnn_multiscale_metrics["early_stop_epoch"]
    multi_stop_str = f"Epoch {multi_stop}" if multi_stop < 10 else "Không dừng sớm"

    # Lấy image-level accuracy của CNN Multiscale (val)
    ms_img     = multiscale_img_eval
    ms_acc_str = f"{ms_img['accuracy']*100:.2f}%"  if ms_img["accuracy"]  is not None else "N/A"
    ms_pre_str = f"{ms_img['precision']:.4f}"       if ms_img["precision"] is not None else "N/A"
    ms_rec_str = f"{ms_img['recall']:.4f}"          if ms_img["recall"]    is not None else "N/A"
    ms_f1_str  = f"{ms_img['f1']:.4f}"              if ms_img["f1"]        is not None else "N/A"
    map_str    = f"{multiscale_map['mAP']:.4f}"     if multiscale_map["mAP"]    is not None else "N/A"
    map50_str  = f"{multiscale_map['mAP_50']:.4f}"  if multiscale_map["mAP_50"] is not None else "N/A"

    # Test set strings
    ms_test_acc = f"{multiscale_test_img['accuracy']*100:.2f}%" if multiscale_test_img["accuracy"] is not None else "N/A"
    ms_test_f1  = f"{multiscale_test_img['f1']:.4f}"            if multiscale_test_img["f1"]       is not None else "N/A"
    ms_test_map = f"{multiscale_test_map['mAP']:.4f}"           if multiscale_test_map["mAP"]      is not None else "N/A"
    ms_test_map50 = f"{multiscale_test_map['mAP_50']:.4f}"      if multiscale_test_map["mAP_50"]   is not None else "N/A"

    # ─── BẢNG 1: Vai trò từng split ────────────────────────────────────────
    print(f"""
  Phân chia dữ liệu (72 train / 20 val / 10 test ảnh):
  ┌─────────────────┬────────────────────────────────────────────────────────────┐
  │ Split           │ Vai trò trong từng mô hình                                 │
  ├─────────────────┼────────────────────────────────────────────────────────────┤
  │ Train (72 ảnh)  │ KNN: fit model  │ CNN Base: cập nhật weights               │
  │                 │ CNN Multiscale : tính loss + backward                      │
  ├─────────────────┼────────────────────────────────────────────────────────────┤
  │ Val   (20 ảnh)  │ KNN: Simulated Epoch tracking │ CNN Base: Early Stopping   │
  │                 │ CNN Multiscale: Early Stopping + chọn best model           │
  ├─────────────────┼────────────────────────────────────────────────────────────┤
  │ Test  (10 ảnh)  │ KNN/CNN Base/CNN Multiscale: Đánh giá CUỐI CÙNG           │
  │ (held-out)      │ KHÔNG bao giờ dùng trong quá trình training               │
  └─────────────────┴────────────────────────────────────────────────────────────┘
    """)

    # ─── BẢNG 2: Kết quả VAL (dùng để chọn model/hyperparameter) ──────────
    print(f"  {'─'*70}")
    print(f"  VAL SET RESULTS (dùng để monitor training, KHÔNG phải kết quả cuối)")
    print(f"  {'─'*70}")
    print(f"  {'Model':<22} {'Anti-Overfit':<30} {'Val Acc':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*82}")
    print(f"  {'KNN+HOG (k='+str(best_k_used)+')':<22}"
          f" {'5-fold CV chọn k tối ưu':<30}"
          f" {best_knn_acc*100:>7.2f}%"
          f" {knn_eval['precision']:>10.4f}"
          f" {knn_eval['recall']:>8.4f}"
          f" {knn_eval['f1']:>8.4f}")
    print(f"  {'CNN Base':<22}"
          f" {'Early Stopping (patience=3)':<30}"
          f" {best_cnn_base_acc*100:>7.2f}%"
          f" {cnn_base_eval['precision']:>10.4f}"
          f" {cnn_base_eval['recall']:>8.4f}"
          f" {cnn_base_eval['f1']:>8.4f}")
    print(f"  {'CNN Multiscale':<22}"
          f" {'Early Stopping (patience=5)':<30}"
          f" {ms_acc_str:>8}"
          f" {ms_pre_str:>10}"
          f" {ms_rec_str:>8}"
          f" {ms_f1_str:>8}")
    print(f"  (mAP@0.50 val = {map50_str}, mAP@0.5:0.95 val = {map_str})")

    # ─── BẢNG 3: Kết quả TEST (đánh giá cuối cùng, khách quan) ────────────
    print(f"\n  {'─'*70}")
    print(f"  TEST SET RESULTS ★ KẾT QUẢ CHÍNH THỨC ★ (held-out, chưa từng thấy)")
    print(f"  {'─'*70}")
    print(f"  {'Model':<22} {'Test Acc':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Inf.Time':>14}")
    print(f"  {'-'*75}")
    print(f"  {'KNN+HOG (k='+str(best_k_used)+')':<22}"
          f" {knn_test_eval['accuracy']*100:>8.2f}%"
          f" {knn_test_eval['precision']:>10.4f}"
          f" {knn_test_eval['recall']:>8.4f}"
          f" {knn_test_eval['f1']:>8.4f}"
          f" {knn_time_mean:>9.3f}ms/patch")
    print(f"  {'CNN Base':<22}"
          f" {cnn_base_test_eval['accuracy']*100:>8.2f}%"
          f" {cnn_base_test_eval['precision']:>10.4f}"
          f" {cnn_base_test_eval['recall']:>8.4f}"
          f" {cnn_base_test_eval['f1']:>8.4f}"
          f" {cnn_base_time_mean:>9.3f}ms/patch")
    print(f"  {'CNN Multiscale':<22}"
          f" {ms_test_acc:>9}"
          f"     F1={ms_test_f1}"
          f"     mAP@50={ms_test_map50}  mAP={ms_test_map}"
          f" {det_time_mean:>9.1f}ms/img")

    print(f"\n  ── Ghi Chú ──")
    print(f"  • KNN: chống overfit bằng 5-fold Cross-Validation → chọn k={best_k_used} (bias-variance balance).")
    print(f"  • CNN Base: Early Stopping theo val_loss (patience=3).")
    print(f"  • CNN Multiscale: Early Stopping theo val_loss (patience=5).")
    print(f"  • 'Val Acc' CNN Multiscale = Image-level accuracy (has-car/no-car),")
    print(f"    khác với mAP (đo chất lượng bounding box). Đây là 2 chỉ số bổ sung nhau.")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
