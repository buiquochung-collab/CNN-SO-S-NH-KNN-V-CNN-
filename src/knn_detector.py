import os
import cv2
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib

def extract_hog_features(image):
    """
    Trích xuất đặc trưng HOG (Histogram of Oriented Gradients) từ ảnh.
    Đây là phương pháp kinh điển thường đi kèm với KNN / SVM.
    """
    hog = cv2.HOGDescriptor()
    # Chuyển ảnh về kích thước cố định (64x128 cho HOG chuẩn)
    resized_img = cv2.resize(image, (64, 128))
    if len(resized_img.shape) == 3:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    features = hog.compute(resized_img)
    return features.flatten()

def train_knn_baseline(data_dir, annotation_file, save_path="models/knn_model.pkl"):
    """
    Huấn luyện mô hình KNN Baseline để so sánh với Deep Learning.
    Lưu ý: KNN chủ yếu dùng cho Classification. Ở đây ta cắt các Bounding Box
    để trích xuất đặc trưng và train KNN Classifier.
    """
    print("Dang doc du lieu cho KNN...")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    X = []
    y = []

    for img_info in annotations.get('images', []):
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_path = os.path.join(data_dir, file_name)
        
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Tìm các annotation cho ảnh này
        img_anns = [ann for ann in annotations.get('annotations', []) if ann['image_id'] == img_id]
        
        for ann in img_anns:
            bbox = ann['bbox'] # [x, y, width, height]
            x, y_coord, w, h = [int(v) for v in bbox]
            
            # Cắt ảnh theo bounding box
            crop = img[y_coord:y_coord+h, x:x+w]
            if crop.size == 0:
                continue
                
            features = extract_hog_features(crop)
            X.append(features)
            y.append(ann['category_id'])

    if len(X) == 0:
        print("Khong tim thay du lieu bbox hop le de train KNN.")
        return

    X = np.array(X)
    y = np.array(y)

    print(f"Bat dau huan luyen KNN voi {len(X)} mau...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X, y)

    # Đánh giá nhanh trên tập train
    preds = knn.predict(X)
    print("Bao cao phan loai KNN tren tap huan luyen:")
    print(classification_report(y, preds))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(knn, save_path)
    print(f"Da luu mo hinh KNN tai {save_path}")

if __name__ == "__main__":
    train_img_dir = "dataset/train/images"
    train_ann_file = "dataset/train/annotations.json"
    if os.path.exists(train_ann_file):
        train_knn_baseline(train_img_dir, train_ann_file)
    else:
        print("Khong tim thay annotations.json, vui long cau hinh dung duong dan.")
