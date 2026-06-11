import os
import time
import pickle
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
from flask import Flask, request, jsonify, render_template
from PIL import Image
from skimage.feature import hog

from src.model import get_multi_scale_detector
from compare_models import CNNBase

app = Flask(__name__)

# ============================================================
# Cấu hình thiết bị
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Đường dẫn trọng số
# ============================================================
KNN_PATH            = "models/knn_model.pkl"
CNN_BASE_PATH       = "models/cnn_base_best.pth"
CNN_MULTISCALE_PATH = "models/best_model.pth"

# Số lượng lớp (đồng bộ toàn dự án)
NUM_CLASSES = 2

# Biến global chứa các model đã nạp
knn_model            = None   # KNNClassifier object
knn_scaler           = None   # StandardScaler (lưu cùng bundle với KNN)
knn_best_k           = 5      # k tối ưu chọn bằng cross-validation
cnn_base_model       = None
cnn_multiscale_model = None

def load_models():
    """Nạp cả 3 mô hình từ file .pkl/.pth vào bộ nhớ."""
    global knn_model, knn_scaler, knn_best_k, cnn_base_model, cnn_multiscale_model

    # 1. Load KNN bundle {knn, scaler, best_k}
    if os.path.exists(KNN_PATH):
        try:
            with open(KNN_PATH, 'rb') as f:
                bundle = pickle.load(f)
            # Hỗ trợ cả format cũ (chỉ lưu KNN) lẫn format mới (bundle dict)
            if isinstance(bundle, dict):
                knn_model  = bundle['knn']
                knn_scaler = bundle.get('scaler', None)
                knn_best_k = bundle.get('best_k', 5)
                print(f"✅ Loaded KNN (k={knn_best_k}, scaler={'có' if knn_scaler else 'không'}).")
            else:
                knn_model  = bundle   # format cũ
                knn_scaler = None
                knn_best_k = 5
                print("✅ Loaded KNN model (format cũ, không có scaler).")
        except Exception as e:
            print(f"❌ Error loading KNN: {e}")
    else:
        print(f"⚠️  KNN model chưa có tại {KNN_PATH}. Hãy chạy compare_models.py trước.")
            
    # 2. Load CNN Base
    if os.path.exists(CNN_BASE_PATH):
        try:
            cnn_base_model = CNNBase(num_classes=NUM_CLASSES)
            cnn_base_model.load_state_dict(
                torch.load(CNN_BASE_PATH, map_location=DEVICE, weights_only=True))
            cnn_base_model.to(DEVICE)
            cnn_base_model.eval()
            print("✅ Loaded CNN Base model.")
        except Exception as e:
            print(f"❌ Error loading CNN Base: {e}")
    else:
        print(f"⚠️  CNN Base chưa có tại {CNN_BASE_PATH}. Hãy chạy compare_models.py trước.")
            
    # 3. Load CNN Multiscale
    # FIX: Đọc backbone_name từ checkpoint (được lưu bởi train.py/compare_models.py)
    # để đảm bảo load_state_dict đúng kiến trúc
    if os.path.exists(CNN_MULTISCALE_PATH):
        try:
            checkpoint    = torch.load(CNN_MULTISCALE_PATH, map_location=DEVICE, weights_only=True)
            backbone_name = checkpoint.get('backbone_name', 'resnet50')  # FIX: đọc từ checkpoint
            
            cnn_multiscale_model = get_multi_scale_detector(
                num_classes=NUM_CLASSES, backbone_name=backbone_name)
            cnn_multiscale_model.load_state_dict(checkpoint['model_state_dict'])
            cnn_multiscale_model.to(DEVICE)
            cnn_multiscale_model.eval()
            print(f"✅ Loaded CNN Multiscale model (backbone: {backbone_name}).")
        except Exception as e:
            print(f"❌ Error loading CNN Multiscale: {e}")
    else:
        print(f"⚠️  CNN Multiscale chưa có tại {CNN_MULTISCALE_PATH}. Hãy chạy compare_models.py trước.")

# Nạp mô hình khi khởi động ứng dụng
load_models()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def status():
    """Trả về trạng thái của các mô hình (đã load hay chưa)."""
    return jsonify({
        "models": {
            "knn":            knn_model is not None,
            "cnn_base":       cnn_base_model is not None,
            "cnn_multiscale": cnn_multiscale_model is not None,
        },
        "device": str(DEVICE)
    })

@app.route("/api/results", methods=["GET"])
def get_results():
    """Trả về kết quả so sánh từ file JSON (sau khi chạy compare_models.py)."""
    results_path = "results/comparison_results.json"
    if os.path.exists(results_path):
        with open(results_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({
            "error": "Chưa có kết quả. Vui lòng chạy compare_models.py trước."
        }), 404


@app.route("/results/<path:filename>", methods=["GET"])
def get_result_image(filename):
    """Serve các hình ảnh biểu đồ kết quả (training curves, confusion matrix, correlation)."""
    from flask import send_from_directory
    return send_from_directory("results", filename)

@app.route("/api/predict", methods=["POST"])
def predict():
    """Chạy inference trên ảnh upload, trả về kết quả của cả 3 mô hình."""
    global knn_model, cnn_base_model, cnn_multiscale_model
    
    # Reload nếu model chưa được nạp
    if knn_model is None or cnn_base_model is None or cnn_multiscale_model is None:
        load_models()
        
    if "image" not in request.files:
        return jsonify({"error": "Không có tệp hình ảnh được gửi lên."}), 400
        
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Không có file nào được chọn."}), 400
        
    # Đọc ảnh
    try:
        pil_image = Image.open(file.stream).convert("RGB")
        image_np  = np.array(pil_image)
    except Exception as e:
        return jsonify({"error": f"Lỗi đọc ảnh: {str(e)}"}), 400
        
    # Kiểm tra model đã sẵn sàng chưa
    models_missing = []
    if knn_model is None:            models_missing.append("KNN")
    if cnn_base_model is None:       models_missing.append("CNN Base")
    if cnn_multiscale_model is None: models_missing.append("CNN Multiscale")
    
    if models_missing:
        return jsonify({
            "error":   "Một số model chưa được train. Hãy chạy compare_models.py trước.",
            "missing": models_missing
        }), 503

    results = {}
    
    # ------------------------------------------
    # 1. Dự đoán với KNN + HOG
    # ------------------------------------------
    try:
        t0      = time.perf_counter()
        img_64  = cv2.resize(image_np, (64, 64))
        gray    = cv2.cvtColor(img_64, cv2.COLOR_RGB2GRAY)
        hog_feat = hog(gray, orientations=8, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)
        hog_feat = hog_feat.reshape(1, -1)

        # Áp dụng StandardScaler nếu bundle có scaler (đồng bộ với training)
        if knn_scaler is not None:
            hog_feat = knn_scaler.transform(hog_feat)

        pred_class  = int(knn_model.predict(hog_feat)[0])
        pred_prob   = knn_model.predict_proba(hog_feat)[0]
        confidence  = float(pred_prob[pred_class])
        inf_time_ms = (time.perf_counter() - t0) * 1000

        results["knn"] = {
            "class":        "Car" if pred_class == 1 else "Background",
            "confidence":   round(confidence, 4),
            "inference_ms": round(inf_time_ms, 3),
            "k_used":       knn_best_k
        }
    except Exception as e:
        results["knn"] = {"error": f"Lỗi dự đoán: {str(e)}"}

    # ------------------------------------------
    # 2. Dự đoán với CNN Base
    # ------------------------------------------
    try:
        t0         = time.perf_counter()
        img_64     = cv2.resize(image_np, (64, 64))
        img_tensor = torch.from_numpy(img_64).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs       = cnn_base_model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            pred_class    = int(torch.argmax(probabilities).item())
            confidence    = float(probabilities[pred_class].item())
            
        inf_time_ms = (time.perf_counter() - t0) * 1000
        
        results["cnn_base"] = {
            "class":        "Car" if pred_class == 1 else "Background",
            "confidence":   round(confidence, 4),
            "inference_ms": round(inf_time_ms, 3)
        }
    except Exception as e:
        results["cnn_base"] = {"error": f"Lỗi dự đoán: {str(e)}"}

    # ------------------------------------------
    # 3. Dự đoán với CNN Multiscale (Faster R-CNN)
    # ------------------------------------------
    try:
        t0         = time.perf_counter()
        img_tensor = F.to_tensor(image_np).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            prediction = cnn_multiscale_model(img_tensor)[0]
            
        inf_time_ms = (time.perf_counter() - t0) * 1000
            
        scores = prediction['scores'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        boxes  = prediction['boxes'].cpu().numpy()
        
        valid_indices = np.where((scores > 0.3) & (labels == 1))[0]
        
        if len(valid_indices) > 0:
            best_idx   = valid_indices[np.argmax(scores[valid_indices])]
            confidence = float(scores[best_idx])
            best_box   = boxes[best_idx].tolist()
            results["cnn_multiscale"] = {
                "class":        "Car",
                "confidence":   round(confidence, 4),
                "box_count":    int(len(valid_indices)),
                "best_box":     [round(v, 1) for v in best_box],
                "inference_ms": round(inf_time_ms, 1)
            }
        else:
            max_score = float(scores[np.argmax(scores)]) if len(scores) > 0 else 0.0
            results["cnn_multiscale"] = {
                "class":        "Background",
                "confidence":   round(1.0 - max_score, 4),
                "box_count":    0,
                "best_box":     None,
                "inference_ms": round(inf_time_ms, 1)
            }
    except Exception as e:
        results["cnn_multiscale"] = {"error": f"Lỗi dự đoán: {str(e)}"}

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
