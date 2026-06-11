Project hiện có:
- src/model.py: mô hình CNN Multiscale + FPN (dùng ResNet50 backbone, MultiScaleContextBlock, FPN)
- src/dataset.py: CustomObjectDetectionDataset (COCO format)
- src/utils.py: save_checkpoint, plot_loss_metrics
- train.py: huấn luyện với Early Stopping, TensorBoard, Adam optimizer
- requirements.txt: torch, torchvision, scikit-learn, albumentations, pycocotools...

Hãy viết file compare_models.py thực hiện:

1. KNN (scikit-learn):
   - Dùng HOG features (skimage) để trích xuất đặc trưng từ ảnh resize về 64x64
   - KNeighborsClassifier(n_neighbors=5)
   - Vì KNN không có loss/accuracy theo epoch, hãy simulate bằng cách train trên các tập con tăng dần (10%, 20%, ..., 100%) để tạo "learning curve" giả lập epoch
   - Tính train_acc, val_acc tại mỗi bước

2. CNN Base:
   - Kiến trúc đơn giản: Conv2d(3,32,3) -> ReLU -> MaxPool -> Conv2d(32,64,3) -> ReLU -> MaxPool -> Flatten -> Linear(64*14*14, 256) -> ReLU -> Linear(256, num_classes)
   - Input ảnh resize 64x64
   - Dùng CrossEntropyLoss + Adam(lr=0.001)
   - Train 10 epochs, batch_size=16
   - Ghi lại train_loss, val_loss, train_acc, val_acc mỗi epoch
   - Early Stopping patience=3

3. CNN Multiscale (từ src/model.py hiện có):
   - Dùng get_multi_scale_detector(num_classes, backbone_name='resnet50')
   - Train tương tự CNN Base nhưng với FasterRCNN loss dict
   - Ghi lại train_loss, val_loss mỗi epoch
   - Early Stopping patience=5

4. Lưu kết quả:
   Sau khi train xong 3 mô hình, lưu toàn bộ metrics vào file results/comparison_results.json với cấu trúc:
   {
     "knn": {"train_acc": [...], "val_acc": [...], "epochs": [...]},
     "cnn_base": {"train_loss": [...], "val_loss": [...], "train_acc": [...], "val_acc": [...], "early_stop_epoch": int},
     "cnn_multiscale": {"train_loss": [...], "val_loss": [...], "early_stop_epoch": int}
   }

5. In bảng so sánh cuối cùng ra console:
   | Model         | Best Val Acc | Best Val Loss | Early Stop Epoch |
   |---------------|--------------|---------------|------------------|
   | KNN           | xx%          | N/A           | N/A              |
   | CNN Base      | xx%          | x.xxxx        | epoch N          |
   | CNN Multiscale| N/A          | x.xxxx        | epoch N          |

Yêu cầu code: clean, có comment tiếng Việt, tương thích với codebase hiện có, không thay đổi src/model.py, src/dataset.py.
Hãy tạo web demo gồm 2 file: app.py (Flask backend) và templates/index.html (frontend).

=== app.py ===
- Route GET "/" -> render index.html
- Route GET "/api/results" -> đọc results/comparison_results.json và trả về JSON
- Route POST "/api/predict" -> nhận file ảnh upload, chạy predict qua cả 3 mô hình:
  + KNN: load models/knn_model.pkl (pickle), trích HOG features rồi predict
  + CNN Base: load models/cnn_base_best.pth, forward pass, lấy top-1 class + confidence
  + CNN Multiscale: load models/best_model.pth, forward pass inference mode
  Trả về JSON: {"knn": {"class": "...", "confidence": 0.xx}, "cnn_base": {...}, "cnn_multiscale": {...}}
- Dùng Flask, torch, PIL, numpy, pickle, skimage

=== templates/index.html ===
Thiết kế dark theme chuyên nghiệp (màu nền #0f1117, accent #00d4ff), font Rajdhani + Inter từ Google Fonts.

PHẦN 1 - "Training Dashboard" (chiếm 60% trang):
- Tiêu đề lớn "📊 Model Training Comparison"
- 3 tabs: [Loss Curve] [Accuracy Curve] [Early Stopping]
- Tab Loss Curve: Chart.js line chart hiển thị train_loss + val_loss của CNN Base và CNN Multiscale theo epoch. Mỗi model một màu khác nhau (CNN Base: #ff6b6b, CNN Multiscale: #00d4ff). Legend rõ ràng.
- Tab Accuracy Curve: Chart.js line chart hiển thị train_acc + val_acc của KNN (learning curve) và CNN Base theo epoch/step. KNN: #ffd93d, CNN Base: #6bcb77.
- Tab Early Stopping: Hiển thị 3 card ngang nhau, mỗi card là 1 mô hình:
  + KNN card: "No Early Stopping" + final accuracy badge
  + CNN Base card: epoch dừng, best val loss, best val acc, badge màu xanh "STOPPED AT EPOCH N"
  + CNN Multiscale card: tương tự, badge màu cam
  Bên dưới mỗi card là progress bar thể hiện patience đã dùng (patience_counter / patience * 100%)

PHẦN 2 - "Live Prediction" (chiếm 40% trang):
- Tiêu đề "🔍 Upload & Predict"
- Vùng drag-and-drop upload ảnh (dashed border, click để chọn file, preview ảnh sau khi chọn)
- Nút "Run All Models" màu gradient #00d4ff -> #7b2ff7
- Kết quả hiển thị dưới dạng 3 card song song:
  + Card KNN: icon 🔵, tên class dự đoán to, confidence bar
  + Card CNN Base: icon 🟢, tương tự
  + Card CNN Multiscale: icon 🟠, tương tự + highlight "⭐ Best Model"
- Loading spinner khi đang predict
- Nếu chưa có model file (.pkl/.pth), hiển thị warning "Model chưa được train, hãy chạy compare_models.py trước"

Dữ liệu cho charts được fetch từ /api/results khi trang load.
Nếu file JSON chưa tồn tại, hiển thị placeholder data giả lập (10 epochs) để demo UI.
Toàn bộ JS viết vanilla (không dùng framework), CSS inline trong file HTML.

https://github.com/buiquochung-collab/CNN-SO-S-NH-KNN-V-CNN-