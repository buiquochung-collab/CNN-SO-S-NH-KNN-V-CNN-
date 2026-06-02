# Phân tích Chi tiết Repository CNN (Object Detection)

Tài liệu này cung cấp một cái nhìn toàn diện, phân tích chi tiết từng thành phần trong repository [fatqy/cnn](https://github.com/fatqy/cnn), không bỏ sót bất kỳ tệp tin nào.

## 1. Tổng quan Dự án (Overview)
Dự án này là một hệ thống **Nhận dạng Vật thể nhỏ (Small Object Detection)** sử dụng PyTorch. Mô hình được thiết kế đặc biệt cho các bài toán nhận diện các vật thể có kích thước nhỏ (như con người, xe cộ từ góc nhìn trên cao trong tập dữ liệu VisDrone). 

Dự án áp dụng các kỹ thuật tiên tiến:
- **Multi-scale CNN & FPN**: Kết hợp đặc trưng ở nhiều tỉ lệ khác nhau.
- **Backbone**: Hỗ trợ `ResNet50` và `MobileNetV2`.
- **Data Augmentation**: Sử dụng `Albumentations`.
- **Quản lý huấn luyện**: Tích hợp Early Stopping, Model Checkpointing và TensorBoard.

---

## 2. Cấu trúc Thư mục (Directory Structure)
```text
cnn/
├── dataset/                     # Thư mục chứa dữ liệu
│   ├── train/                   # Tập train
│   ├── val/                     # Tập validation
│   └── test/                    # Tập test
├── src/                         # Mã nguồn cốt lõi
│   ├── dataset.py               # Lớp xử lý dữ liệu (Dataset)
│   ├── model.py                 # Định nghĩa mạng Multi-scale CNN & FPN
│   └── utils.py                 # Hàm vẽ hộp (box), tính toán, lưu checkpoint
├── README.md                    # Tài liệu hướng dẫn sử dụng gốc
├── GEMINI.md                    # Tài liệu ngữ cảnh cho AI Agent
├── requirements.txt             # Các thư viện phụ thuộc
├── train.py                     # Script chạy huấn luyện
├── test.py                      # Script đánh giá mô hình
└── predict.py                   # Script dự đoán trên một ảnh
```

---

## 3. Phân tích chi tiết Mã nguồn (Source Code Analysis)

### 3.1. Thư mục `src/` (Core Source Code)

#### 3.1.1. `src/model.py` (Kiến trúc Mô hình)
Tệp này đóng vai trò quan trọng nhất, định nghĩa kiến trúc mạng.
- **`MultiScaleContextBlock`**: Một khối mạng tự định nghĩa (Custom block) giúp trích xuất đặc trưng đa tỉ lệ bằng 4 nhánh:
  - Nhánh 1: Convolution 1x1.
  - Nhánh 2: Convolution 1x1 $\rightarrow$ Convolution 3x3.
  - Nhánh 3: Convolution 1x1 $\rightarrow$ Convolution 3x3 $\rightarrow$ Convolution 3x3 (Mô phỏng 5x5).
  - Nhánh 4: MaxPooling 3x3 $\rightarrow$ Convolution 1x1.
  - Kết quả từ 4 nhánh được gộp lại (`torch.cat`) và đi qua một lớp Feature Fusion 1x1 Conv.
- **`CustomBackboneWithFPN`**: Tích hợp `MultiScaleContextBlock` vào các Backbone chuẩn (ResNet50 hoặc MobileNetV2), sau đó đưa qua `FeaturePyramidNetwork` (FPN) để xuất ra các đặc trưng ở nhiều mức độ (C2, C3, C4, C5).
- **`get_multi_scale_detector`**: 
  - Khởi tạo kiến trúc `FasterRCNN` của Torchvision.
  - **Điểm nhấn đặc biệt:** Cấu hình `AnchorGenerator` với kích thước siêu nhỏ `(8, 16, 32, 64)` và tỉ lệ `(0.5, 1.0, 2.0)` để mô hình nhạy bén tối đa với các đối tượng có kích thước chỉ vài pixel.

#### 3.1.2. `src/dataset.py` (Xử lý Dữ liệu)
Đảm nhiệm việc đọc và chuẩn bị dữ liệu cho quá trình huấn luyện.
- **`CustomObjectDetectionDataset`**: Kế thừa `torch.utils.data.Dataset`.
  - Đọc file JSON theo chuẩn **COCO format**.
  - Chuyển đổi định dạng bounding box từ COCO `[x_min, y_min, width, height]` sang Pascal VOC `[x_min, y_min, x_max, y_max]` theo yêu cầu của PyTorch Faster R-CNN.
- **Data Augmentation (`get_transforms`)**: Sử dụng thư viện `albumentations` cho các biến đổi mạnh mẽ:
  - Train: Resize(800x800), HorizontalFlip(0.5), Rotate(15 độ, 0.3), GaussianBlur(0.2), ColorJitter(0.3), Normalize.
  - Test/Val: Resize(800x800), Normalize.
- **`collate_fn`**: Hàm để DataLoader xử lý đúng các batch có số lượng bounding boxes khác nhau trên mỗi ảnh.

#### 3.1.3. `src/utils.py` (Các hàm Tiện ích)
- **`draw_bounding_boxes`**: Vẽ hộp dự đoán lên ảnh bằng OpenCV, kèm theo nhãn và độ tin cậy (score). Tự động denormalize ảnh nếu ảnh ở dạng Tensor được chuẩn hóa ImageNet.
- **`plot_loss_metrics`**: Vẽ và lưu biểu đồ Loss trong quá trình huấn luyện bằng `matplotlib`.
- **`save_checkpoint` & `load_checkpoint`**: Hàm để lưu/tải trạng thái mô hình (model, optimizer, epoch, loss) vào các file `.pth`.

---

### 3.2. Script Thực thi (Execution Scripts)

#### 3.2.1. `train.py` (Huấn luyện Mô hình)
Script chính để bắt đầu quá trình Training.
- **Siêu tham số mặc định**: `NUM_CLASSES = 2` (Background + Car), `BATCH_SIZE = 2`, `EPOCHS = 10`, `LR = 0.001`.
- **Hệ thống Optimizer & Scheduler**: Dùng `Adam` kết hợp với `StepLR` (giảm LR đi 10 lần sau mỗi 10 epoch).
- **Tính năng nổi bật**:
  - Tích hợp **TensorBoard** (`SummaryWriter`) để theo dõi biểu đồ Loss trực quan qua thư mục `runs/`.
  - **Early Stopping**: Dừng sớm nếu loss không cải thiện sau 5 epoch liên tiếp (`patience = 5`).
  - Lưu checkpoint mô hình tốt nhất (`best_model.pth`) và lưu tự động sau mỗi 5 epoch (`model_epoch_X.pth`).

#### 3.2.2. `test.py` (Đánh giá Mô hình)
Được sử dụng để chạy thử trên tập Test và báo cáo kết quả.
- **`test_single_image`**: Lấy 1 ảnh từ `dataset/test/images/`, truyền qua mô hình đã huấn luyện, vẽ kết quả bounding boxes và hiển thị thông qua `matplotlib`.
- **Đánh giá hiệu năng**: Tạo dữ liệu giả lập để vẽ Confusion Matrix bằng `seaborn` và tính các chỉ số Precision, Recall, F1-score bằng `sklearn.metrics`. Trong thực tế, cần lặp qua toàn bộ Validation Dataset để lấy y_true và y_pred chính xác.

#### 3.2.3. `predict.py` (Dự đoán Độc lập)
Một công cụ CLI tiện dụng để nhận diện vật thể trên bất kỳ ảnh nào.
- Hỗ trợ gọi từ dòng lệnh: `python predict.py <ảnh> <ngưỡng_threshold>`.
- Đọc trọng số từ `models/best_model.pth`.
- Đếm số lượng vật thể tìm thấy với độ tự tin (score) lớn hơn `threshold` (mặc định 0.3).
- Lưu kết quả ra file `result_prediction.png`.

---

## 4. Cấu hình và Môi trường (Environment & Configs)

### 4.1. `requirements.txt`
Chứa toàn bộ các thư viện thiết yếu để khởi chạy project:
- PyTorch stack: `torch>=2.0.0`, `torchvision>=0.15.0`.
- Computer Vision: `opencv-python`, `albumentations`, `pycocotools`.
- Phân tích & Trực quan: `numpy`, `matplotlib`, `pandas`, `scikit-learn`.
- Tooling: `tensorboard`, `tqdm`.
- (Bonus): `ultralytics` cho việc đối chiếu với YOLOv8 (mặc dù file `bonus_yolo_faster_rcnn.py` nhắc tới trong README dường như không có trong repo hiện tại).

### 4.2. `README.md` & `GEMINI.md`
- **README.md**: Hướng dẫn chi tiết cách cài đặt Conda environment, định dạng cấu trúc JSON COCO Dataset bắt buộc, và các lệnh chạy. Có nhắc đến file bonus YOLOv8.
- **GEMINI.md**: Bản tóm tắt cho AI agent hiểu ngữ cảnh, tuân thủ PEP8, sử dụng logging (dù mã nguồn thực tế đang dùng `print`), viết test case trước.

---

## 5. Nhận xét & Đánh giá Kiến trúc

- **Ưu điểm:**
  - Kiến trúc rất tối ưu cho các vật thể nhỏ nhờ áp dụng **Anchor Boxes kích thước cực nhỏ (8, 16, 32, 64)** kết hợp với Feature Pyramid Network (FPN) và MultiScaleContextBlock. 
  - Code được cấu trúc modular rất gọn gàng và dễ đọc (tách biệt src/, root scripts).
  - Tích hợp Albumentations là một điểm cộng rất lớn so với `torchvision.transforms` truyền thống trong Object Detection, vì nó xử lý biến đổi bounding boxes đồng bộ với ảnh cực kỳ hiệu quả.
- **Điểm có thể cải thiện:**
  - Chưa sử dụng `logging` chuẩn như đã hứa trong `GEMINI.md` (vẫn đang dùng `print()`).
  - Trong `test.py`, ma trận nhầm lẫn (Confusion Matrix) và F1-score vẫn đang dùng dữ liệu mảng giả lập (`y_true = [...]`), chưa code logic mAP hoặc vòng lặp tính thực tế toàn bộ test set.
  - Thiếu script/thư mục YOLO baseline (`bonus_yolo_faster_rcnn.py` không hiện diện trong list folder).
