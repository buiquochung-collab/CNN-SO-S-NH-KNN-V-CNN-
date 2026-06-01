# Đồ án: Nhận dạng Vật thể nhỏ với Multi-scale CNN & FPN

Đây là một project Object Detection hoàn chỉnh sử dụng PyTorch, được thiết kế đặc biệt cho bài toán nhận dạng các vật thể nhỏ (Small Object Detection) như con người, xe cộ từ góc nhìn trên cao (ví dụ: VisDrone dataset).

## Tính năng nổi bật

- **Backbone tuỳ chỉnh:** Tích hợp `MultiScaleContextBlock` với các nhánh Convolution kích thước khác nhau (1x1, 3x3, 5x5, MaxPooling) kết hợp cùng ResNet/MobileNet.
- **FPN (Feature Pyramid Network):** Trích xuất và hợp nhất đặc trưng ở nhiều tỉ lệ giúp mô hình nhạy bén hơn với các vật thể nhỏ.
- **Data Augmentation mạnh mẽ:** Sử dụng `Albumentations` để xử lý ảnh và Bounding Box (Flip, Rotation, Blur, Color Jitter).
- **Hệ thống huấn luyện chuẩn chỉnh:** Tích hợp TensorBoard, Early Stopping, Checkpointing, và Custom DataLoader.
- **Bonus YOLOv8/FasterRCNN Baseline:** Mã nguồn so sánh với các mô hình SOTA.

## Cấu trúc thư mục
```text
cnn_dl/
│
├── dataset/                     # Thư mục chứa dữ liệu
│   ├── train/                   # Ảnh và annotation cho tập train
│   ├── val/                     # Tập validation
│   └── test/                    # Tập test
│
├── models/                      # Thư mục lưu Checkpoint (.pth)
├── runs/                        # Log của TensorBoard
│
├── src/
│   ├── dataset.py               # Xử lý data (COCO format -> PyTorch Dataset)
│   ├── model.py                 # Định nghĩa mạng Multi-scale CNN và FPN
│   └── utils.py                 # Các hàm vẽ box, tính mAP, lưu checkpoint
│
├── train.py                     # Script huấn luyện chính
├── test.py                      # Script đánh giá và hiển thị kết quả
├── bonus_yolo_faster_rcnn.py    # Code huấn luyện YOLOv8 (Bonus)
└── requirements.txt             # Các thư viện cần thiết
```

## Hướng dẫn cài đặt

Dự án yêu cầu Python >= 3.8 và GPU có hỗ trợ CUDA.

```bash
# 1. Tạo môi trường ảo (Khuyến khích dùng Conda)
conda create -n cnn_dl python=3.9
conda activate cnn_dl

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

## Chuẩn bị Dataset

Dataset cần được chuyển về định dạng **COCO JSON format**. 
Bạn có thể tải VisDrone Dataset hoặc COCO Dataset, sau đó đặt vào thư mục `dataset/train/` và cập nhật đường dẫn trong `train.py`.

Cấu trúc file `annotations.json` yêu cầu:
```json
{
  "images": [{"id": 1, "file_name": "img1.jpg", "width": 800, "height": 800}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x_min, y_min, width, height], "area": 1500, "iscrowd": 0}]
}
```

## Hướng dẫn sử dụng

### 1. Huấn luyện (Training)
Chạy script sau để bắt đầu huấn luyện. Hãy chắc chắn bạn đã uncomment phần DataLoader trong file `train.py` sau khi thêm data.

```bash
python train.py
```

Xem biểu đồ Loss bằng Tensorboard:
```bash
tensorboard --logdir=runs/
```

### 2. Kiểm thử (Testing)
Chạy file test để dự đoán trên ảnh mới và hiển thị Bounding Box:

```bash
python test.py
```

### 3. Bonus: Chạy YOLOv8 để so sánh
```bash
python bonus_yolo_faster_rcnn.py
```
Đoạn code trong file cung cấp template để gọi Ultralytics YOLOv8 nhằm đo lường hiệu năng (mAP) so với mô hình Custom CNN.

---
*Đồ án mẫu được thiết kế chuẩn chuyên nghiệp, mã nguồn sạch sẽ, chú thích rõ ràng, sẵn sàng để báo cáo hoặc trình bày trên Jupyter Notebook/Google Colab.*
