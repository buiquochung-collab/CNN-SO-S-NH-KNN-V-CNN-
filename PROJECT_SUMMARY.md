# Project Summary: Multi-scale CNN vs Baseline Models

## Tổng quan dự án (Overview)
Dự án này là một hệ thống Deep Learning hoàn chỉnh tập trung vào việc nghiên cứu, triển khai và đánh giá hiệu năng của mạng **Multi-scale CNN (dựa trên Faster R-CNN với Feature Pyramid Network & Context Block)** so với các mô hình cơ sở (Baselines).

Dự án đặc biệt nhắm tới việc giải quyết bài toán phát hiện vật thể (Object Detection) siêu nhỏ (ví dụ: ảnh vệ tinh) thông qua các lớp trích xuất đặc trưng đa tỉ lệ (Multi-scale Feature Extraction).

---

## Kiến trúc Hệ thống (System Architecture)

Hệ thống được thiết kế thành 3 thành phần chính theo chuẩn công nghiệp:

### 1. Model Training & Inference (Python/PyTorch)
- **Mô hình đề xuất (Multi-scale CNN):** Faster R-CNN kết hợp với `CustomBackboneWithFPN` và `MultiScaleContextBlock`. Anchor generator được tinh chỉnh để bắt các vật thể có kích thước cực nhỏ (8, 16, 32, 64 pixels).
- **Mô hình so sánh 1 (Standard CNN):** Faster R-CNN nguyên thủy với backbone ResNet50 để so sánh mức độ hiệu quả của việc thêm Context Block.
- **Mô hình so sánh 2 (KNN Baseline):** Phương pháp cổ điển sử dụng trích xuất đặc trưng **HOG (Histogram of Oriented Gradients)** kết hợp bộ phân loại **K-Nearest Neighbors (KNN)**.

### 2. FastAPI Backend (`api.py`)
- Đóng vai trò là cầu nối (Middleware) giữa Model và giao diện người dùng.
- Load các mô hình PyTorch vào VRAM/RAM.
- Cung cấp API `GET /api/metrics` đọc kết quả so sánh thực tế (`comparison_results.json`) đã được đo đạc bằng script `compare_models.py`.
- Cung cấp API `POST /api/predict` (tuỳ chọn) cho phép upload ảnh và chạy suy luận đồng thời trên cả 3 mô hình.

### 3. Next.js Dashboard (Frontend)
- **Công nghệ:** Next.js 15 (App Router), React, Tailwind CSS, Recharts, Lucide Icons.
- **Type-Safety (Chuẩn Matt Pocock):** Sử dụng **Zod** để xác thực chặt chẽ mọi dữ liệu JSON từ Backend trước khi đưa vào React State, đảm bảo không bao giờ xảy ra lỗi runtime do sai lệch kiểu dữ liệu.
- **Tính năng:** Hiển thị Dashboard so sánh trực quan các chỉ số:
  - **mAP (Mean Average Precision):** Độ chính xác trung bình.
  - **Inference Time:** Tốc độ suy luận (tính bằng ms).
  - **Training Loss Curve:** Biểu đồ tốc độ hội tụ trong quá trình huấn luyện.

---

## Cấu trúc thư mục (Project Structure)
```text
cnn/
├── api.py                  # FastAPI Server
├── compare_models.py       # Script đo đạc tự động 3 mô hình
├── frontend/               # Next.js Web Dashboard
│   ├── src/app             # Các trang giao diện (Routing)
│   ├── src/components      # Các React Components (Dashboard.tsx)
│   └── src/lib/schemas.ts  # Zod Schemas định nghĩa cấu trúc dữ liệu
├── src/
│   ├── model.py            # Chứa Multi-scale CNN & Standard CNN
│   ├── knn_detector.py     # Pipeline huấn luyện KNN với HOG
│   └── dataset.py          # Custom PyTorch Dataset
├── dataset/                # (Bỏ qua trên git) Chứa ảnh và annotations
├── models/                 # Chứa trọng số (.pth, .pkl) và kết quả (.json)
└── PROJECT_SUMMARY.md      # Tài liệu tổng quan (file này)
```

## Các lệnh vận hành (Commands)

**1. Cài đặt môi trường:**
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**2. Chạy quá trình đo đạc (Cập nhật kết quả so sánh mới nhất):**
```bash
python compare_models.py
```

**3. Khởi động Backend (FastAPI):**
```bash
python api.py
```
*(Server chạy tại `http://localhost:8000`)*

**4. Khởi động Frontend (Next.js Dashboard):**
```bash
cd frontend
npm run dev
```
*(Web chạy tại `http://localhost:3000`)*
