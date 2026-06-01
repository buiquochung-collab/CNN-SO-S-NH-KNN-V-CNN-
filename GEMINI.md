# CNN Deep Learning Project (cnn_dl)

Dự án này tập trung vào việc nghiên cứu, triển khai và huấn luyện các mô hình Mạng thần kinh tích chập (Convolutional Neural Networks - CNN) cho các bài toán Deep Learning (ví dụ: phân loại hình ảnh, phát hiện vật thể).

## Tổng quan dự án (Project Overview)

- **Mục tiêu:** Xây dựng pipeline hoàn chỉnh từ xử lý dữ liệu đến huấn luyện và đánh giá mô hình CNN.
- **Công nghệ dự kiến:**
    - Ngôn ngữ: Python 3.x
    - Frameworks: PyTorch hoặc TensorFlow/Keras
    - Thư viện hỗ trợ: OpenCV, NumPy, Matplotlib, Pandas, Scikit-learn
    - Quản lý môi trường: `venv` hoặc `conda`

## Cấu trúc thư mục dự kiến (Proposed Architecture)

```text
cnn_dl/
├── data/               # Dữ liệu thô và dữ liệu đã xử lý
├── models/             # Lưu trữ các trọng số mô hình đã huấn luyện (.pth, .h5)
├── notebooks/          # Jupyter Notebooks cho EDA và thử nghiệm nhanh
├── src/                # Mã nguồn chính
│   ├── data_loader.py  # Xử lý và tải dữ liệu
│   ├── model.py        # Định nghĩa kiến trúc mạng CNN
│   ├── train.py        # Script huấn luyện
│   └── utils.py        # Các hàm tiện ích
├── tests/              # Unit tests cho mã nguồn
├── GEMINI.md           # Hướng dẫn và ngữ cảnh cho Agent (tệp này)
├── requirements.txt    # Danh sách các thư viện phụ thuộc
└── README.md           # Giới thiệu tổng quan dự án
```

## Cài đặt và Chạy (Building and Running)

### 1. Thiết lập môi trường
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (Windows)
.\venv\Scripts\activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Huấn luyện mô hình
```bash
python src/train.py --config config.yaml
```

## Quy ước phát triển (Development Conventions)

- **Mã nguồn:** Tuân thủ chuẩn PEP 8 cho Python.
- **Documentation:** Sử dụng Docstrings cho tất cả các hàm và lớp.
- **Logging:** Sử dụng thư viện `logging` của Python thay vì `print()` cho các thông tin quan trọng.
- **Testing:** Viết test case cho các module xử lý dữ liệu và kiến trúc mô hình trước khi huấn luyện quy mô lớn.

## Ghi chú cho Agent

- Luôn kiểm tra cấu trúc dữ liệu trong thư mục `data/` trước khi sửa đổi `data_loader.py`.
- Khi đề xuất kiến trúc mô hình mới, hãy giải thích lý do lựa chọn các lớp (layers) và tham số.
- Ưu tiên sử dụng các phương pháp tối ưu hóa bộ nhớ khi làm việc với tập dữ liệu hình ảnh lớn.
