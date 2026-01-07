# Phân loại ảnh Cat - Dog - Fox

Dự án phân loại ảnh sử dụng Deep Learning với EfficientNetB0.
Dataset: https://www.kaggle.com/datasets/snmahsa/animal-image-dataset-cats-dogs-and-foxes

## Cấu trúc thư mục

```
BTL/
├── Dataset/          # Chứa dữ liệu ảnh gốc (cần tải về)
│   ├── cat/
│   ├── dog/
│   └── fox/
├── Output/           # Kết quả xử lý (tự động tạo)
│   ├── X_images.npy
│   ├── y_labels.npy
│   └── features.npy
├── Source/           # Source code
│   ├── config.py
│   ├── preprocess.py
│   ├── extract_features.py
│   └── requirements.txt
└── README.md
```

## Hướng dẫn cài đặt

### 1. Cài đặt thư viện

```bash
cd Source
pip install -r requirements.txt
```

### 2. Tải Dataset

Tải dataset và giải nén vào thư mục `Dataset/` với cấu trúc:

- `Dataset/cat/` - chứa ảnh mèo
- `Dataset/dog/` - chứa ảnh chó
- `Dataset/fox/` - chứa ảnh cáo

### 3. Chạy chương trình

**Bước 1**: Tiền xử lý ảnh

```bash
python preprocess.py
```

→ Tạo `X_images.npy` và `y_labels.npy` trong thư mục `Output/`

**Bước 2**: Trích xuất đặc trưng

```bash
python extract_features.py
```

→ Tạo `features.npy` trong thư mục `Output/`

**Bước 3**: Chạy phân cụm

```bash
python clustering_kmeans.py
python clustering_hierarchical.py
```

## Quy tắc đặt tên file

| Loại file | Quy tắc                           | Ví dụ                                                |
| --------- | --------------------------------- | ---------------------------------------------------- |
| Phân cụm  | `clustering_<tên phương pháp>.py` | `clustering_kmeans.py`, `clustering_hierarchical.py` |
