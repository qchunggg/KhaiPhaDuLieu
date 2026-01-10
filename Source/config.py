"""
File cấu hình chung cho project.
Đường dẫn được tính tự động dựa trên vị trí file này.
KHÔNG cần sửa gì khi clone project về máy.
"""

import os

# Thư mục gốc của project (thư mục cha của Source)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Thư mục chứa dataset ảnh gốc
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

# Thư mục chứa output (dữ liệu đã xử lý)
OUTPUT_DIR = os.path.join(BASE_DIR, "Output")

# Đường dẫn các file output
X_IMAGES_PATH = os.path.join(OUTPUT_DIR, "X_images.npy")
Y_LABELS_PATH = os.path.join(OUTPUT_DIR, "y_labels.npy")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "features.npy")

# Cấu hình tiền xử lý ảnh
IMG_SIZE = 224

# Danh sách các lớp
CLASS_NAMES = ["cat", "dog", "fox"]
