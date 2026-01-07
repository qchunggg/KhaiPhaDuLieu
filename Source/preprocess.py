import os
import cv2
import numpy as np
from tqdm import tqdm

from config import (
    DATASET_DIR, OUTPUT_DIR, 
    X_IMAGES_PATH, Y_LABELS_PATH,
    IMG_SIZE, CLASS_NAMES
)

def preprocess_images(dataset_dir, class_names, img_size):
    images = []
    labels = []

    print("Đường dẫn Dataset:", dataset_dir)

    for label_id, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"Không tìm thấy thư mục: {class_dir}")
            continue

        for file_name in tqdm(os.listdir(class_dir), desc=f"Đang xử lý {class_name}"):
            file_path = os.path.join(class_dir, file_name)

            img = cv2.imread(file_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(
                img,
                (img_size, img_size),
                interpolation=cv2.INTER_AREA
            )

            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(label_id)

    return np.array(images), np.array(labels)

if __name__ == "__main__":
    X_images, y_labels = preprocess_images(
        DATASET_DIR,
        CLASS_NAMES,
        IMG_SIZE
    )

    print("\nKẾT QUẢ:")
    print("Số lượng ảnh:", X_images.shape[0])
    print("Kích thước ảnh:", X_images.shape)
    print("Kích thước nhãn:", y_labels.shape)
    print("Giá trị pixel min/max:", X_images.min(), X_images.max())

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(X_IMAGES_PATH, X_images)
    np.save(Y_LABELS_PATH, y_labels)

    print("\nĐã lưu dữ liệu tiền xử lý vào thư mục:", OUTPUT_DIR)
