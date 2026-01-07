import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from config import X_IMAGES_PATH, FEATURES_PATH, IMG_SIZE

print("Đang tải ảnh đã tiền xử lý...")
X_images = np.load(X_IMAGES_PATH)
print(f"Kích thước ảnh: {X_images.shape}")

assert X_images.shape[1:3] == (IMG_SIZE, IMG_SIZE), \
    f"Kích thước ảnh không khớp! Mong đợi {IMG_SIZE}x{IMG_SIZE}, nhận được {X_images.shape[1:3]}"
print(f"Giá trị pixel: [{X_images.min():.2f}, {X_images.max():.2f}]")
print("Đang tải mô hình EfficientNetB0...")
model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
model.trainable = False

print("Đang trích xuất đặc trưng...")
X_preprocessed = preprocess_input(X_images * 255.0)

features = model.predict(X_preprocessed, batch_size=16, verbose=1)
print(f"Kích thước đặc trưng: {features.shape}")

nan_count = np.isnan(features).sum()
inf_count = np.isinf(features).sum()
if nan_count > 0 or inf_count > 0:
    print(f"Đặc trưng chứa {nan_count} giá trị NaN và {inf_count} giá trị Inf!")
else:
    print("Không có NaN/Inf trong đặc trưng")

np.save(FEATURES_PATH, features)
print(f"\nĐã lưu đặc trưng vào: {FEATURES_PATH}")
