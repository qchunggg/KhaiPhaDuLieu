import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from config import X_IMAGES_PATH, FEATURES_PATH, IMG_SIZE

# Load dữ liệu
print("Loading preprocessed images...")
X_images = np.load(X_IMAGES_PATH)
print(f"Image shape: {X_images.shape}")

# Kiểm tra dữ liệu đầu vào
assert X_images.shape[1:3] == (IMG_SIZE, IMG_SIZE), \
    f"Image size mismatch! Expected {IMG_SIZE}x{IMG_SIZE}, got {X_images.shape[1:3]}"
print(f"Pixel range: [{X_images.min():.2f}, {X_images.max():.2f}]")

# Load model EfficientNetB0
print("Loading EfficientNetB0 model...")
model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)  # Dùng IMG_SIZE từ config
)
model.trainable = False

# Preprocess: chuyển từ [0,1] về [0,255] rồi áp dụng preprocess_input
print("Extracting features...")
X_preprocessed = preprocess_input(X_images * 255.0)

features = model.predict(X_preprocessed, batch_size=16, verbose=1)
print(f"Feature shape: {features.shape}")

# Kiểm tra NaN/Inf trong features
nan_count = np.isnan(features).sum()
inf_count = np.isinf(features).sum()
if nan_count > 0 or inf_count > 0:
    print(f"[WARNING] Features contain {nan_count} NaN and {inf_count} Inf values!")
else:
    print("[OK] No NaN/Inf in features")

# Lưu kết quả
np.save(FEATURES_PATH, features)
print(f"\nĐã lưu đặc trưng vào: {FEATURES_PATH}")
