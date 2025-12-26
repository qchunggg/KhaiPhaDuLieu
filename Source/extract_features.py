import os
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from config import OUTPUT_DIR, X_IMAGES_PATH, FEATURES_PATH, IMG_SIZE

print("Loading preprocessed images...")
X_images = np.load(X_IMAGES_PATH)
print("Image shape:", X_images.shape)

print("Loading EfficientNetB0 model...")

model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)

model.trainable = False

print("Extracting features...")
X_preprocessed = preprocess_input(X_images * 255.0)

features = model.predict(
    X_preprocessed,
    batch_size=16,
    verbose=1
)

print("Feature shape:", features.shape)

np.save(FEATURES_PATH, features)

print("\nĐã lưu đặc trưng vào:", FEATURES_PATH)
