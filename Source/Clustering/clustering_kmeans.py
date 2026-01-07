import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import FEATURES_PATH, OUTPUT_DIR

print("Đang tải dữ liệu đặc trưng:")
X = np.load(FEATURES_PATH)
print(f"Kích thước đặc trưng: {X.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Đang chạy K-Means:")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

print(f"Các nhãn cụm: {np.unique(labels)}")
print(f"Số lượng mẫu: {len(labels)}")

output_path = os.path.join(OUTPUT_DIR, "kmeans_labels.npy")
np.save(output_path, labels)
print(f"\nĐã lưu kết quả phân cụm vào: {output_path}")
