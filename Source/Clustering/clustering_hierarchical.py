import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from config import FEATURES_PATH, OUTPUT_DIR

print("Đang tải đặc trưng CNN:")
features = np.load(FEATURES_PATH)
print(f"Kích thước đặc trưng: {features.shape}")

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

print("Đang chạy Hierarchical Clustering:")
hc = AgglomerativeClustering(
    n_clusters=3,
    metric="euclidean",
    linkage="ward"
)
cluster_labels = hc.fit_predict(features_scaled)

print(f"Các nhãn cụm: {np.unique(cluster_labels)}")
print(f"Số lượng mẫu: {len(cluster_labels)}")

output_path = os.path.join(OUTPUT_DIR, "hierarchical_labels.npy")
np.save(output_path, cluster_labels)
print(f"\nĐã lưu kết quả phân cụm vào: {output_path}")

abc
