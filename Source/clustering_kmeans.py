import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from config import FEATURES_PATH, Y_LABELS_PATH, OUTPUT_DIR
import os

print("Đang tải dữ liệu đặc trưng...")
X = np.load(FEATURES_PATH)
y_true = np.load(Y_LABELS_PATH)

print("Kích thước đặc trưng:", X.shape)

# 🔹 Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Đang chạy K-Means...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Đánh giá
ari = adjusted_rand_score(y_true, labels)
nmi = normalized_mutual_info_score(y_true, labels)

print("\nKẾT QUẢ PHÂN CỤM K-MEANS")
print(f"ARI: {ari:.4f}")
print(f"NMI: {nmi:.4f}")

# Lưu kết quả
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, "kmeans_labels.npy"), labels)

print("Đã lưu nhãn cụm vào Output/kmeans_labels.npy")
