"""
Phân cụm ảnh động vật bằng thuật toán K-Means
Sử dụng đặc trưng trích xuất từ CNN (EfficientNetB0)
"""

import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from config import FEATURES_PATH, Y_LABELS_PATH, OUTPUT_DIR


def main():
    print("Đang tải dữ liệu đặc trưng...")
    X = np.load(FEATURES_PATH)
    y_true = np.load(Y_LABELS_PATH)

    print("Kích thước đặc trưng:", X.shape)

    # Khởi tạo K-Means
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    print("Đang chạy K-Means...")
    y_pred = kmeans.fit_predict(X)

    # Đánh giá kết quả (chỉ để kiểm tra)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    print("\nKẾT QUẢ PHÂN CỤM K-MEANS")
    print("ARI:", round(ari, 4))
    print("NMI:", round(nmi, 4))

    # Lưu nhãn cụm (nếu cần)
    np.save(os.path.join(OUTPUT_DIR, "kmeans_labels.npy"), y_pred)
    print("Đã lưu nhãn cụm vào Output/kmeans_labels.npy")


if __name__ == "__main__":
    main()
