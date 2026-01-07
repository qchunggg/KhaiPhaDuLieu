import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from config import Y_LABELS_PATH, OUTPUT_DIR

print("Đang tải dữ liệu:")
y_true = np.load(Y_LABELS_PATH)
hierarchical_labels = np.load(os.path.join(OUTPUT_DIR, "hierarchical_labels.npy"))

print(f"Số lượng mẫu: {len(y_true)}")

ari = adjusted_rand_score(y_true, hierarchical_labels)
nmi = normalized_mutual_info_score(y_true, hierarchical_labels)

print("\nKẾT QUẢ ĐÁNH GIÁ HIERARCHICAL CLUSTERING:")
print(f"ARI (Adjusted Rand Index): {round(ari, 4)}")
print(f"NMI (Normalized Mutual Information): {round(nmi, 4)}")
