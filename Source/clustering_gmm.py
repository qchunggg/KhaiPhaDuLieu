import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from config import FEATURES_PATH, Y_LABELS_PATH, OUTPUT_DIR

print("Loading data...")
X = np.load(FEATURES_PATH)
y_true = np.load(Y_LABELS_PATH)

print("Feature shape:", X.shape)

# ===== Scale =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== GMM =====
K = 3  # cat/dog/fox
gmm = GaussianMixture(
    n_components=K,
    covariance_type="diag",  # ổn cho dữ liệu nhiều chiều
    random_state=42,
    n_init=5
)

print(f"Running GMM (K={K})...")
y_pred = gmm.fit_predict(X_scaled)

# ===== Evaluate =====
ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)
sil = silhouette_score(X_scaled, y_pred)

print("\n====== KẾT QUẢ GMM ======")
print("ARI:", round(ari, 4))
print("NMI:", round(nmi, 4))
print("Silhouette:", round(sil, 4))

# ===== Save =====
out_path = os.path.join(OUTPUT_DIR, "gmm_clusters.npy")
np.save(out_path, y_pred)
print("\nSaved:", out_path)
