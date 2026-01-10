import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt

def visualize_clusters(x_images, cluster_labels, n_per_cluster=2):
    """Hiển thị n_per_cluster ảnh mỗi cụm"""
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    fig, axes = plt.subplots(n_clusters, n_per_cluster, figsize=(n_per_cluster * 3, n_clusters * 3))
    fig.suptitle("Ảnh mẫu từ mỗi cụm", fontsize=14)
    
    for i, cluster_id in enumerate(unique_clusters):
        indices = np.where(cluster_labels == cluster_id)[0][:n_per_cluster]
        for j, idx in enumerate(indices):
            ax = axes[i, j] if n_clusters > 1 else axes[j]
            ax.imshow(x_images[idx])
            ax.set_title(f"Cụm {cluster_id}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()
