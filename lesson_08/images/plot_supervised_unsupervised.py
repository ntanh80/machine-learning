# Học có giám sát và học không giám sát trên cùng một tập điểm (mục 8.1.1)
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

IMAGES = Path(__file__).resolve().parent   # thư mục images/

X, y = make_blobs(n_samples=150, centers=[[2, 2], [6, 3], [4, 7]], cluster_std=0.9, random_state=4)
labels = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(X)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
markers = ['o', 's', '^']
for k in range(3):
    axes[0].scatter(X[y == k, 0], X[y == k, 1], marker=markers[k], s=25, label='Lớp %d' % (k + 1))
axes[0].set_title('(a) Có giám sát: mỗi điểm có nhãn $y$')
axes[0].legend(fontsize=9)
axes[1].scatter(X[:, 0], X[:, 1], s=25, color='gray')
axes[1].set_title('(b) Không giám sát: chỉ có $\\mathbf{x}$, không có nhãn')
axes[2].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=25)
axes[2].set_title('(c) Phân cụm tự tìm ra 3 nhóm')
for ax in axes:
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
fig.tight_layout()
fig.savefig(IMAGES / 'supervised_unsupervised.png', dpi=150)
