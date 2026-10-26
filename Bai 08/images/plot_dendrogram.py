# Phân cụm phân cấp trên 6 điểm và dendrogram (mục 8.1.4)
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

IMAGES = Path(__file__).resolve().parent   # thư mục images/

# Cùng dữ liệu với codes/04_hierarchical_clustering.ipynb
X = np.array([[1, 1], [1.5, 1.5], [5, 5], [3, 4], [4, 4], [3, 3.5]])
names = ['A', 'B', 'C', 'D', 'E', 'F']

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
axes[0].scatter(X[:, 0], X[:, 1], s=80, color='#1F77B4')
for name, (x, y) in zip(names, X):
    axes[0].text(x + 0.12, y + 0.08, name, fontsize=12)
axes[0].set_xlim(0, 6); axes[0].set_ylim(0, 6); axes[0].set_aspect('equal'); axes[0].grid(alpha=0.3)
axes[0].set_title('(a) Dữ liệu 6 điểm')

for ax, method, cut, title in [(axes[1], 'single', 1.8, '(b) Single linkage'), (axes[2], 'complete', 3.5, '(c) Complete linkage')]:
    dendrogram(linkage(X, method=method), labels=names, ax=ax, color_threshold=cut)
    ax.axhline(cut, color='k', ls='--', lw=1)
    ax.text(ax.get_xlim()[1] * 0.98, cut + 0.08, 'cắt ở độ cao %.1f' % cut, ha='right', fontsize=9)
    ax.set_ylabel('Khoảng cách gộp'); ax.set_title(title)
fig.tight_layout()
fig.savefig(IMAGES / 'dendrogram.png', dpi=150)
