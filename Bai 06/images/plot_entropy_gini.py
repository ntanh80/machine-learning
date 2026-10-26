# So sánh các độ đo vẩn đục với bài toán hai lớp (mục 6.1.2, 6.1.3)
# Entropy, chỉ số Gini và tỉ lệ lỗi phân loại theo tỉ lệ p của một lớp
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

IMAGES = Path(__file__).resolve().parent   # thư mục images/

p = np.linspace(1e-6, 1 - 1e-6, 400)                       # tỉ lệ lớp dương
entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)       # H(p)
gini = 1 - p**2 - (1 - p)**2                               # Gini(p) = 2p(1 - p)
error = 1 - np.maximum(p, 1 - p)                           # tỉ lệ lỗi khi gán lớp đa số

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(p, entropy, lw=2, label='Entropy $H(p)$')
ax.plot(p, gini, lw=2, label='Gini $= 1 - p^2 - (1-p)^2$')
ax.plot(p, error, lw=2, ls='--', label='Tỉ lệ lỗi $= 1 - \\max(p, 1-p)$')
ax.plot([0.6], [0.971], 'o', color='C0')
ax.annotate('6 A, 4 B: H = 0.971', (0.6, 0.971), xytext=(0.66, 1.02), fontsize=9)
ax.plot([0.6], [0.48], 'o', color='C1')
ax.annotate('Gini = 0.48', (0.6, 0.48), xytext=(0.66, 0.43), fontsize=9)
ax.set_xlabel('Tỉ lệ mẫu thuộc một lớp $p$')
ax.set_ylabel('Độ vẩn đục')
ax.set_title('Các độ đo vẩn đục của một nút có hai lớp')
ax.set_ylim(0, 1.1)
ax.legend(loc='lower center', fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMAGES / 'entropy_gini.png', dpi=150)
