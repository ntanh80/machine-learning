# Vẽ dữ liệu chiều cao – cân nặng của tập huấn luyện (mục 3.2)
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

IMAGES = Path(__file__).resolve().parent   # thư mục images/

# Chiều cao (cm) của 13 người, mỗi hàng là một mẫu
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170,
               173, 175, 178, 180, 183]]).T
# Cân nặng tương ứng (kg) – biến mục tiêu
y = np.array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(X, y, 'ro')
ax.set_xlabel("Chiều cao (cm)"); ax.set_ylabel("Cân nặng (kg)")
ax.set_title("Chiều cao và cân nặng của 13 người trong tập huấn luyện")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(IMAGES / "height_weight.png", dpi=150)
