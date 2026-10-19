# Ve du lieu chieu cao - can nang cua tap huan luyen (muc 3.2)
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[147, 150, 153, 158, 163, 165, 168, 170,
               173, 175, 178, 180, 183]]).T
y = np.array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(X, y, 'ro')
ax.set_xlabel("Chiều cao (cm)"); ax.set_ylabel("Cân nặng (kg)")
ax.set_title("Chiều cao và cân nặng của 13 người trong tập huấn luyện")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("images/chieu_cao_can_nang.png", dpi=150)
