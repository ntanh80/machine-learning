"""
Tạo ảnh đa cấp xám 20x20, đọc lại ảnh và in giá trị điểm ảnh
dưới dạng mảng hai chiều (2D) và mảng một chiều (1D).
"""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

H, W = 20, 20
TEN_FILE = Path(__file__).resolve().parent / "anh_xam_20x20.png"

# ---------- 1. Tạo ảnh đa cấp xám ----------
# Nền là dải chuyển màu ngang từ đen (0) sang trắng (255)
anh = np.tile(np.linspace(0, 255, W), (H, 1)).astype(np.uint8)

# Vẽ thêm một hình vuông xám trung bình (128) ở giữa
anh[6:14, 6:14] = 128

# Vẽ một đường chéo trắng (255)
for i in range(H):
    anh[i, i] = 255

Image.fromarray(anh, mode="L").save(TEN_FILE)
print(f"Đã lưu ảnh: {TEN_FILE}")

# ---------- 2. Đọc ảnh ----------
img = Image.open(TEN_FILE).convert("L")   # "L" = ảnh xám 8 bit
mang_2d = np.array(img)
print("Kích thước ảnh (cao x rộng):", mang_2d.shape)

# ---------- 3. In mảng hai chiều ----------
print("\n===== MẢNG HAI CHIỀU (20 x 20) =====")
for hang in mang_2d:
    print(" ".join(f"{p:3d}" for p in hang))

# ---------- 4. In mảng một chiều ----------
mang_1d = mang_2d.flatten()               # duỗi theo từng hàng
print(f"\n===== MẢNG MỘT CHIỀU ({mang_1d.size} phần tử) =====")
np.set_printoptions(threshold=np.inf, linewidth=120)
print(mang_1d)

# ---------- 5. Hiển thị ảnh kèm giá trị điểm ảnh ----------
fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(mang_2d, cmap="gray", vmin=0, vmax=255)
for r in range(H):
    for c in range(W):
        v = mang_2d[r, c]
        ax.text(c, r, str(v), ha="center", va="center", fontsize=6,
                color="black" if v > 127 else "white")
ax.set_title("Ảnh đa cấp xám 20x20")
ax.set_xticks(range(W))
ax.set_yticks(range(H))
plt.tight_layout()
plt.show()
