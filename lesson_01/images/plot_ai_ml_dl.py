# Vẽ hình quan hệ bao hàm: Trí tuệ nhân tạo ⊃ Học máy ⊃ Học sâu
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

OUTPUT = Path(__file__).resolve().parent / 'ai_ml_dl.png'   # ảnh lưu cạnh script

fig, ax = plt.subplots(figsize=(8, 5.5))

# Mỗi tầng: (tâm x, tâm y, rộng, cao, màu, tên, ví dụ, vị trí y của chữ)
layers = [
    (0.0, 0.0, 9.6, 6.4, '#dbe9f6', 'Trí tuệ nhân tạo (AI)',
     'Hệ chuyên gia, tìm kiếm, lập kế hoạch, logic', 2.55),
    (0.0, -0.55, 6.8, 4.4, '#9ecae1', 'Học máy (Machine Learning)',
     'Hồi quy, cây quyết định, SVM, K-means', 1.05),
    (0.0, -1.1, 4.0, 2.2, '#3182bd', 'Học sâu (Deep Learning)',
     'CNN, RNN, Transformer', -0.85),
]
for x, y, width, height, color, name, example, text_y in layers:
    ax.add_patch(Ellipse((x, y), width, height, facecolor=color, edgecolor='#08306b', linewidth=1.5))
    text_color = 'white' if color == '#3182bd' else '#08306b'
    ax.text(x, text_y, name, ha='center', va='center', fontsize=13, fontweight='bold', color=text_color)
    ax.text(x, text_y - 0.42, example, ha='center', va='center', fontsize=10, color=text_color)

ax.set_xlim(-5, 5)
ax.set_ylim(-3.4, 3.4)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Quan hệ giữa Trí tuệ nhân tạo, Học máy và Học sâu', fontsize=13)

fig.tight_layout()
fig.savefig(OUTPUT, dpi=150)
print('Đã lưu hình:', OUTPUT)
