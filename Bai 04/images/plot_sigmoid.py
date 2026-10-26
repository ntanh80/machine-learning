# Vẽ hàm sigmoid (mục 4.1.2): so sánh các hàm kích hoạt và đạo hàm của sigmoid
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

IMAGES = Path(__file__).resolve().parent   # thư mục images/

s = np.linspace(-6, 6, 400)
sigmoid = 1 / (1 + np.exp(-s))   # giá trị σ(s)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
ax.plot(s, np.clip(0.5 + s / 6, 0, 1), "--", color="#7F7F7F", lw=2, label="Tuyến tính bị chặn")
ax.step(s, (s >= 0).astype(float), where="post", color="#D62728", lw=2, label="Ngưỡng cứng")
ax.plot(s, sigmoid, color="#1F77B4", lw=2.5, label=r"Sigmoid $\sigma(s)$")
ax.axhline(0.5, color="black", lw=0.8, ls=":")
ax.set_xlabel("s"); ax.set_ylabel("f(s)")
ax.set_title("(a) Một số hàm kích hoạt")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(s, sigmoid, color="#1F77B4", lw=2.5, label=r"$\sigma(s) = \frac{1}{1+e^{-s}}$")
ax.plot(s, sigmoid * (1 - sigmoid), color="#2CA02C", lw=2.5, label=r"$\sigma'(s) = \sigma(s)(1-\sigma(s))$")
ax.plot(0, 0.5, "ko")
ax.annotate(r"$\sigma(0)=0.5$", (0, 0.5), (0.6, 0.42))
ax.set_xlabel("s")
ax.set_title("(b) Hàm sigmoid và đạo hàm")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(IMAGES / "sigmoid.png", dpi=150)
