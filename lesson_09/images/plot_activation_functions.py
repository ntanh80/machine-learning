# Các hàm kích hoạt thông dụng và đạo hàm của chúng (mục 9.1.2)
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

IMAGES = Path(__file__).resolve().parent   # thư mục images/

z = np.linspace(-5, 5, 500)
sigmoid = 1 / (1 + np.exp(-z))
functions = [('Bước (step)', (z >= 0).astype(float), np.zeros_like(z)),
             ('Sigmoid', sigmoid, sigmoid * (1 - sigmoid)),
             ('Tanh', np.tanh(z), 1 - np.tanh(z)**2),
             ('ReLU', np.maximum(0, z), (z > 0).astype(float)),
             ('Leaky ReLU ($\\alpha$ = 0.1)', np.where(z >= 0, z, 0.1 * z), np.where(z >= 0, 1, 0.1))]

fig, axes = plt.subplots(1, 5, figsize=(19, 3.8))
for ax, (name, f, df) in zip(axes, functions):
    ax.plot(z, f, lw=2.2, label='$f(z)$')
    ax.plot(z, df, lw=1.6, ls='--', label="$f'(z)$")
    ax.axhline(0, color='gray', lw=0.6); ax.axvline(0, color='gray', lw=0.6)
    ax.set_title(name, fontsize=12); ax.set_xlabel('$z$')
    ax.set_ylim(-1.3, 2.2); ax.grid(alpha=0.3)
axes[0].legend(loc='upper left')
fig.tight_layout()
fig.savefig(IMAGES / 'activation_functions.png', dpi=150)
