# Ve Gradient Descent cho ham mot bien f(x) = x^2 + 5 sin(x) (muc 5.1.3)
# voi hai diem khoi tao x0 = -5 va x0 = 5, cung toc do hoc eta = 0.1
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 + 5 * np.sin(x)

def grad(x):
    return 2 * x + 5 * np.cos(x)

def gd(x0, eta, max_iter=100, tol=1e-3):
    x = x0
    lich_su = [x0]
    for _ in range(max_iter):
        x = x - eta * grad(x)
        lich_su.append(x)
        if abs(grad(x)) < tol:
            break
    return np.array(lich_su)

xs = np.linspace(-6, 6, 400)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, x0 in zip(axes, [-5, 5]):
    h = gd(x0, 0.1)
    ax.plot(xs, f(xs), color="#7F7F7F", lw=2, label=r"$f(x) = x^2 + 5\sin x$")
    ax.plot(h, f(h), "o-", color="#D62728", ms=5, lw=1, label="Các điểm $x_t$")
    ax.plot(h[0], f(h[0]), "s", color="black", ms=9, label=f"Khởi tạo $x_0 = {x0}$")
    ax.plot(h[-1], f(h[-1]), "*", color="#2CA02C", ms=16,
            label=f"Kết thúc $x = {h[-1]:.3f}$")
    ax.set_title(f"$x_0 = {x0}$, $\\eta = 0.1$: dừng sau {len(h) - 1} vòng lặp")
    ax.set_xlabel("x"); ax.grid(alpha=0.3)
    ax.legend(loc="upper center", fontsize=9)
axes[0].set_ylabel("f(x)")
fig.tight_layout()
fig.savefig("images/gd_mot_bien.png", dpi=150)
