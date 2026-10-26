# Ve anh huong cua toc do hoc (muc 5.1.4): f(x) = x^2 + 5 sin(x), x0 = -5,
# eta = 0.01 (qua nho), 0.1 (vua phai), 0.5 (qua lon)
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
cau_hinh = [(0.01, "quá nhỏ: hội tụ chậm"), (0.1, "vừa phải"), (0.5, "quá lớn: dao động")]
fig, axes = plt.subplots(1, 4, figsize=(17, 4))
for ax, (eta, nhan) in zip(axes[:3], cau_hinh):
    h = gd(-5, eta)
    ax.plot(xs, f(xs), color="#7F7F7F", lw=2)
    ax.plot(h, f(h), "o-", color="#D62728", ms=4, lw=0.8)
    ax.plot(h[0], f(h[0]), "s", color="black", ms=8)
    ax.set_title(f"$\\eta = {eta}$ ({nhan})\n{len(h) - 1} vòng lặp, $x$ cuối $= {h[-1]:.3f}$", fontsize=10)
    ax.set_xlabel("x"); ax.grid(alpha=0.3)
    axes[3].plot(f(h), lw=2, label=f"$\\eta = {eta}$")
axes[0].set_ylabel("f(x)")
axes[3].axhline(f(-1.1105), color="black", ls=":", lw=1, label="Giá trị nhỏ nhất")
axes[3].set_xlabel("Vòng lặp"); axes[3].set_ylabel("$f(x_t)$")
axes[3].set_title("Giá trị hàm theo vòng lặp", fontsize=10)
axes[3].set_ylim(-4, 12); axes[3].legend(fontsize=9); axes[3].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("images/learning_rate.png", dpi=150)
