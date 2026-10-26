# Ve hien tuong mac ket tai cuc tieu dia phuong (muc 5.1.4): f(x) = x^4 - 3x^2 + x
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**4 - 3 * x**2 + x

def grad(x):
    return 4 * x**3 - 6 * x + 1

def gd(x0, eta=0.05, so_vong_lap=50):
    x = x0
    lich_su = [x0]
    for _ in range(so_vong_lap):
        x = x - eta * grad(x)
        lich_su.append(x)
    return np.array(lich_su)

xs = np.linspace(-2.1, 2.1, 400)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(xs, f(xs), color="#7F7F7F", lw=2, label=r"$f(x) = x^4 - 3x^2 + x$")
for x0, mau in [(-2.0, "#1F77B4"), (2.0, "#D62728")]:
    h = gd(x0)
    ax.plot(h, f(h), "o-", color=mau, ms=4, lw=1,
            label=f"$x_0 = {x0}$ → $x = {h[-1]:.3f}$, $f = {f(h[-1]):.3f}$")
ax.annotate("cực tiểu toàn cục", (-1.30, f(-1.30)), (-1.2, -4.6), ha="center",
            arrowprops=dict(arrowstyle="->"))
ax.annotate("cực tiểu địa phương", (1.13, f(1.13)), (1.2, -2.6), ha="center",
            arrowprops=dict(arrowstyle="->"))
ax.set_ylim(-5, 6)
ax.set_xlabel("x"); ax.set_ylabel("f(x)")
ax.set_title(r"Gradient Descent với $\eta = 0.05$ từ hai điểm khởi tạo")
ax.legend(loc="upper center", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("images/cuc_tieu_dia_phuong.png", dpi=150)
