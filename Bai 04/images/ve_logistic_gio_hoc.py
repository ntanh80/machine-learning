# Ve hoi quy Logistic cho vi du gio on tap - ket qua thi (muc 4.1.4):
# (a) xac suat thi do theo so gio, (b) ham mat mat giam dan qua cac vong lap Gradient Descent
import numpy as np
import matplotlib.pyplot as plt

X = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
Xbar = np.column_stack([np.ones(len(X)), X])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(w):
    a = sigmoid(Xbar @ w)
    return -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))

w = np.zeros(2)
lich_su = [loss(w)]
for _ in range(2000):
    w = w - 0.5 * Xbar.T @ (sigmoid(Xbar @ w) - y) / len(y)
    lich_su.append(loss(w))
print(w, lich_su[-1])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
x0 = np.linspace(0, 6, 300)
ax.plot(X[y == 0], y[y == 0], "o", color="#D62728", ms=8, label="Trượt (y = 0)")
ax.plot(X[y == 1], y[y == 1], "s", color="#1F77B4", ms=8, label="Đỗ (y = 1)")
ax.plot(x0, sigmoid(w[0] + w[1] * x0), color="#2CA02C", lw=2.5,
        label=rf"$\hat{{y}} = \sigma({w[1]:.2f}x {w[0]:+.2f})$")
nguong = -w[0] / w[1]
ax.axhline(0.5, color="black", lw=0.8, ls=":")
ax.axvline(nguong, color="black", lw=1, ls="--")
ax.annotate(f"ranh giới x ≈ {nguong:.2f} giờ", (nguong, 0.5), (nguong + 0.3, 0.25),
            arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("Số giờ ôn tập"); ax.set_ylabel("Xác suất thi đỗ")
ax.set_title("(a) Mô hình hồi quy Logistic")
ax.legend(loc="center right", fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(lich_su, color="#1F77B4", lw=2)
ax.set_xlabel("Số vòng lặp"); ax.set_ylabel(r"$J(\mathbf{w})$")
ax.set_title("(b) Hàm mất mát giảm dần theo Gradient Descent")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("images/logistic_gio_hoc.png", dpi=150)
