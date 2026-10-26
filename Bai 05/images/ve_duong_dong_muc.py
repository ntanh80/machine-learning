# Ve duong dong muc cua ham mat mat MSE va quy dao Gradient Descent (muc 5.1.4)
# Du lieu: 1000 diem gan duong thang y = 4 + 3x; khoi tao w = (2, 1); eta = 1 va eta = 0.1
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)
X = np.random.rand(1000)
y = 4 + 3 * X + .2 * np.random.randn(1000)
Xbar = np.column_stack([np.ones(len(X)), X])
w_ols = np.linalg.pinv(Xbar.T @ Xbar) @ Xbar.T @ y

def cost(w):
    return .5 / len(y) * np.sum((Xbar @ w - y)**2)

def grad(w):
    return Xbar.T @ (Xbar @ w - y) / len(y)

def gd(eta, so_vong_lap=100):
    w = np.array([2., 1.])
    lich_su = [w]
    for _ in range(so_vong_lap):
        w = w - eta * grad(w)
        lich_su.append(w)
    return np.array(lich_su)

W0, W1 = np.meshgrid(np.linspace(1.5, 6.5, 200), np.linspace(0.5, 5, 200))
J = np.array([[cost(np.array([a, b])) for a, b in zip(r0, r1)] for r0, r1 in zip(W0, W1)])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, eta in zip(axes, [1, 0.1]):
    h = gd(eta)
    ax.contour(W0, W1, J, levels=np.logspace(-1.6, 1.2, 15), cmap="viridis", linewidths=1)
    ax.plot(h[:, 0], h[:, 1], "o-", color="#D62728", ms=3, lw=1)
    ax.plot(2, 1, "ks", ms=8); ax.annotate("khởi tạo", (2, 1), (2.15, 0.65))
    ax.plot(w_ols[0], w_ols[1], "*", color="#2CA02C", ms=16)
    ax.annotate("nghiệm OLS", w_ols, (w_ols[0] + 0.25, w_ols[1] + 0.35))
    ax.set_xlabel("$w_0$"); ax.set_ylabel("$w_1$")
    ax.set_title(f"$\\eta = {eta}$: sau 100 vòng lặp, $J = {cost(h[-1]):.4f}$")
fig.tight_layout()
fig.savefig("images/duong_dong_muc.png", dpi=150)
