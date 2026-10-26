# So sanh Batch, Stochastic va Mini-batch Gradient Descent (muc 5.2)
# (a) quy dao 60 lan cap nhat dau tien tren duong dong muc, (b) ham mat mat theo epoch
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)
X = np.random.rand(1000)
y = 4 + 3 * X + .2 * np.random.randn(1000)
Xbar = np.column_stack([np.ones(len(X)), X])
w_ols = np.linalg.pinv(Xbar.T @ Xbar) @ Xbar.T @ y

def cost(w):
    return .5 / len(y) * np.sum((Xbar @ w - y)**2)

def minibatch_gd(eta, batch_size, so_epoch, seed=0):
    rng = np.random.default_rng(seed)
    w = np.array([2., 1.])
    quy_dao, mat_mat = [w], [cost(w)]
    N = len(y)
    for _ in range(so_epoch):
        idx = rng.permutation(N)
        for s in range(0, N, batch_size):
            b = idx[s:s + batch_size]
            w = w - eta * Xbar[b].T @ (Xbar[b] @ w - y[b]) / len(b)
            quy_dao.append(w)
        mat_mat.append(cost(w))
    return np.array(quy_dao), mat_mat

cau_hinh = [("Batch GD ($\\eta$ = 1)", 1000, 1.0, "#1F77B4"),
            ("Stochastic GD ($\\eta$ = 0.05)", 1, 0.05, "#D62728"),
            ("Mini-batch GD, m = 32 ($\\eta$ = 0.5)", 32, 0.5, "#2CA02C")]

W0, W1 = np.meshgrid(np.linspace(1.5, 5, 200), np.linspace(0.5, 4, 200))
J = np.array([[cost(np.array([a, b])) for a, b in zip(r0, r1)] for r0, r1 in zip(W0, W1)])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
axes[0].contour(W0, W1, J, levels=np.logspace(-1.6, 1.2, 15), cmap="Greys", linewidths=0.8)
for ten, bs, eta, mau in cau_hinh:
    quy_dao, mat_mat = minibatch_gd(eta, bs, 20)
    q = quy_dao[:61]
    axes[0].plot(q[:, 0], q[:, 1], "o-", color=mau, ms=2.5, lw=1, label=ten)
    axes[1].plot(mat_mat, "o-", color=mau, ms=4, lw=2, label=ten)
axes[0].plot(2, 1, "ks", ms=8)
axes[0].plot(w_ols[0], w_ols[1], "*", color="black", ms=15, label="Nghiệm OLS")
axes[0].set_xlabel("$w_0$"); axes[0].set_ylabel("$w_1$")
axes[0].set_title("(a) 60 lần cập nhật đầu tiên")
axes[0].legend(fontsize=8, loc="upper right")
axes[1].axhline(cost(w_ols), color="black", ls=":", label="Giá trị nhỏ nhất")
axes[1].set_yscale("log"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("$J(\\mathbf{w})$ (thang log)")
axes[1].set_title("(b) Hàm mất mát sau mỗi epoch")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("images/bien_the_gd.png", dpi=150)
