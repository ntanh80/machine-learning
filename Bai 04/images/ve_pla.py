# Minh hoa thuat toan PLA (muc 4.2.3): moi o la mot lan cap nhat w <- w + y_i x_i
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)
means = [[-1, 0], [1, 0]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X = np.concatenate((X0, X1), axis=0)
y = np.concatenate((np.ones(N), -1 * np.ones(N)))
Xbar = np.concatenate((np.ones((2 * N, 1)), X), axis=1)

def predict(w, Xbar):
    return np.where(Xbar @ w >= 0, 1, -1)

# Chay PLA, luu lai w truoc/sau va diem bi phan loai sai o moi lan cap nhat
w = np.array([0.5, -1.0, 1.0])
lich_su = []
while True:
    sai = np.where(predict(w, Xbar) != y)[0]
    if len(sai) == 0:
        break
    i = sai[0]
    lich_su.append((w.copy(), i, w + y[i] * Xbar[i]))
    w = w + y[i] * Xbar[i]
print("So lan cap nhat:", len(lich_su), "w =", w)

def ve_duong(ax, w, **kw):
    xs = np.array([-3, 3])
    ax.plot(xs, -(w[0] + w[1] * xs) / w[2], **kw)

so_o = min(len(lich_su), 6)
fig, axes = plt.subplots(1, so_o, figsize=(3.2 * so_o, 3.4), squeeze=False)
for k, ax in enumerate(axes[0]):
    idx = k if k < so_o - 1 else len(lich_su) - 1
    w_cu, i, w_moi = lich_su[idx]
    ax.plot(X0[:, 0], X0[:, 1], "s", color="#1F77B4", ms=6)
    ax.plot(X1[:, 0], X1[:, 1], "o", color="#D62728", ms=6)
    ve_duong(ax, w_cu, color="black", lw=2)
    ve_duong(ax, w_moi, color="#2CA02C", lw=2, ls="--")
    ax.plot(X[i, 0], X[i, 1], "o", ms=16, mfc="none", mec="black", mew=1.5)
    ax.set_xlim(-3, 3); ax.set_ylim(-2.5, 2.5)
    ax.set_title(f"Cập nhật {idx + 1}/{len(lich_su)}")
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig("images/pla.png", dpi=150)
