# Muc 5.2 va 5.3.1, Buoc 2. Batch, Stochastic va Mini-batch Gradient Descent
# Mot ham chung voi tham so batch_size: N -> Batch GD, 1 -> SGD, o giua -> Mini-batch GD
import numpy as np

np.random.seed(2)
X = np.random.rand(1000)
y = 4 + 3 * X + .2 * np.random.randn(1000)
Xbar = np.column_stack([np.ones(len(X)), X])


def loss(w, Xb, y):
    return np.sum((Xb @ w - y)**2) / (2 * len(y))


def grad(w, Xb, y):
    return Xb.T @ (Xb @ w - y) / len(y)


def minibatch_gd(Xb, y, eta, batch_size, so_epoch, w_init=None, seed=0):
    rng = np.random.default_rng(seed)
    N = len(y)
    w = np.zeros(Xb.shape[1]) if w_init is None else w_init.copy()
    lich_su = [loss(w, Xb, y)]                    # loss tren toan bo du lieu sau moi epoch
    for epoch in range(so_epoch):
        thu_tu = rng.permutation(N)               # xao tron du lieu moi epoch
        for s in range(0, N, batch_size):
            B = thu_tu[s:s + batch_size]          # chi so cac mau trong mini-batch
            w = w - eta * grad(w, Xb[B], y[B])    # gradient trung binh tren mini-batch
        lich_su.append(loss(w, Xb, y))
    return w, lich_su


w_ols = np.linalg.pinv(Xbar.T @ Xbar) @ Xbar.T @ y
print('OLS: w = %s, loss nho nhat = %.4f' % (w_ols.round(4), loss(w_ols, Xbar, y)))

w0 = np.array([2., 1.])
for ten, bs, eta in [('Batch', 1000, 1.0), ('SGD', 1, 0.05), ('Mini-batch', 32, 0.5)]:
    w, lich_su = minibatch_gd(Xbar, y, eta, bs, so_epoch=20, w_init=w0)
    print('%-10s cap nhat/epoch = %4d, w = %s, loss sau epoch 1, 2, 20: %s' %
          (ten, int(np.ceil(1000 / bs)), w.round(4),
           np.round([lich_su[1], lich_su[2], lich_su[20]], 4)))
