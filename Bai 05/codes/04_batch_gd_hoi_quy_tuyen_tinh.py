# Muc 5.3.1, Buoc 1. Toi uu ham mat mat MSE cua hoi quy tuyen tinh bang Batch GD
# Du lieu: 1000 diem gan duong thang y = 4 + 3x
import numpy as np

np.random.seed(2)
X = np.random.rand(1000)                       # x trong [0, 1)
y = 4 + 3 * X + .2 * np.random.randn(1000)     # y = 4 + 3x + nhieu
Xbar = np.column_stack([np.ones(len(X)), X])   # moi hang [1, x_i]

# Nghiem OLS de doi chieu
w_ols = np.linalg.pinv(Xbar.T @ Xbar) @ Xbar.T @ y
print('OLS: w =', w_ols)


def loss(w, Xb, y):
    return np.sum((Xb @ w - y)**2) / (2 * len(y))


def grad(w, Xb, y):
    return Xb.T @ (Xb @ w - y) / len(y)


def batch_gd(Xb, y, w_init, eta, max_iter=10000, tol=1e-4):
    w = w_init.copy()
    lich_su = [loss(w, Xb, y)]
    for t in range(1, max_iter + 1):
        w = w - eta * grad(w, Xb, y)
        lich_su.append(loss(w, Xb, y))
        if np.linalg.norm(grad(w, Xb, y)) < tol:   # gradient du nho -> dung
            break
    return w, t, lich_su


for eta in [1, 0.1]:
    w, t, lich_su = batch_gd(Xbar, y, np.array([2., 1.]), eta)
    print('eta = %-4s w = %s, sau %d vong lap, loss = %.6f' % (eta, w.round(4), t, lich_su[-1]))

# Toc do hoc qua lon: GD phan ky
with np.errstate(over='ignore', invalid='ignore'):
    w, t, lich_su = batch_gd(Xbar, y, np.array([2., 1.]), 2.5, max_iter=1000)
print('eta = 2.5  w =', w, '-> phan ky')
