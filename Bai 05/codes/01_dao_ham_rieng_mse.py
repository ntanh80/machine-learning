# Muc 5.1.2. Dao ham rieng cua ham mat mat MSE va kiem tra gradient bang dao ham so
# Du lieu 3 diem (1, 2), (2, 3), (3, 5); mo hinh y = w0 + w1 x
import numpy as np

X = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)   # moi hang [1, x_i]
y = np.array([2, 3, 5], dtype=float)


def loss(w):
    return np.sum((X @ w - y)**2) / (2 * len(y))


def grad(w):
    # (1/N) X^T (Xw - y): thanh phan j la (1/N) sum(e_i * x_ij)
    return X.T @ (X @ w - y) / len(y)


def grad_so(w, eps=1e-6):
    # Xap xi sai phan trung tam cho tung dao ham rieng
    g = np.zeros_like(w)
    for j in range(len(w)):
        e = np.zeros_like(w); e[j] = eps
        g[j] = (loss(w + e) - loss(w - e)) / (2 * eps)
    return g


w = np.zeros(2)
e = X @ w - y
print('Sai so e         :', e)                  # [-2 -3 -5]
print('L(0, 0)          : %.3f' % loss(w))      # 6.333
print('dL/dw0 = mean(e) : %.3f' % e.mean())     # -3.333
print('dL/dw1 = mean(ex): %.3f' % np.mean(e * X[:, 1]))  # -7.667
print('grad (cong thuc) :', grad(w))
print('grad (dao ham so):', grad_so(w))
