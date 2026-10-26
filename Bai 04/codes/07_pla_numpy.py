# Muc 4.2.3 e). Cai dat thuat toan PLA bang NumPy tren du lieu hai chieu sinh ngau nhien
import numpy as np


def predict(w, Xbar):
    """Du doan nhan +1/-1 cho moi hang cua Xbar (quy uoc sgn(0) = 1)."""
    return np.where(Xbar @ w >= 0, 1, -1)


def perceptron(Xbar, y, w_init, max_iter=1000):
    """Thuat toan PLA. Xbar: (N, d+1) da them cot 1; y: nhan +1/-1."""
    w = w_init.copy()
    for t in range(max_iter):
        sai = np.where(predict(w, Xbar) != y)[0]   # chi so cac diem bi phan loai sai
        if len(sai) == 0:                          # khong con diem sai -> dung
            return w, t
        i = np.random.choice(sai)                  # chon ngau nhien mot diem sai
        w = w + y[i] * Xbar[i]                     # cap nhat w <- w + y_i x_i
    return w, max_iter


# Hai nhom diem sinh ngau nhien theo phan phoi chuan
np.random.seed(2)
means = [[-1, 0], [1, 0]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)   # lop +1
X1 = np.random.multivariate_normal(means[1], cov, N)   # lop -1
X = np.concatenate((X0, X1), axis=0)
y = np.concatenate((np.ones(N), -1 * np.ones(N)))
Xbar = np.concatenate((np.ones((2 * N, 1)), X), axis=1)

w, so_lan = perceptron(Xbar, y, np.random.randn(3))
print('w =', w, '- so lan cap nhat:', so_lan)
print('So diem sai:', np.sum(predict(w, Xbar) != y))   # 0

# Du lieu XOR khong tach biet tuyen tinh: PLA khong hoi tu, dung khi het max_iter
X_xor = np.array([[1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
y_xor = np.array([-1, -1, 1, 1])
w, so_lan = perceptron(X_xor, y_xor, np.zeros(3), max_iter=1000)
print('XOR: so lan lap =', so_lan, '- so diem sai:', np.sum(predict(w, X_xor) != y_xor))
