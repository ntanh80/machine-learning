# Muc 4.1.4. Hoi quy Logistic: ham mat mat cross-entropy (MLE) va toi uu bang Gradient Descent
# Vi du: so gio on tap va ket qua thi cua 20 sinh vien
import numpy as np

# Du lieu: so gio on tap va ket qua thi (1: do, 0: truot)
X = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# Vector mo rong: them cot 1 vao truoc -> moi hang la [1, x_i]
Xbar = np.column_stack([np.ones(len(X)), X])


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def loss(w, Xbar, y):
    a = sigmoid(Xbar @ w)
    return -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))


def logistic_gd(Xbar, y, eta=0.5, so_vong_lap=2000):
    w = np.zeros(Xbar.shape[1])
    lich_su = [loss(w, Xbar, y)]
    for _ in range(so_vong_lap):
        a = sigmoid(Xbar @ w)                    # xac suat du doan cho N mau
        grad = Xbar.T @ (a - y) / len(y)         # gradient = (1/N) X^T (a - y)
        w = w - eta * grad                       # cap nhat trong so
        lich_su.append(loss(w, Xbar, y))
    return w, lich_su


# 1. Tinh tay mot buoc lap tu w = [0, 0], eta = 0.5
w = np.zeros(2)
grad = Xbar.T @ (sigmoid(Xbar @ w) - y) / len(y)
print('J(0)      = %.4f (= ln 2)' % loss(w, Xbar, y))
print('gradient  =', grad)                      # [0, -0.50625]
w1 = w - 0.5 * grad
print('w sau 1 buoc =', w1, ' J = %.4f' % loss(w1, Xbar, y))

# 2. Chay Gradient Descent den khi hoi tu
w, lich_su = logistic_gd(Xbar, y)
print('\nw0 = %.4f, w1 = %.4f' % (w[0], w[1]))   # -4.0777, 1.5046
print('J(w) = %.4f' % lich_su[-1])               # 0.4015
print('Ranh gioi quyet dinh: x = %.2f gio' % (-w[0] / w[1]))
print('Moi gio on them nhan odds thi do len %.2f lan' % np.exp(w[1]))

# 3. Xac suat thi do theo so gio
for gio in [1, 2, 3, 4, 5]:
    print(gio, 'gio -> P(do) = %.3f' % sigmoid(w[0] + w[1] * gio))
