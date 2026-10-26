# Muc 5.1.3. Cap nhat trong so trong Gradient Descent
# a) Tinh tay voi f(x) = (x + 2)^2; b) mot buoc GD cho MSE; c) f(x) = x^2 + 5 sin(x)
import numpy as np

# a) f(x) = x^2 + 4x + 4, f'(x) = 2x + 4, x0 = 0, eta = 0.25
print('a) f(x) = x^2 + 4x + 4')
x = 0.0
for t in range(4):
    g = 2 * x + 4
    x_moi = x - 0.25 * g                     # x_{t+1} = x_t - eta * f'(x_t)
    print('t = %d: x = %7.4f, f\'(x) = %6.3f -> x moi = %7.4f, f = %.4f'
          % (t, x, g, x_moi, (x_moi + 2)**2))
    x = x_moi

# b) Mot buoc GD cho MSE voi 3 diem (1, 2), (2, 3), (3, 5), eta = 0.1
print('\nb) Mot buoc GD cho ham mat mat MSE')
X = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
y = np.array([2, 3, 5], dtype=float)
loss = lambda w: np.sum((X @ w - y)**2) / (2 * len(y))
w = np.zeros(2)
g = X.T @ (X @ w - y) / len(y)
w_moi = w - 0.1 * g                          # cap nhat dong thoi moi tham so
print('w = %s, L = %.3f  ->  w = %s, L = %.3f' % (w, loss(w), w_moi.round(3), loss(w_moi)))


# c) f(x) = x^2 + 5 sin(x) voi hai diem khoi tao
print('\nc) f(x) = x^2 + 5 sin(x)')


def f(x):
    return x**2 + 5 * np.sin(x)


def grad(x):
    return 2 * x + 5 * np.cos(x)


def gd_mot_bien(x0, eta, max_iter=100, tol=1e-3):
    x = x0
    lich_su = [x0]
    for t in range(max_iter):
        x = x - eta * grad(x)        # x_{t+1} = x_t - eta * f'(x_t)
        lich_su.append(x)
        if abs(grad(x)) < tol:       # dao ham du nho -> dung
            break
    return lich_su


for x0 in [-5, 5]:
    h = gd_mot_bien(x0, eta=0.1)
    print('x0 = %2d: x = %.4f, f(x) = %.4f, sau %d vong lap' % (x0, h[-1], f(h[-1]), len(h) - 1))
