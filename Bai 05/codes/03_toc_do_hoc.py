# Muc 5.1.4. Toc do hoc: hoi tu cham, dao dong, phan ky; mac ket tai cuc tieu dia phuong
import numpy as np


def gd(grad, x0, eta, max_iter=100, tol=1e-3):
    x = x0
    lich_su = [x0]
    for _ in range(max_iter):
        x = x - eta * grad(x)
        lich_su.append(x)
        if abs(grad(x)) < tol or not np.isfinite(x):
            break
    return lich_su


# a) f(x) = x^2 + 5 sin(x), x0 = -5, cac toc do hoc khac nhau
f = lambda x: x**2 + 5 * np.sin(x)
df = lambda x: 2 * x + 5 * np.cos(x)
print('a) f(x) = x^2 + 5 sin(x), x0 = -5')
for eta in [0.01, 0.1, 0.5]:
    h = gd(df, -5, eta)
    print('eta = %-4s: %3d vong lap, x = %7.4f, f(x) = %7.4f' % (eta, len(h) - 1, h[-1], f(h[-1])))

# b) f(x) = (x + 2)^2: khoang cach toi cuc tieu nhan voi (1 - 2*eta) sau moi buoc
print('\nb) f(x) = (x + 2)^2, x0 = 0, 6 vong lap')
for eta in [0.25, 0.5, 0.75, 1.0, 1.1]:
    h = gd(lambda x: 2 * x + 4, 0.0, eta, max_iter=6, tol=0)
    print('eta = %-4s (he so %5.2f): x_t = %s' % (eta, 1 - 2 * eta, np.round(h, 3)))

# c) Cuc tieu dia phuong: f(x) = x^4 - 3x^2 + x
g = lambda x: x**4 - 3 * x**2 + x
dg = lambda x: 4 * x**3 - 6 * x + 1
print('\nc) f(x) = x^4 - 3x^2 + x, eta = 0.05')
for x0 in [-2.0, 2.0]:
    h = gd(dg, x0, 0.05, max_iter=200, tol=1e-6)
    print('x0 = %4.1f -> x = %7.4f, f(x) = %7.4f' % (x0, h[-1], g(h[-1])))
print('Cac diem dung (f\'(x) = 0):', np.sort(np.roots([4, 0, -6, 1])).round(4))
