# Muc 4.2.3 d). Thuat toan PLA: in tung buoc cua vi du tinh tay voi 4 diem
import numpy as np

# Diem A, B (lop +1) va C, D (lop -1)
ten = ['A', 'B', 'C', 'D']
X = np.array([[1, 1], [2, 1], [1, -1], [-1, -2]])
y = np.array([1, 1, -1, -1])
Xbar = np.column_stack([np.ones(len(X)), X])   # vector mo rong [1, x1, x2]


def sgn(z):
    return 1 if z >= 0 else -1                 # quy uoc sgn(0) = 1


w = np.zeros(3)                                # w = [w0, w1, w2]
vong = 0
while True:
    vong += 1
    co_cap_nhat = False
    print('Vong %d:' % vong)
    for i in range(len(y)):                    # duyet cac diem theo thu tu
        z = Xbar[i] @ w
        if sgn(z) != y[i]:
            w = w + y[i] * Xbar[i]             # w <- w + y_i x_i
            co_cap_nhat = True
            print('  %s: z = %5.1f -> SAI, cap nhat w = %s' % (ten[i], z, w))
        else:
            print('  %s: z = %5.1f -> dung' % (ten[i], z))
    if not co_cap_nhat:
        break

print('Nghiem w =', w, '-> sieu phang %.0f + %.0f x1 + %.0f x2 = 0' % tuple(w))
