# Muc 4.1.2. Ham sigmoid va dao ham cua no
import numpy as np


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def dao_ham_sigmoid(s):
    # sigma'(s) = sigma(s) * (1 - sigma(s))
    return sigmoid(s) * (1 - sigmoid(s))


s = np.array([-5, -1, 0, 1, 5])
print('s          :', s)
print('sigmoid(s) :', sigmoid(s).round(4))          # [0.0067 0.2689 0.5 0.7311 0.9933]
print("sigmoid'(s):", dao_ham_sigmoid(s).round(4))  # cuc dai 0.25 tai s = 0

# Kiem tra tinh chat doi xung: sigma(-s) = 1 - sigma(s)
print('sigma(-s) == 1 - sigma(s):', np.allclose(sigmoid(-s), 1 - sigmoid(s)))

# Kiem tra cong thuc dao ham bang sai phan trung tam
eps = 1e-6
dao_ham_so = (sigmoid(s + eps) - sigmoid(s - eps)) / (2 * eps)
print('Dao ham so khop cong thuc:', np.allclose(dao_ham_so, dao_ham_sigmoid(s)))
