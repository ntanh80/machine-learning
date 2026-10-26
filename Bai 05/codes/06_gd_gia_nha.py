# Muc 5.3.1, Buoc 3. Gradient Descent tren du lieu gia nha: vai tro cua chuan hoa dac trung
# va so sanh Batch, SGD, Mini-batch voi nghiem OLS
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parent.parent / 'data'

df = pd.read_csv(DATA / 'gia_nha.csv')
X = df[['dien_tich', 'so_phong_ngu', 'khoang_cach']].values
y = df['gia'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def them_cot_1(X):
    return np.column_stack([np.ones(len(X)), X])


def loss(w, Xb, y):
    return np.sum((Xb @ w - y)**2) / (2 * len(y))


def grad(w, Xb, y):
    return Xb.T @ (Xb @ w - y) / len(y)


def minibatch_gd(Xb, y, eta, batch_size, so_epoch, seed=0):
    rng = np.random.default_rng(seed)
    N = len(y)
    w = np.zeros(Xb.shape[1])
    lich_su = [loss(w, Xb, y)]
    for epoch in range(so_epoch):
        thu_tu = rng.permutation(N)
        for s in range(0, N, batch_size):
            B = thu_tu[s:s + batch_size]
            w = w - eta * grad(w, Xb[B], y[B])
        lich_su.append(loss(w, Xb, y))
    return w, lich_su


def danh_gia(w, Xb, y):
    pred = Xb @ w
    rmse = np.sqrt(np.mean((pred - y)**2))
    r2 = 1 - np.sum((pred - y)**2) / np.sum((y - y.mean())**2)
    return rmse, r2


# Nghiem OLS de doi chieu
Xb_train = them_cot_1(X_train)
w_ols = np.linalg.pinv(Xb_train.T @ Xb_train) @ Xb_train.T @ y_train
print('OLS: w = %s, RMSE test = %.2f, R2 = %.4f'
      % (w_ols.round(2), *danh_gia(w_ols, them_cot_1(X_test), y_test)))

# 1. GD tren du lieu goc (chua chuan hoa)
with np.errstate(over='ignore', invalid='ignore'):
    w, lich_su = minibatch_gd(Xb_train, y_train, eta=3e-4, batch_size=len(y_train), so_epoch=50)
print('\nDu lieu goc, eta = 3e-4: loss =', np.round(lich_su[:4], 1), '... -> phan ky')
w, lich_su = minibatch_gd(Xb_train, y_train, eta=1e-4, batch_size=len(y_train), so_epoch=1000)
print('Du lieu goc, eta = 1e-4, 1000 vong lap: w =', w.round(2), '-> con rat xa OLS')

# 2. Chuan hoa dac trung (chi dung thong ke cua tap huan luyen)
mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
Zb_train = them_cot_1((X_train - mu) / sd)
Zb_test = them_cot_1((X_test - mu) / sd)

print()
for ten, bs, eta, so_epoch in [('Batch', len(y_train), 0.5, 100),
                               ('SGD', 1, 0.01, 10),
                               ('Mini-batch', 64, 0.1, 10)]:
    w, lich_su = minibatch_gd(Zb_train, y_train, eta, bs, so_epoch)
    # Doi trong so ve thang do goc: w_j = w'_j / sd_j, w_0 = w'_0 - sum(w'_j * mu_j / sd_j)
    w_goc = np.r_[w[0] - np.sum(w[1:] * mu / sd), w[1:] / sd]
    rmse, r2 = danh_gia(w, Zb_test, y_test)
    print('%-10s loss = %.1f, w (thang goc) = %s, RMSE test = %.2f, R2 = %.4f'
          % (ten, lich_su[-1], w_goc.round(2), rmse, r2))
