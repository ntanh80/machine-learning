# Anh huong cua chuan hoa dac trung den Gradient Descent tren du lieu gia nha (muc 5.3.1)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich", "so_phong_ngu", "khoang_cach"]].values
y = df["gia"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def them_cot_1(X):
    return np.column_stack([np.ones(len(X)), X])

def cost(w, Xb, y):
    return .5 / len(y) * np.sum((Xb @ w - y)**2)

def batch_gd(Xb, y, eta, so_vong_lap):
    w = np.zeros(Xb.shape[1])
    mat_mat = [cost(w, Xb, y)]
    for _ in range(so_vong_lap):
        w = w - eta * Xb.T @ (Xb @ w - y) / len(y)
        mat_mat.append(cost(w, Xb, y))
    return mat_mat

mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
Xb_goc = them_cot_1(X_train)
Xb_chuan = them_cot_1((X_train - mu) / sd)
w_ols = np.linalg.pinv(Xb_goc.T @ Xb_goc) @ Xb_goc.T @ y_train
J_min = cost(w_ols, Xb_goc, y_train)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(batch_gd(Xb_goc, y_train, 1e-4, 300), color="#D62728", lw=2,
        label=r"Dữ liệu gốc, $\eta = 10^{-4}$")
ax.plot(batch_gd(Xb_chuan, y_train, 0.5, 300), color="#2CA02C", lw=2,
        label=r"Đã chuẩn hóa, $\eta = 0.5$")
ax.axhline(J_min, color="black", ls=":", label="Giá trị nhỏ nhất (OLS)")
ax.set_yscale("log")
ax.set_xlabel("Vòng lặp"); ax.set_ylabel(r"$J(\mathbf{w})$ (thang log)")
ax.set_title("Batch GD trên dữ liệu giá nhà: có và không chuẩn hóa")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("images/chuan_hoa.png", dpi=150)
