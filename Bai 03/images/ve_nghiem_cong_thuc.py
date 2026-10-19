# Ve duong hoi quy tim duoc bang cong thuc (muc 3.3) va du doan tren tap kiem tra
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[147, 150, 153, 158, 163, 165, 168, 170,
               173, 175, 178, 180, 183]]).T
y = np.array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# w = (Xbar^T Xbar)^+ Xbar^T y
Xbar = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
w_0, w_1 = np.linalg.pinv(Xbar.T @ Xbar) @ Xbar.T @ y

# Tap kiem tra
x_test = np.array([155, 160])
y_test = np.array([52, 56])
y_pred = w_0 + w_1 * x_test

x0 = np.linspace(145, 185, 2)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(X, y, 'ro', label="Tập huấn luyện")
ax.plot(x0, w_0 + w_1 * x0, color="#1F77B4", lw=2,
        label=rf"Đường hồi quy: $\hat{{y}} = {w_1:.4f}x {w_0:+.4f}$")
ax.plot(x_test, y_test, 's', color="#2CA02C", ms=8, label="Tập kiểm tra (thực tế)")
ax.plot(x_test, y_pred, 'x', color="black", ms=10, mew=2, label="Tập kiểm tra (dự đoán)")
for xt, yt, yp in zip(x_test, y_test, y_pred):
    ax.vlines(xt, min(yt, yp), max(yt, yp), color="gray", ls=":")
    ax.annotate(f"{xt} cm: thực tế {yt} kg, dự đoán {yp:.2f} kg", xy=(xt, yp),
                xytext=(xt + 1.5, yp - 3.5), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"))

ax.set_xlabel("Chiều cao (cm)"); ax.set_ylabel("Cân nặng (kg)")
ax.set_title("Nghiệm theo công thức và dự đoán trên tập kiểm tra")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("images/nghiem_cong_thuc.png", dpi=150)
print(w_0, w_1, y_pred)
