# Ve hien tuong qua khop (muc 3.2.1): hoi quy da thuc bac 1, bac 10 va bac 10 co dieu chuan Ridge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]].values
y = df["gia"].values
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tap huan luyen nho: 20 can nha
idx = np.random.RandomState(0).choice(len(df), 20, replace=False)
X_train, y_train = X[idx], y[idx]

mo_hinh = [
    ("Bậc 1", LinearRegression(), 1, "#1F77B4", "-"),
    ("Bậc 10", LinearRegression(), 10, "#D62728", "--"),
    ("Bậc 10 + Ridge", Ridge(alpha=1.0), 10, "#2CA02C", "-"),
]

x0 = np.linspace(X_train.min(), X_train.max(), 400).reshape(-1, 1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(X_test[:300], y_test[:300], '.', color="lightgray", label="Tập kiểm tra (300 căn)")
ax.plot(X_train, y_train, 'ko', label="Tập huấn luyện (20 căn)")
for ten, model, bac, mau, net in mo_hinh:
    p = make_pipeline(PolynomialFeatures(bac, include_bias=False), StandardScaler(), model)
    p.fit(X_train, y_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, p.predict(X_train)))
    rmse_test = np.sqrt(mean_squared_error(y_test, p.predict(X_test)))
    ax.plot(x0, p.predict(x0), color=mau, ls=net, lw=2,
            label=f"{ten}: RMSE train {rmse_train:.0f}, test {rmse_test:.0f}")
    print(ten, rmse_train, rmse_test)

ax.set_ylim(0, 9000)
ax.set_xlabel("Diện tích (m²)"); ax.set_ylabel("Giá (triệu VND)")
ax.set_title("Quá khớp khi dùng đa thức bậc cao trên tập huấn luyện nhỏ")
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("images/qua_khop.png", dpi=150)
