# Minh hoa vai tro cua he so chech w0 trong hoi quy tuyen tinh
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
x = rng.uniform(30, 120, 40)                   # dien tich (m2)
y = 800 + 15 * x + rng.normal(0, 120, x.size)  # gia nha (trieu dong)

# Co w0: nghiem binh phuong toi thieu
X_bar = np.column_stack([np.ones_like(x), x])
w0, w1 = np.linalg.lstsq(X_bar, y, rcond=None)[0]
# Khong co w0: duong thang bat buoc di qua goc toa do
w_no = (x @ y) / (x @ x)

xs = np.linspace(0, 130, 200)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, s=25, color="#4C72B0", alpha=0.8, label="Dữ liệu")
ax.plot(xs, w0 + w1 * xs, color="#2CA02C", lw=2.2,
        label=rf"Có $w_0$: $\hat{{y}} = {w0:.0f} + {w1:.1f}x$")
ax.plot(xs, w_no * xs, color="#D62728", lw=2.2, ls="--",
        label=rf"Không có $w_0$: $\hat{{y}} = {w_no:.1f}x$")
ax.scatter([0], [0], color="#D62728", zorder=5)
ax.annotate("Buộc đi qua gốc (0, 0)", xy=(0, 0), xytext=(22, 120),
            arrowprops=dict(arrowstyle="->", color="#D62728"), color="#D62728")
ax.annotate("", xy=(0, w0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="<->", color="#2CA02C", lw=1.5))
ax.text(2, w0 / 2, rf"$w_0 \approx {w0:.0f}$", color="#2CA02C", va="center")

ax.set_xlim(-5, 130); ax.set_ylim(-150, 3000)
ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
ax.set_xlabel("Diện tích $x$ (m²)"); ax.set_ylabel("Giá nhà $y$ (triệu đồng)")
ax.set_title("Vai trò của hệ số chệch $w_0$")
ax.legend(loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("images/he_so_chech.png", dpi=150)
print(w0, w1, w_no)
