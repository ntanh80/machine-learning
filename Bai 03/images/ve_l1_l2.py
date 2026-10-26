# Ve minh hoa hinh hoc cua dieu chuan L1 va L2 (muc 3.2.3)
import numpy as np
import matplotlib.pyplot as plt

# Ham mat mat MSE co dang elip, tam (nghiem OLS) tai w_ols
w_ols = np.array([2.0, 0.4])
H = np.array([[1.0, 0.3], [0.3, 1.2]])
w1, w2 = np.meshgrid(np.linspace(-2, 3.5, 400), np.linspace(-2, 3, 400))
D = np.stack([w1 - w_ols[0], w2 - w_ols[1]], axis=-1)
L = np.einsum('...i,ij,...j->...', D, H, D)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, ten in zip(axes, ["L1 (Lasso)", "L2 (Ridge)"]):
    ax.contour(w1, w2, L, levels=[0.3, 0.8, 1.165, 1.432, 2.5], colors="#1F77B4", linewidths=1)
    ax.plot(*w_ols, 'o', color="#1F77B4")
    ax.annotate("Nghiệm OLS", w_ols, xytext=(w_ols[0] + 0.2, w_ols[1] + 0.3))
    if ten.startswith("L1"):
        t = 1.0
        ax.fill([t, 0, -t, 0], [0, t, 0, -t], color="#FF7F0E", alpha=0.35)
        # Diem tren bien hinh thoi co gia tri ham mat mat nho nhat
        s = np.linspace(0, 1, 2001)[:, None]
        b = np.concatenate([(1 - s) * [t, 0] + s * [0, t], (1 - s) * [0, t] + s * [-t, 0],
                            (1 - s) * [-t, 0] + s * [0, -t], (1 - s) * [0, -t] + s * [t, 0]])
        c = b - w_ols
        p = b[np.argmin(np.einsum('ni,ij,nj->n', c, H, c))]
        print("Nghiem Lasso:", p)
        ax.plot(*p, 'o', color="#D62728")
        ax.annotate("Nghiệm Lasso\n(w₂ = 0)", p, xytext=(1.2, -1.2),
                    arrowprops=dict(arrowstyle="->", color="gray"))
        ax.set_title(r"Ràng buộc L1: $|w_1| + |w_2| \leq t$")
    else:
        g = np.linspace(0, 2 * np.pi, 200)
        ax.fill(np.cos(g), np.sin(g), color="#2CA02C", alpha=0.35)
        # Diem tren duong tron co gia tri ham mat mat nho nhat
        c = np.stack([np.cos(g), np.sin(g)], axis=1) - w_ols
        k = np.argmin(np.einsum('ni,ij,nj->n', c, H, c))
        ax.plot(np.cos(g[k]), np.sin(g[k]), 'o', color="#D62728")
        ax.annotate("Nghiệm Ridge\n(w₁, w₂ ≠ 0)", (np.cos(g[k]), np.sin(g[k])), xytext=(1.2, -1.2),
                    arrowprops=dict(arrowstyle="->", color="gray"))
        ax.set_title(r"Ràng buộc L2: $w_1^2 + w_2^2 \leq t^2$")
    ax.axhline(0, color="black", lw=0.8); ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("$w_1$"); ax.set_ylabel("$w_2$")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("images/l1_l2.png", dpi=150)
