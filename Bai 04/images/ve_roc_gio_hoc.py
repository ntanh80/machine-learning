# Ve duong cong ROC cho mo hinh hoi quy Logistic o vi du gio on tap (muc 4.1.6)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

X = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
w0, w1 = -4.0777, 1.5046          # nghiem tim duoc o muc 4.1.4
p = 1 / (1 + np.exp(-(w0 + w1 * X)))

fpr, tpr, nguong = roc_curve(y, p)
auc = roc_auc_score(y, p)
print("AUC =", auc)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.fill_between(fpr, tpr, alpha=0.15, color="#1F77B4")
ax.plot(fpr, tpr, color="#1F77B4", lw=2.5, label=f"Hồi quy Logistic (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "--", color="#7F7F7F", label="Đoán ngẫu nhiên (AUC = 0.5)")
# Diem ung voi nguong 0.5: TPR = 0.8, FPR = 0.2
ax.plot(0.2, 0.8, "o", color="#D62728", ms=9)
ax.annotate("ngưỡng 0.5\n(FPR = 0.2, TPR = 0.8)", (0.2, 0.8), (0.35, 0.55),
            arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("FPR (tỉ lệ dương tính giả)"); ax.set_ylabel("TPR (Recall)")
ax.set_title("Đường cong ROC")
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
fig.tight_layout()
fig.savefig("images/roc_gio_hoc.png", dpi=150)
