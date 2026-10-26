# Muc 4.1.6. Cac chi so danh gia: Accuracy, Precision, Recall, F1-Score, ROC-AUC
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, roc_curve, roc_auc_score)

X = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
w0, w1 = -4.0777, 1.5046                      # nghiem tim duoc o file 02

p = 1 / (1 + np.exp(-(w0 + w1 * X)))          # xac suat du doan lop 1
y_pred = (p >= 0.5).astype(int)

# 1. Tinh tay tu TP, TN, FP, FN
tp = np.sum((y == 1) & (y_pred == 1))
tn = np.sum((y == 0) & (y_pred == 0))
fp = np.sum((y == 0) & (y_pred == 1))
fn = np.sum((y == 1) & (y_pred == 0))
print('Accuracy  = %.3f' % ((tp + tn) / (tp + tn + fp + fn)))
print('Precision = %.3f' % (tp / (tp + fp)))
print('Recall    = %.3f' % (tp / (tp + fn)))
print('F1        = %.3f' % (2 * tp / (2 * tp + fp + fn)))

# 2. Dung scikit-learn
print('\nAccuracy :', accuracy_score(y, y_pred))    # 0.8
print('Precision:', precision_score(y, y_pred))   # 0.8
print('Recall   :', recall_score(y, y_pred))      # 0.8
print('F1       :', f1_score(y, y_pred))          # 0.8
print(classification_report(y, y_pred))           # bao cao cho tung lop

# 3. Duong cong ROC va AUC (phai dung xac suat, khong dung nhan 0/1)
fpr, tpr, nguong = roc_curve(y, p)
for a, b, t in zip(fpr, tpr, nguong):
    print('nguong %.3f -> FPR = %.2f, TPR = %.2f' % (t, a, b))
print('AUC =', roc_auc_score(y, p))               # 0.895

# 4. Minh hoa han che cua Accuracy voi du lieu mat can bang:
# 1000 nguoi, 10 nguoi benh, mo hinh luon tra loi "khong benh"
y_benh = np.r_[np.ones(10), np.zeros(990)]
y_luon_0 = np.zeros(1000)
print('\nAccuracy = %.2f, Recall = %.2f' % (accuracy_score(y_benh, y_luon_0),
                                           recall_score(y_benh, y_luon_0)))
