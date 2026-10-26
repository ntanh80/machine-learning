# Muc 4.1.5. Ma tran nham lan cho mo hinh du doan thi do (nguong 0.5)
import numpy as np
from sklearn.metrics import confusion_matrix

X = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
w0, w1 = -4.0777, 1.5046                      # nghiem tim duoc o file 02

p = 1 / (1 + np.exp(-(w0 + w1 * X)))          # xac suat thi do
y_pred = (p >= 0.5).astype(int)               # du doan do khi so gio >= 2.75

# Liet ke tung mau va loai ket qua
for gio, that, du_doan in zip(X, y, y_pred):
    loai = {(1, 1): 'TP', (0, 0): 'TN', (0, 1): 'FP', (1, 0): 'FN'}[(that, du_doan)]
    print('%.2f gio: that = %d, du doan = %d -> %s' % (gio, that, du_doan, loai))

# Ma tran nham lan: hang la nhan that, cot la nhan du doan, lop 0 truoc
cm = confusion_matrix(y, y_pred)
print(cm)                                     # [[8 2] [2 8]]
tn, fp, fn, tp = cm.ravel()
print('TN = %d, FP = %d, FN = %d, TP = %d' % (tn, fp, fn, tp))
