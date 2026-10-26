# Muc 4.3.3 - 4.3.4. Xay dung va danh gia mo hinh hoi quy Logistic du doan ung thu phoi
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, classification_report)

DATA = Path(__file__).resolve().parent.parent / 'data'

# Tien xu ly (giong file 09)
df = pd.read_csv(DATA / 'survey lung cancer.csv')
df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df['GENDER'] = (df['GENDER'] == 'M').astype(int)
df['LUNG_CANCER'] = (df['LUNG_CANCER'] == 'YES').astype(int)
cot_trieu_chung = df.columns.drop(['GENDER', 'AGE', 'LUNG_CANCER'])
df[cot_trieu_chung] = df[cot_trieu_chung] - 1

X = df.drop(columns='LUNG_CANCER')
y = df['LUNG_CANCER']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 4.3.3. Xay dung mo hinh: chuan hoa + hoi quy Logistic trong mot Pipeline
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)                 # nhan 0/1 (nguong 0.5)
y_prob = model.predict_proba(X_test)[:, 1]     # xac suat mac benh

lr = model[-1]
he_so = pd.Series(lr.coef_[0], index=X.columns).sort_values(ascending=False)
print('w0 =', lr.intercept_)
print(he_so.round(3))

# 4.3.4 a) Ma tran nham lan va cac chi so
print('\n--- Mo hinh mac dinh ---')
print(confusion_matrix(y_test, y_pred))
print('Accuracy :', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall   :', recall_score(y_test, y_pred))
print('F1       :', f1_score(y_test, y_pred))
print('ROC-AUC  :', roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred, digits=3))

# So sanh voi mo hinh "ngay tho" luon du doan "co benh"
dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
y_dummy = dummy.predict(X_test)
print('Luon du doan co benh: Accuracy = %.3f, F1 = %.3f'
      % (accuracy_score(y_test, y_dummy), f1_score(y_test, y_dummy)))

# 4.3.4 b) Xu ly mat can bang lop
model_cb = make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight='balanced', max_iter=1000))
model_cb.fit(X_train, y_train)
y_pred_cb = model_cb.predict(X_test)
print("\n--- class_weight='balanced' ---")
print(confusion_matrix(y_test, y_pred_cb))
print('Accuracy = %.3f, Precision = %.3f, Recall = %.3f, F1 = %.3f, ROC-AUC = %.3f' % (
    accuracy_score(y_test, y_pred_cb), precision_score(y_test, y_pred_cb),
    recall_score(y_test, y_pred_cb), f1_score(y_test, y_pred_cb),
    roc_auc_score(y_test, model_cb.predict_proba(X_test)[:, 1])))

# 4.3.4 c) Chon nguong phan loai
print('\n--- Thay doi nguong ---')
for tau in [0.5, 0.7, 0.8, 0.9]:
    y_tau = (y_prob >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_tau).ravel()
    print(tau, 'TN=%d FP=%d FN=%d TP=%d' % (tn, fp, fn, tp),
          'Recall=%.3f Dac hieu=%.3f' % (tp / (tp + fn), tn / (tn + fp)))

# 4.3.4 d) Kiem dinh cheo 5 phan
print('\n--- Kiem dinh cheo ---')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for chi_so in ['accuracy', 'f1', 'roc_auc']:
    diem = cross_val_score(model, X, y, cv=cv, scoring=chi_so)
    print(chi_so, diem.round(3), 'trung binh = %.3f' % diem.mean())
