# Muc 4.1.4 d). Hoi quy Logistic bang thu vien scikit-learn, doi chieu voi nghiem Gradient Descent
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# penalty=None: khong dieu chuan, de so sanh voi nghiem MLE (w0 = -4.0777, w1 = 1.5046)
model = LogisticRegression(penalty=None)
model.fit(X, y)                         # sklearn can X dang ma tran (N, d)
print('Khong dieu chuan: w0 =', model.intercept_, ' w1 =', model.coef_)

print(model.predict_proba([[2], [4]]))  # cot 0: P(y=0), cot 1: P(y=1)
print(model.predict([[2], [4]]))        # [0 1]

# Mac dinh C = 1.0 (co dieu chuan L2, C = 1/lambda): he so bi co lai
model_l2 = LogisticRegression().fit(X, y)
print('Mac dinh C = 1  : w0 =', model_l2.intercept_, ' w1 =', model_l2.coef_)
