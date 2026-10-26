# Muc 4.2.3 e). Perceptron bang thu vien scikit-learn, so sanh voi hoi quy Logistic
import numpy as np
from sklearn.linear_model import Perceptron, LogisticRegression

np.random.seed(2)
means = [[-1, 0], [1, 0]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)   # lop +1
X1 = np.random.multivariate_normal(means[1], cov, N)   # lop -1
X = np.concatenate((X0, X1), axis=0)
y = np.concatenate((np.ones(N), -1 * np.ones(N)))

clf = Perceptron(max_iter=1000, random_state=0).fit(X, y)
print('Perceptron: w0 =', clf.intercept_, ' w =', clf.coef_, ' Accuracy =', clf.score(X, y))

# Hoi quy Logistic tren cung du lieu: co them xac suat du doan
lr = LogisticRegression().fit(X, y)
print('Logistic  : w0 =', lr.intercept_, ' w =', lr.coef_, ' Accuracy =', lr.score(X, y))
print('Xac suat lop +1 cua 3 diem dau:', lr.predict_proba(X[:3])[:, 1].round(3))
