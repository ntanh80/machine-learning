# Muc 5.3.1, Buoc 4. Hoi quy tuyen tinh huan luyen bang SGD voi scikit-learn (SGDRegressor)
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent.parent / 'data'

df = pd.read_csv(DATA / 'gia_nha.csv')
X = df[['dien_tich', 'so_phong_ngu', 'khoang_cach']].values
y = df['gia'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(StandardScaler(),
                      SGDRegressor(max_iter=1000, tol=1e-6, random_state=42))
model.fit(X_train, y_train)
sgd = model[-1]
print('So epoch:', sgd.n_iter_)
print('w0 =', sgd.intercept_, ' w =', sgd.coef_)   # tren du lieu da chuan hoa
print('R2 test =', model.score(X_test, y_test))    # ~0.9685
