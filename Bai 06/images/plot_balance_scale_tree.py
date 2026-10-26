# Cây quyết định (Gini, max_depth = 3, min_samples_leaf = 5) trên dữ liệu Balance Scale (mục 6.2)
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

IMAGES = Path(__file__).resolve().parent   # thư mục images/
DATA = IMAGES.parent / 'data'

data = pd.read_csv(DATA / 'balance_scale.csv')
feature_names = ['left-weight', 'left-distance', 'right-weight', 'right-distance']
X = data[feature_names].to_numpy()
y = data['class'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

model = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_leaf=5, random_state=100)
model.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(15, 6.5))
plot_tree(model, feature_names=feature_names, class_names=list(model.classes_), filled=True,
          rounded=True, fontsize=9, ax=ax)
ax.set_title('Cây quyết định trên dữ liệu Balance Scale (Gini, max_depth = 3): độ chính xác kiểm tra %.3f'
             % model.score(X_test, y_test))
fig.tight_layout()
fig.savefig(IMAGES / 'balance_scale_tree.png', dpi=150)
