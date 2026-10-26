# Muc 4.3.1 - 4.3.2. Du lieu ung thu phoi (Kaggle): kham pha va tien xu ly
# Nguon: https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parent.parent / 'data'

# 4.3.1. Doc va kham pha du lieu
df = pd.read_csv(DATA / 'survey lung cancer.csv')
print(df.shape)                          # (309, 16)
print(df.head())
df.info()
print('Gia tri thieu:', df.isna().sum().sum())
print(df['LUNG_CANCER'].value_counts())  # YES 270, NO 39
print('Ten cot:', df.columns.tolist())   # 'FATIGUE ', 'ALLERGY ' co khoang trang thua

# 4.3.2. Tien xu ly
# 1. Chuan hoa ten cot
df.columns = df.columns.str.strip()

# 2. Xoa dong trung lap
print('So dong trung:', df.duplicated().sum())   # 33
df = df.drop_duplicates()
print(df.shape)                                  # (276, 16)

# 3. Ma hoa ve 0/1
df['GENDER'] = (df['GENDER'] == 'M').astype(int)             # nam = 1, nu = 0
df['LUNG_CANCER'] = (df['LUNG_CANCER'] == 'YES').astype(int) # co benh = 1 (lop duong)
cot_trieu_chung = df.columns.drop(['GENDER', 'AGE', 'LUNG_CANCER'])
df[cot_trieu_chung] = df[cot_trieu_chung] - 1                # 1/2 -> 0/1

print(df['LUNG_CANCER'].value_counts())          # 1: 238, 0: 38
print(df.head(3).T)

# 4. Chia tap huan luyen / kiem tra, giu ti le hai lop
X = df.drop(columns='LUNG_CANCER')
y = df['LUNG_CANCER']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print(len(X_train), len(X_test))        # 220 56
print(y_test.value_counts())            # 1: 48, 0: 8
