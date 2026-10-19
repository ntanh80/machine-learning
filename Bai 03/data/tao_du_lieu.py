# Sinh du lieu mau 1000 can nha cho bai toan du doan gia nha (Bai 03 - Hoi quy tuyen tinh)
#   x1: dien tich (m2)
#   x2: so phong ngu (phong)
#   x3: khoang cach toi trung tam (km)
#   y : gia nha (trieu VND)
# Gia nha tang theo dien tich, so phong ngu va giam khi xa trung tam:
#   y = 500 + 30*x1 + 150*x2 - 60*x3 + nhieu
import os
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

dien_tich = np.round(np.random.uniform(30, 200, N), 1)
# So phong ngu phu thuoc vao dien tich (nha rong thuong nhieu phong hon)
so_phong_ngu = np.clip(np.round(dien_tich / 40 + np.random.normal(0, 0.7, N)), 1, 6).astype(int)
khoang_cach = np.round(np.random.uniform(0.5, 25, N), 1)

nhieu = np.random.normal(0, 300, N)
gia = 500 + 30 * dien_tich + 150 * so_phong_ngu - 60 * khoang_cach + nhieu
gia = np.round(np.maximum(gia, 300), 0)

df = pd.DataFrame({
    'dien_tich': dien_tich,
    'so_phong_ngu': so_phong_ngu,
    'khoang_cach': khoang_cach,
    'gia': gia,
})

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gia_nha.csv')
df.to_csv(out, index=False)
print('Da luu', len(df), 'mau vao', out)
print(df.describe())
