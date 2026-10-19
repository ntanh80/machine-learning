# Demo Streamlit: dự đoán giá nhà bằng hồi quy tuyến tính
# Chạy: streamlit run "7. app_gia_nha.py"
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

DAC_TRUNG = ['dien_tich', 'so_phong_ngu', 'khoang_cach']


@st.cache_data
def doc_du_lieu():
    return pd.read_csv(Path(__file__).parent / 'data' / 'gia_nha.csv')


@st.cache_resource
def huan_luyen(df):
    X = df[DAC_TRUNG].values   # mỗi hàng là một mẫu
    y = df['gia'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cách 1: scikit-learn
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Cách 2: tính thủ công w = (Xbar^T Xbar)^+ Xbar^T y
    Xbar = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
    w = np.linalg.pinv(Xbar.T @ Xbar) @ Xbar.T @ y_train
    return model, w, len(X_train)


st.set_page_config(page_title='Dự đoán giá nhà', page_icon=':material/home:')
st.title('Dự đoán giá nhà bằng hồi quy tuyến tính')
st.latex(r'\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3')

df = doc_du_lieu()
model, w, so_mau = huan_luyen(df)

# ---- Phần 1: So sánh hệ số ----
st.subheader('Hệ số của mô hình')
st.caption(f'Huấn luyện trên {so_mau} căn nhà (80% dữ liệu).')

bang_he_so = pd.DataFrame({
    'Ý nghĩa': ['Hệ số chệch', 'Diện tích (m²)', 'Số phòng ngủ', 'Khoảng cách (km)'],
    'scikit-learn': np.r_[model.intercept_, model.coef_],
    'Tính thủ công': w,
}, index=['w0', 'w1', 'w2', 'w3'])
st.dataframe(bang_he_so.style.format({'scikit-learn': '{:.4f}', 'Tính thủ công': '{:.4f}'}))

if np.allclose(bang_he_so['scikit-learn'], bang_he_so['Tính thủ công']):
    st.success('Hai cách cho cùng một nghiệm.', icon=':material/check_circle:')

# ---- Phần 2: Dự đoán giá cho căn nhà mới ----
st.subheader('Dự đoán giá cho căn nhà mới')
with st.container(border=True):
    cot1, cot2, cot3 = st.columns(3)
    dien_tich = cot1.number_input('Diện tích (m²)', min_value=20.0, max_value=300.0, value=100.0, step=5.0)
    so_phong = cot2.number_input('Số phòng ngủ', min_value=1, max_value=10, value=3)
    khoang_cach = cot3.number_input('Khoảng cách tới trung tâm (km)', min_value=0.0, max_value=50.0, value=10.0, step=0.5)

    gia = model.predict([[dien_tich, so_phong, khoang_cach]])[0]
    st.metric('Giá dự đoán', f'{gia:,.1f} triệu VND')
    st.caption(
        f'= {w[0]:.2f} + {w[1]:.2f} × {dien_tich:g} + {w[2]:.2f} × {so_phong} '
        f'+ ({w[3]:.2f}) × {khoang_cach:g}'
    )

with st.expander('Xem dữ liệu'):
    st.dataframe(df.rename(columns={
        'dien_tich': 'Diện tích (m²)', 'so_phong_ngu': 'Số phòng ngủ',
        'khoang_cach': 'Khoảng cách (km)', 'gia': 'Giá (triệu VND)',
    }))
