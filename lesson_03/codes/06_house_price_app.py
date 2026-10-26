# Demo Streamlit: dự đoán giá nhà bằng hồi quy tuyến tính
# Chạy: streamlit run "codes/06_house_price_app.py"
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

FEATURES = ['area', 'bedrooms', 'distance']   # diện tích, số phòng ngủ, khoảng cách
COLUMN_NAMES = {                              # tên tiếng Việt để hiển thị
    'area': 'Diện tích (m²)', 'bedrooms': 'Số phòng ngủ',
    'distance': 'Khoảng cách (km)', 'price': 'Giá (triệu VND)',
}


@st.cache_data
def load_data():
    """Đọc dữ liệu giá nhà từ thư mục data/ của bài."""
    return pd.read_csv(Path(__file__).resolve().parent.parent / 'data' / 'house_prices.csv')


@st.cache_resource
def train(df):
    """Huấn luyện mô hình bằng scikit-learn và bằng công thức, trả về (model, w, số mẫu huấn luyện)."""
    X = df[FEATURES].values   # mỗi hàng là một mẫu
    y = df['price'].values
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

df = load_data()
model, w, n_train = train(df)

# ---- Phần 1: So sánh hệ số ----
st.subheader('Hệ số của mô hình')
st.caption(f'Huấn luyện trên {n_train} căn nhà (80% dữ liệu).')

coef_table = pd.DataFrame({
    'Ý nghĩa': ['Hệ số chệch', 'Diện tích (m²)', 'Số phòng ngủ', 'Khoảng cách (km)'],
    'scikit-learn': np.r_[model.intercept_, model.coef_],
    'Tính thủ công': w,
}, index=['w0', 'w1', 'w2', 'w3'])
st.dataframe(coef_table.style.format({'scikit-learn': '{:.4f}', 'Tính thủ công': '{:.4f}'}))

if np.allclose(coef_table['scikit-learn'], coef_table['Tính thủ công']):
    st.success('Hai cách cho cùng một nghiệm.', icon=':material/check_circle:')

# ---- Phần 2: Dự đoán giá cho căn nhà mới ----
st.subheader('Dự đoán giá cho căn nhà mới')
with st.container(border=True):
    col1, col2, col3 = st.columns(3)
    area = col1.number_input('Diện tích (m²)', min_value=20.0, max_value=300.0, value=100.0, step=5.0)
    bedrooms = col2.number_input('Số phòng ngủ', min_value=1, max_value=10, value=3)
    distance = col3.number_input('Khoảng cách tới trung tâm (km)', min_value=0.0, max_value=50.0, value=10.0, step=0.5)

    price = model.predict([[area, bedrooms, distance]])[0]   # giá dự đoán (triệu VND)
    st.metric('Giá dự đoán', f'{price:,.1f} triệu VND')
    st.caption(
        f'= {w[0]:.2f} + {w[1]:.2f} × {area:g} + {w[2]:.2f} × {bedrooms} '
        f'+ ({w[3]:.2f}) × {distance:g}'
    )

with st.expander('Xem dữ liệu'):
    st.dataframe(df.rename(columns=COLUMN_NAMES))
