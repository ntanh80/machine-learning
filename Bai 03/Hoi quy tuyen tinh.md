# HỒI QUY TUYẾN TÍNH (LINEAR REGRESSION)

> Tài liệu tham khảo: *Giáo trình Machine Learning 2026*, Chương 4 – Hồi quy tuyến tính.

**Hồi quy tuyến tính** là một phương pháp **học có giám sát** dùng cho bài toán **hồi quy**, trong đó mối quan hệ giữa biến đầu vào và biến đầu ra được mô hình hóa bằng một **hàm tuyến tính**. Đây là một trong những mô hình cơ bản và đơn giản nhất trong học máy, đồng thời là nền tảng cho nhiều phương pháp hồi quy phức tạp hơn.

---

## 1. Phát biểu bài toán

Xét bài toán **dự đoán giá nhà**. Mỗi căn nhà được mô tả bởi ba đặc trưng:

| Ký hiệu | Đặc trưng | Đơn vị |
|---|---|---|
| $x_1$ | Diện tích | m² |
| $x_2$ | Số phòng ngủ | phòng |
| $x_3$ | Khoảng cách tới trung tâm | km |

Cho tập dữ liệu gồm 10.000 căn nhà đã biết giá bán. Với một căn nhà mới, ta cần xây dựng hàm dự đoán:

$$
y = f(\mathbf{x}), \qquad \mathbf{x} = [x_1, x_2, x_3]^T
$$

Thực tế cho thấy giá nhà **tăng** khi diện tích lớn hơn, số phòng nhiều hơn, và **giảm** khi xa trung tâm hơn. Do đó, một giả thiết hợp lý là giá nhà được xấp xỉ bởi một **hàm tuyến tính**:

$$
y \approx \hat{y} = f(\mathbf{x}) = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_3
$$

trong đó:
- $w_0$: **hệ số chệch** (bias)
- $\mathbf{w} = [w_1, w_2, w_3]^T$: **vector trọng số** cần ước lượng từ dữ liệu

### Viết gọn bằng vector mở rộng

Thêm phần tử $1$ vào đầu vector đặc trưng:

$$
\bar{\mathbf{x}} = [1, x_1, x_2, x_3]^T, \qquad \bar{\mathbf{w}} = [w_0, w_1, w_2, w_3]^T
$$

Khi đó mô hình viết gọn thành:

$$
\hat{y} = \bar{\mathbf{x}}^T \bar{\mathbf{w}}
$$

> **Vì sao cần hệ số chệch $w_0$?**
> Nếu không có $w_0$, đường hồi quy buộc phải đi qua gốc tọa độ (khi $\mathbf{x} = \mathbf{0}$ thì $\hat{y} = 0$). Điều này thường không hợp lý trong thực tế. Hệ số chệch cho phép đường hồi quy dịch chuyển song song với trục tung, giúp mô tả dữ liệu tốt hơn.

![Vai trò của hệ số chệch w0](images/he_so_chech.png)

*Hình: Mô phỏng 40 căn nhà với giá thực $y \approx 800 + 15x$. Đường **có** $w_0$ (xanh) bám sát dữ liệu. Đường **không có** $w_0$ (đỏ, nét đứt) bị buộc đi qua gốc tọa độ nên phải "dốc" hơn để bù: nó dự đoán thấp với nhà nhỏ và cao với nhà lớn. Mã nguồn vẽ hình: [images/ve_he_so_chech.py](images/ve_he_so_chech.py).*

---

## 2. Xây dựng và tối ưu hàm mất mát

### 2.1. Mô hình tổng quát

Với dữ liệu có $d$ đặc trưng, $\mathbf{x} \in \mathbb{R}^d$:

$$
y = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_d x_d = w_0 + \mathbf{x}^T \mathbf{w}
$$

### 2.2. Sai số dự đoán

Sai số giữa giá trị thực và giá trị dự đoán:

$$
e = y - \hat{y}
$$

Dùng **sai số bình phương**:

$$
\frac{1}{2} e^2 = \frac{1}{2}(y - \hat{y})^2 = \frac{1}{2}(y - \mathbf{x}^T \mathbf{w})^2
$$

**Tại sao dùng bình phương?**

1. Sai số $e$ có thể âm hoặc dương. Nếu cộng trực tiếp, các sai số sẽ **triệt tiêu lẫn nhau**. Bình phương đảm bảo mọi sai số đều không âm.
2. Có thể dùng trị tuyệt đối $|e|$, nhưng hàm này **không khả vi tại $e = 0$**, gây khó khăn cho các thuật toán tối ưu dựa trên đạo hàm như Gradient Descent.
3. Hệ số $\frac{1}{2}$ chỉ để thuận tiện: khi lấy đạo hàm, nó triệt tiêu với số $2$ sinh ra từ bình phương.

> **Lưu ý:** Trong bài này ta dùng **sai số bình phương** vì nó khả vi, lồi và cho **nghiệm dạng đóng** (mục 2.5). Khi lấy trung bình các sai số trên toàn bộ tập dữ liệu, ta thu được các độ đo đánh giá mô hình như MAE, MSE, RMSE (xem mục 6).

### 2.3. Hàm mất mát

Với tập huấn luyện gồm $N$ mẫu $(\mathbf{x}_i, y_i)$, hàm mất mát là **trung bình sai số bình phương**:

$$
\mathcal{L}(\mathbf{w}) = \frac{1}{2N} \sum_{i=1}^{N} (y_i - \mathbf{x}_i^T \mathbf{w})^2
$$

Bài toán huấn luyện trở thành bài toán tối ưu:

$$
\mathbf{w}^* = \arg\min_{\mathbf{w}} \mathcal{L}(\mathbf{w})
$$

> **Vì sao chia cho $N$?**
> Nhân hàm mất mát với một hằng số dương **không làm thay đổi nghiệm tối ưu**. Tuy nhiên, lấy trung bình giúp giá trị hàm mất mát **không phụ thuộc vào kích thước tập dữ liệu**, nhờ đó có thể so sánh chất lượng mô hình giữa các tập dữ liệu khác nhau.

### 2.4. Dạng ma trận

Đặt ma trận dữ liệu $\mathbf{X} \in \mathbb{R}^{N \times d}$, **mỗi hàng là một mẫu** $\mathbf{x}_i^T$, và $\mathbf{y} = [y_1, y_2, \dots, y_N]^T$:

$$
\mathbf{X} =
\begin{bmatrix}
\mathbf{x}_1^T \\ \mathbf{x}_2^T \\ \vdots \\ \mathbf{x}_N^T
\end{bmatrix}
\quad\Rightarrow\quad
\mathcal{L}(\mathbf{w}) = \frac{1}{2N} \left\| \mathbf{y} - \mathbf{X}\mathbf{w} \right\|_2^2
$$

Như vậy hàm mất mát là bình phương **chuẩn $\ell_2$** của vector sai số.

### 2.5. Nghiệm của hồi quy tuyến tính (Normal Equation)

Hàm mất mát khả vi với mọi $\mathbf{w}$, nên nghiệm tối ưu thỏa mãn gradient bằng $0$:

$$
\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}) = \frac{1}{N} \mathbf{X}^T (\mathbf{X}\mathbf{w} - \mathbf{y}) = \mathbf{0}
\quad\Leftrightarrow\quad
\mathbf{X}^T \mathbf{X}\, \mathbf{w} = \mathbf{X}^T \mathbf{y}
$$

**Trường hợp 1:** $\mathbf{X}^T\mathbf{X}$ khả nghịch → nghiệm duy nhất:

$$
\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

**Trường hợp 2:** $\mathbf{X}^T\mathbf{X}$ không khả nghịch → dùng **giả nghịch đảo** (pseudo-inverse) $(\cdot)^\dagger$:

$$
\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{\dagger} \mathbf{X}^T \mathbf{y}
$$

Giả nghịch đảo **luôn tồn tại**, kể cả khi ma trận không vuông hoặc không khả nghịch. Nghiệm này là vector $\mathbf{w}$ có **chuẩn $\ell_2$ nhỏ nhất** trong số các nghiệm làm cực tiểu sai số bình phương. Khi ma trận vuông và khả nghịch, giả nghịch đảo trùng với nghịch đảo thông thường.

Vì vậy, **nghiệm tổng quát** là:

$$
\boxed{\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{\dagger} \mathbf{X}^T \mathbf{y}}
$$

Trong Python, giả nghịch đảo được tính bằng `numpy.linalg.pinv()`.

> **Ghi chú:** Giáo trình viết $\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_N]$ (mỗi **cột** là một mẫu), khi đó công thức có dạng $\mathbf{w} = (\mathbf{X}\mathbf{X}^T)^\dagger \mathbf{X}\mathbf{y}$. Tài liệu này dùng quy ước **mỗi hàng là một mẫu** để khớp với NumPy và scikit-learn. Hai cách viết là tương đương.

### 2.6. Giải bằng Gradient Descent

Ngoài nghiệm giải tích, có thể tìm $\mathbf{w}$ bằng **Gradient Descent** (xem Chương 3), lặp lại cập nhật:

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \, \nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}) = \mathbf{w} - \frac{\eta}{N} \mathbf{X}^T (\mathbf{X}\mathbf{w} - \mathbf{y})
$$

với $\eta$ là tốc độ học (learning rate). Cách này phù hợp khi dữ liệu lớn, vì việc tính nghịch đảo ma trận $d \times d$ trở nên tốn kém.

---

## 3. Ví dụ trên Python: Dự đoán cân nặng theo chiều cao

### 3.1. Bài toán

Cho dữ liệu chiều cao và cân nặng của 15 người. Hai người có chiều cao **155 cm** và **160 cm** được tách ra làm **tập kiểm tra**, 13 người còn lại là **tập huấn luyện**.

| Chiều cao (cm) | Cân nặng (kg) | Chiều cao (cm) | Cân nặng (kg) | Chiều cao (cm) | Cân nặng (kg) |
|---|---|---|---|---|---|
| 147 | 49 | 158 | 54 | 173 | 63 |
| 150 | 50 | **160** | **56** | 175 | 64 |
| 153 | 51 | 163 | 58 | 178 | 66 |
| **155** | **52** | 165 | 59 | 180 | 67 |
| 168 | 60 | 170 | 62 | 183 | 68 |

*(Các dòng in đậm là tập kiểm tra.)*

> **Lưu ý:** Bảng 4.1 trong giáo trình ghi cân nặng của người cao 170 cm là **72 kg**, nhưng code minh họa trong giáo trình dùng **62 kg**. Giá trị 62 kg phù hợp với xu hướng chung của dữ liệu và cho đúng kết quả trong sách, nên tài liệu này dùng 62 kg.

Giữa chiều cao và cân nặng có mối quan hệ **thuận chiều** (người cao hơn thường nặng hơn), nên dùng mô hình:

$$
\text{(cân nặng)} = w_1 \cdot \text{(chiều cao)} + w_0
$$

### 3.2. Khai báo và hiển thị dữ liệu

```python
import numpy as np
import matplotlib.pyplot as plt

# Du lieu chieu cao (cm) - moi hang la mot mau du lieu
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170,
               173, 175, 178, 180, 183]]).T

# Du lieu can nang tuong ung (kg) - bien muc tieu
y = np.array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

plt.plot(X, y, 'ro')
plt.xlabel('Chieu cao (cm)')
plt.ylabel('Can nang (kg)')
plt.show()
```

Các điểm dữ liệu nằm **gần như trên một đường thẳng**, vì vậy hồi quy tuyến tính có khả năng cho kết quả tốt.

### 3.3. Nghiệm theo công thức

```python
# Tao ma tran Xbar (them cot 1 vao truoc ma tran X)
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

# Tinh w = (Xbar^T Xbar)^+ Xbar^T y
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)

w_0, w_1 = w[0], w[1]
print('w_0 =', w_0)
print('w_1 =', w_1)

# Ve duong hoi quy
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1 * x0
plt.plot(X, y, 'ro')
plt.plot(x0, y0)
plt.xlabel('Chieu cao (cm)')
plt.ylabel('Can nang (kg)')
plt.show()

# Du doan tren tap kiem tra
y1 = w_1 * 155 + w_0
y2 = w_1 * 160 + w_0
print('Chieu cao 155 cm, can nang thuc te 52 kg, du doan %.2f kg' % y1)
print('Chieu cao 160 cm, can nang thuc te 56 kg, du doan %.2f kg' % y2)
```

**Kết quả:**

```
w_0 = -33.7354
w_1 = 0.5592
Chieu cao 155 cm, can nang thuc te 52 kg, du doan 52.94 kg
Chieu cao 160 cm, can nang thuc te 56 kg, du doan 55.74 kg
```

Kết quả dự đoán khá sát với giá trị thực tế.

### 3.4. Sử dụng thư viện scikit-learn

```python
from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(X, y)   # moi mau du lieu la mot hang

print("Ket qua tu scikit-learn:")
print("w1 =", model.coef_[0])     # he so goc
print("w0 =", model.intercept_)   # he so chech

print("Ket qua tinh toan thu cong:")
print("w1 =", w[1])
print("w0 =", w[0])
```

Hai kết quả **trùng nhau**: $w_1 \approx 0.5592$, $w_0 \approx -33.7354$.

---

## 4. Thảo luận

### 4.1. Hồi quy đa thức

Mô hình $y \approx \mathbf{x}^T\mathbf{w} + b$ tuyến tính theo cả $\mathbf{w}$ và $\mathbf{x}$. Thực ra, hồi quy tuyến tính **chỉ cần tuyến tính theo $\mathbf{w}$**. Ví dụ:

$$
y \approx w_1 x_1 + w_2 x_2 + w_3 x_1^2 + w_4 \sin(x_2) + w_5 x_1 x_2 + w_0
$$

không tuyến tính theo $\mathbf{x}$ nhưng vẫn tuyến tính theo $\mathbf{w}$. Ta chỉ cần tạo vector đặc trưng mới:

$$
\tilde{\mathbf{x}} = [x_1,\ x_2,\ x_1^2,\ \sin(x_2),\ x_1 x_2]^T
$$

rồi áp dụng hồi quy tuyến tính như bình thường.

Trong thực tế, **hồi quy đa thức** (polynomial regression) hay được dùng hơn, với vector đặc trưng dạng $[1, x, x^2, x^3, \dots]^T$.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Hoi quy da thuc bac 3
poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
poly_model.fit(X, y)
```

### 4.2. Hạn chế của hồi quy tuyến tính

#### a) Nhạy cảm với nhiễu và giá trị ngoại lai (outliers)

Giả sử dữ liệu có thêm một điểm bất thường **(150 cm, 90 kg)**. Vì hồi quy tuyến tính tối thiểu hóa **sai số bình phương**, các sai số lớn bị **khuếch đại**, nên chỉ một vài điểm ngoại lai cũng có thể làm lệch đáng kể đường hồi quy.

**Cách khắc phục:**
- Loại bỏ điểm nhiễu trước khi huấn luyện. Cách này khó áp dụng khi dữ liệu lớn hoặc nhiều chiều.
- Dùng **hàm mất mát kháng nhiễu** (robust loss):
  - **Huber loss** (hồi quy Huber): hoạt động như **L2 loss** khi sai số nhỏ và như **L1 loss** khi sai số lớn.
  - **Least Absolute Deviations** (L1 regression): tối thiểu hóa tổng sai số tuyệt đối.
  - **RANSAC**: chọn ngẫu nhiên các tập con dữ liệu để tìm mô hình tốt nhất và loại bỏ điểm ngoại lai.
  - **Theil–Sen estimator**: ước lượng hệ số dựa trên **trung vị** của nhiều ước lượng.

```python
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor

huber = HuberRegressor().fit(X, y)
ransac = RANSACRegressor().fit(X, y)
theil = TheilSenRegressor().fit(X, y)
```

#### b) Khả năng biểu diễn hạn chế

Mô hình giả định quan hệ **tuyến tính**, trong khi nhiều bài toán thực tế có quan hệ **phi tuyến**. Có thể thêm đặc trưng phi tuyến ($x_1^2$, $\sin(x_2)$, $x_1x_2$, …), nhưng khi số chiều tăng thì số đặc trưng tăng rất nhanh, dẫn đến **bùng nổ số chiều** (curse of dimensionality).

#### c) Các hạn chế khác

- **Đa cộng tuyến** (multicollinearity): khi các biến đầu vào tương quan cao, ước lượng hệ số trở nên **không ổn định**.
- **Giả định tuyến tính mạnh**: không phải lúc nào cũng đúng với dữ liệu thực.
- **Phương sai thay đổi** (heteroscedasticity): khi phương sai của sai số thay đổi theo đầu vào, hiệu quả mô hình giảm.

→ Với bài toán phức tạp, thường dùng **cây quyết định, random forest, mạng nơ-ron** hoặc **mô hình kernel**.

---

## 5. Hồi quy Ridge và Lasso

Để khắc phục hiện tượng **quá khớp** (overfitting) và **đa cộng tuyến**, ta dùng kỹ thuật **điều chuẩn** (regularization): bổ sung một **thành phần phạt** vào hàm mất mát để hạn chế độ lớn của các tham số.

Hàm mất mát thông thường:

$$
\mathcal{L}(\mathbf{w}) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

### 5.1. Hồi quy Ridge (điều chuẩn L2)

$$
\mathcal{L}_{\text{Ridge}}(\mathbf{w}) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{d} w_j^2
$$

- $\lambda \ge 0$: **hệ số điều chuẩn**. $\lambda$ càng lớn thì các hệ số càng bị "co" về gần $0$.
- Thành phần phạt làm giảm độ lớn các hệ số, giúp mô hình **ổn định hơn** và **giảm overfitting**.

Nghiệm dạng ma trận:

$$
\mathbf{w} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}
$$

với $\mathbf{I}$ là ma trận đơn vị. Khi $\lambda > 0$, ma trận $\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I}$ **luôn khả nghịch**, vì vậy Ridge còn giải quyết được trường hợp $\mathbf{X}^T\mathbf{X}$ suy biến.

### 5.2. Hồi quy Lasso (điều chuẩn L1)

**Lasso** = *Least Absolute Shrinkage and Selection Operator*.

$$
\mathcal{L}_{\text{Lasso}}(\mathbf{w}) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{d} |w_j|
$$

Lasso có thể đưa **một số hệ số $w_j$ về đúng bằng $0$**, tương đương với việc **loại bỏ các đặc trưng không quan trọng**. Vì vậy Lasso thường được dùng để **lựa chọn đặc trưng** (feature selection).

### 5.3. So sánh Ridge và Lasso

| Tiêu chí | Ridge | Lasso |
|---|---|---|
| Thành phần phạt | $\lambda \sum w_j^2$ (chuẩn L2) | $\lambda \sum \lvert w_j \rvert$ (chuẩn L1) |
| Ảnh hưởng lên hệ số | Giảm độ lớn, **hiếm khi bằng 0** | Có thể đưa hệ số **về đúng 0** |
| Lựa chọn đặc trưng | Không | Có |
| Nghiệm giải tích | Có | Không (giải bằng phương pháp lặp) |
| Phù hợp khi | **Tất cả** đặc trưng đều có ảnh hưởng | Chỉ **một số ít** đặc trưng thực sự quan trọng |

### 5.4. Ví dụ Python

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso

X = np.array([[147, 150, 153, 158, 163, 165, 168, 170,
               173, 175, 178, 180, 183]]).T
y = np.array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# Ridge Regression (alpha chinh la lambda)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X, y)
print("Ridge: w1 =", ridge_model.coef_[0], ", w0 =", ridge_model.intercept_)

# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X, y)
print("Lasso: w1 =", lasso_model.coef_[0], ", w0 =", lasso_model.intercept_)

# Du doan can nang cho nguoi cao 160 cm
h = np.array([[160]])
print("Du doan bang Ridge:", ridge_model.predict(h))
print("Du doan bang Lasso:", lasso_model.predict(h))
```

**Kết quả:**

```
Ridge: w1 = 0.5589 , w0 = -33.6786
Lasso: w1 = 0.5584 , w0 = -33.6033
Du doan bang Ridge: [55.74]
Du doan bang Lasso: [55.74]
```

Với dữ liệu chỉ có 1 đặc trưng và không bị nhiễu, Ridge và Lasso cho kết quả rất gần với hồi quy tuyến tính thông thường. Sự khác biệt thể hiện rõ hơn khi dữ liệu có **nhiều đặc trưng** hoặc các đặc trưng **tương quan mạnh**.

---

## 6. Đánh giá mô hình hồi quy

| Độ đo | Công thức | Ý nghĩa |
|---|---|---|
| MAE | $\frac{1}{N}\sum \lvert y_i - \hat{y}_i \rvert$ | Sai số tuyệt đối trung bình |
| MSE | $\frac{1}{N}\sum (y_i - \hat{y}_i)^2$ | Sai số bình phương trung bình |
| RMSE | $\sqrt{\text{MSE}}$ | Cùng đơn vị với $y$, dễ diễn giải |
| $R^2$ | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Tỉ lệ phương sai được giải thích (càng gần 1 càng tốt) |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_test = np.array([[155], [160]])
y_test = np.array([52, 56])
y_pred = model.predict(X_test)

print("MAE  =", mean_absolute_error(y_test, y_pred))
print("MSE  =", mean_squared_error(y_test, y_pred))
print("RMSE =", np.sqrt(mean_squared_error(y_test, y_pred)))
```

---

## 7. Kết luận

- **Hồi quy tuyến tính** mô tả quan hệ giữa biến phụ thuộc và một hoặc nhiều biến độc lập bằng một **hàm tuyến tính**.
- Tham số được tìm bằng cách **tối thiểu hóa sai số bình phương trung bình (MSE)**, thông qua:
  - **Nghiệm giải tích** (Normal Equation): $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^\dagger\mathbf{X}^T\mathbf{y}$
  - **Gradient Descent**: phù hợp khi dữ liệu lớn.
- **Hạn chế**: nhạy cảm với ngoại lai, chỉ biểu diễn được quan hệ tuyến tính, gặp vấn đề đa cộng tuyến.
- **Điều chuẩn** giúp giảm overfitting: **Ridge** (L2) làm co hệ số; **Lasso** (L1) có thể loại bỏ đặc trưng.
- Hồi quy tuyến tính đơn giản nhưng là **nền tảng** cho nhiều mô hình phức tạp hơn trong học sâu và thống kê hiện đại.

---

## 8. Câu hỏi ôn tập

### Câu hỏi lý thuyết

1. Trình bày khái niệm mô hình hồi quy tuyến tính. Mục tiêu của bài toán hồi quy tuyến tính là gì?
2. Viết biểu thức toán học của mô hình hồi quy tuyến tính với $d$ biến đầu vào.
3. Trình bày hàm mất mát thường dùng trong hồi quy tuyến tính. Tại sao MSE được sử dụng phổ biến?
4. Giải thích phương pháp tìm nghiệm giải tích (Normal Equation) trong hồi quy tuyến tính.
5. Trình bày nguyên lý của Gradient Descent trong việc tối ưu tham số mô hình hồi quy.
6. Giải thích hiện tượng quá khớp (overfitting) trong hồi quy tuyến tính. Tại sao hiện tượng này thường xảy ra khi số chiều dữ liệu lớn?
7. So sánh Ridge Regression và Lasso Regression. Hai phương pháp khác nhau như thế nào trong cách điều chuẩn tham số?

### Bài tập thực hành

8. Cho tập dữ liệu: $(1, 2), (2, 3), (3, 5), (4, 4), (5, 6)$.
   - a. Xây dựng mô hình hồi quy tuyến tính $y = w_0 + w_1 x$.
   - b. Tính $w_0$, $w_1$ bằng phương pháp bình phương tối thiểu. *(Đáp số kiểm tra: $w_1 = 0.9$, $w_0 = 1.3$.)*

9. Cho hàm mất mát của Ridge Regression:
   $J(\mathbf{w}) = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{d} w_j^2$
   - a. Giải thích ý nghĩa của tham số $\lambda$.
   - b. Điều gì xảy ra khi $\lambda$ rất lớn hoặc rất nhỏ?
   - c. Ridge Regression giúp giảm overfitting như thế nào?

10. Cho hàm mất mát của Lasso Regression:
    $J(\mathbf{w}) = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{d} |w_j|$
    - a. So sánh thành phần điều chuẩn của Lasso với Ridge.
    - b. Giải thích tại sao Lasso có khả năng lựa chọn đặc trưng.
    - c. Trong trường hợp nào nên dùng Lasso thay vì Ridge?
