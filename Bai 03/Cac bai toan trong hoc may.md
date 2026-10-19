# GIỚI THIỆU: CÁC LỚP BÀI TOÁN TRONG HỌC MÁY

## 1. Học máy là gì?

**Học máy (Machine Learning)** là một nhánh của Trí tuệ nhân tạo, nghiên cứu các phương pháp giúp máy tính **tự học từ dữ liệu** để cải thiện hiệu năng trên một nhiệm vụ mà không cần lập trình tường minh từng quy tắc.

Định nghĩa của Tom Mitchell (1997):

> Một chương trình máy tính được gọi là *học* từ kinh nghiệm **E** đối với nhiệm vụ **T** và độ đo hiệu năng **P**, nếu hiệu năng của nó trên **T**, đo bằng **P**, được cải thiện nhờ kinh nghiệm **E**.

Ví dụ: Lọc thư rác
- **T**: phân loại email là *rác* hay *không rác*
- **E**: tập email đã được gán nhãn
- **P**: tỉ lệ email được phân loại đúng

---

## 2. Các lớp bài toán trong học máy

Dựa vào **cách thức học** và **dạng dữ liệu huấn luyện**, học máy thường được chia thành các lớp chính sau:

| Lớp bài toán | Dữ liệu huấn luyện | Mục tiêu | Ví dụ |
|---|---|---|---|
| Học có giám sát (Supervised) | Có nhãn `(x, y)` | Học ánh xạ `x → y` | Dự đoán giá nhà, nhận dạng chữ viết |
| Học không giám sát (Unsupervised) | Không nhãn `x` | Tìm cấu trúc ẩn trong dữ liệu | Phân cụm khách hàng |
| Học bán giám sát (Semi-supervised) | Ít dữ liệu có nhãn + nhiều dữ liệu không nhãn | Tận dụng dữ liệu không nhãn để học tốt hơn | Phân loại ảnh y tế |
| Học tự giám sát (Self-supervised) | Không nhãn, nhãn được tự sinh từ dữ liệu | Học biểu diễn dữ liệu | Mô hình ngôn ngữ (BERT, GPT) |
| Học tăng cường (Reinforcement) | Tương tác với môi trường, nhận phần thưởng | Học chính sách hành động tối ưu | Chơi cờ, robot, xe tự lái |

### 2.1. Học có giám sát (Supervised Learning)

Dữ liệu huấn luyện gồm các cặp `(xᵢ, yᵢ)`, trong đó `xᵢ` là đặc trưng đầu vào và `yᵢ` là nhãn (đầu ra mong muốn). Mô hình học một hàm `f` sao cho `f(x) ≈ y`.

Hai bài toán con chính:
- **Phân lớp (Classification)**: `y` là giá trị rời rạc (nhãn lớp).
- **Hồi quy (Regression)**: `y` là giá trị liên tục (số thực).

### 2.2. Học không giám sát (Unsupervised Learning)

Chỉ có dữ liệu đầu vào `xᵢ`, không có nhãn. Mô hình tự khám phá cấu trúc, quy luật trong dữ liệu.

Các bài toán con chính:
- **Phân cụm (Clustering)**
- **Giảm chiều dữ liệu (Dimensionality Reduction)**
- **Khai phá luật kết hợp (Association Rule Mining)**
- **Phát hiện bất thường (Anomaly Detection)**

### 2.3. Học bán giám sát (Semi-supervised Learning)

Kết hợp một lượng nhỏ dữ liệu có nhãn với lượng lớn dữ liệu không nhãn. Phù hợp khi việc gán nhãn tốn kém (cần chuyên gia, mất thời gian).

Kỹ thuật tiêu biểu: *Self-training*, *Co-training*, *Label Propagation*.

### 2.4. Học tự giám sát (Self-supervised Learning)

Mô hình tự tạo nhãn từ chính dữ liệu, ví dụ: che một từ trong câu và yêu cầu mô hình đoán lại từ đó. Đây là nền tảng của các mô hình ngôn ngữ lớn và mô hình nền tảng (foundation models).

### 2.5. Học tăng cường (Reinforcement Learning)

Một **tác tử (agent)** tương tác với **môi trường (environment)**: tại mỗi **trạng thái (state)**, tác tử chọn một **hành động (action)** và nhận về **phần thưởng (reward)**. Mục tiêu là học **chính sách (policy)** tối đa hóa tổng phần thưởng tích lũy.

Thuật toán tiêu biểu: *Q-Learning*, *SARSA*, *Deep Q-Network (DQN)*, *Policy Gradient*, *PPO*.

---

## 3. Các bài toán cơ bản trong học máy

### 3.1. Bài toán phân lớp (Classification)

- **Mục tiêu**: Gán mỗi đối tượng vào một trong các lớp đã biết trước.
- **Đầu ra**: Nhãn rời rạc.
- **Phân loại**:
  - Phân lớp nhị phân (2 lớp): email rác / không rác.
  - Phân lớp đa lớp (nhiều lớp): nhận dạng chữ số 0–9.
  - Phân lớp đa nhãn: một bài báo có thể thuộc nhiều chủ đề.
- **Thuật toán**: k-NN, Naive Bayes, Logistic Regression, Cây quyết định, Random Forest, SVM, Mạng nơ-ron.
- **Độ đo đánh giá**: Accuracy, Precision, Recall, F1-score, Ma trận nhầm lẫn, ROC-AUC.

### 3.2. Bài toán hồi quy (Regression)

- **Mục tiêu**: Dự đoán một giá trị số liên tục.
- **Ví dụ**: Dự đoán giá nhà, nhiệt độ, doanh thu, giá cổ phiếu.
- **Thuật toán**: Hồi quy tuyến tính, Hồi quy đa thức, Ridge, Lasso, SVR, Cây hồi quy, Gradient Boosting.
- **Độ đo đánh giá**: MAE, MSE, RMSE, R².

### 3.3. Bài toán phân cụm (Clustering)

- **Mục tiêu**: Chia tập dữ liệu thành các nhóm sao cho các phần tử trong cùng nhóm giống nhau, khác nhóm thì khác nhau.
- **Ví dụ**: Phân khúc khách hàng, gom nhóm văn bản, phân đoạn ảnh.
- **Thuật toán**: K-Means, Phân cụm phân cấp (Hierarchical), DBSCAN, Gaussian Mixture Model (GMM).
- **Độ đo đánh giá**: Silhouette score, Davies–Bouldin index, Inertia (SSE).

### 3.4. Bài toán giảm chiều dữ liệu (Dimensionality Reduction)

- **Mục tiêu**: Biểu diễn dữ liệu nhiều chiều trong không gian ít chiều hơn mà vẫn giữ được thông tin quan trọng.
- **Ứng dụng**: Trực quan hóa dữ liệu, giảm nhiễu, tăng tốc huấn luyện, tránh "lời nguyền số chiều".
- **Thuật toán**: PCA, LDA, t-SNE, UMAP, Autoencoder.

### 3.5. Bài toán khai phá luật kết hợp (Association Rule Mining)

- **Mục tiêu**: Tìm các mối quan hệ dạng `A ⇒ B` giữa các thuộc tính/mặt hàng.
- **Ví dụ**: Phân tích giỏ hàng — "Khách mua bánh mì thường mua thêm sữa".
- **Thuật toán**: Apriori, FP-Growth.
- **Độ đo**: Support, Confidence, Lift.

### 3.6. Bài toán phát hiện bất thường (Anomaly Detection)

- **Mục tiêu**: Phát hiện các điểm dữ liệu khác biệt đáng kể so với phần lớn dữ liệu.
- **Ví dụ**: Phát hiện gian lận thẻ tín dụng, xâm nhập mạng, lỗi thiết bị.
- **Thuật toán**: Isolation Forest, One-Class SVM, Local Outlier Factor (LOF).

### 3.7. Bài toán xếp hạng và gợi ý (Ranking & Recommendation)

- **Mục tiêu**: Sắp xếp hoặc gợi ý các đối tượng phù hợp nhất với người dùng.
- **Ví dụ**: Gợi ý phim (Netflix), sản phẩm (Shopee), xếp hạng kết quả tìm kiếm.
- **Phương pháp**: Lọc cộng tác (Collaborative Filtering), Lọc theo nội dung (Content-based), Phân rã ma trận (Matrix Factorization).

---

## 4. Quy trình chung giải một bài toán học máy

1. **Xác định bài toán**: Thuộc lớp nào? Phân lớp, hồi quy hay phân cụm?
2. **Thu thập dữ liệu**.
3. **Tiền xử lý dữ liệu**: làm sạch, xử lý giá trị thiếu, chuẩn hóa, mã hóa biến phân loại.
4. **Trích chọn / xây dựng đặc trưng** (Feature engineering).
5. **Chia dữ liệu**: tập huấn luyện (train), kiểm định (validation), kiểm tra (test).
6. **Lựa chọn và huấn luyện mô hình**.
7. **Đánh giá mô hình** bằng các độ đo phù hợp.
8. **Tinh chỉnh siêu tham số** (Grid Search, Random Search, Cross-validation).
9. **Triển khai và giám sát mô hình**.

---

## 5. Gợi ý lựa chọn bài toán

```
Dữ liệu có nhãn?
├── Có
│   ├── Nhãn rời rạc    → PHÂN LỚP
│   └── Nhãn liên tục   → HỒI QUY
└── Không
    ├── Muốn gom nhóm         → PHÂN CỤM
    ├── Muốn giảm số chiều    → GIẢM CHIỀU
    ├── Muốn tìm quy luật     → LUẬT KẾT HỢP
    └── Muốn tìm điểm lạ      → PHÁT HIỆN BẤT THƯỜNG

Học qua tương tác + phần thưởng → HỌC TĂNG CƯỜNG
```

---

## 6. Tóm tắt

- Học máy có các lớp chính: **có giám sát**, **không giám sát**, **bán giám sát**, **tự giám sát** và **tăng cường**.
- Các bài toán cơ bản: **phân lớp**, **hồi quy**, **phân cụm**, **giảm chiều**, **luật kết hợp**, **phát hiện bất thường**, **gợi ý**.
- Việc xác định đúng lớp bài toán là bước đầu tiên và quan trọng nhất để chọn thuật toán và độ đo đánh giá phù hợp.
