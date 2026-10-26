# Chỉ dẫn biên soạn bài giảng – Học máy K22 KHMT

Tài liệu này áp dụng khi viết hoặc sửa bất kỳ nội dung nào của một bài (`lesson_01` … `lesson_09`): file lý thuyết `.md`, notebook, script vẽ hình, dữ liệu. Gồm bốn phần:

1. Nguyên tắc nội dung – viết *cái gì*.
2. Cấu trúc thư mục – để *ở đâu*.
3. Quy ước đặt tên và ngôn ngữ – viết *thế nào*.
4. Tra cứu giáo trình – lấy kiến thức *từ đâu*.

---

## 1. Nguyên tắc nội dung: mục tiêu là chuẩn, giáo trình để tham khảo

1. **`1_objectives.md` của bài là chuẩn.** Cấu trúc, phạm vi, thứ tự mục và mức độ (Hiểu/Vận dụng) của nội dung bám theo các mục CĐR (ví dụ 3.1.1, 3.1.2, …) trong file này. Không thêm, bớt hay đổi thứ tự mục.
2. **Giáo trình chỉ để tham khảo**: `textbook/Machine_Learning_Textbook_2026.pdf` (275 trang). Với mỗi mục CĐR, tra phần tương ứng trong giáo trình (mục 4) để lấy kiến thức, ví dụ, công thức; ưu tiên dùng thuật ngữ, ký hiệu giống giáo trình cho thống nhất.
3. **Khi giáo trình không khớp mục tiêu** (chương không trùng một-một với bài, hoặc thiếu mục CĐR):
   - Mục tiêu có, giáo trình không có → vẫn viết đủ, dùng nguồn chuẩn khác (sách, tài liệu scikit-learn, …). Không cắt mục tiêu cho vừa giáo trình.
   - Giáo trình có, mục tiêu không yêu cầu → không đưa vào.
4. **Khi giáo trình sai hoặc mâu thuẫn với mục tiêu** → báo lại cho tác giả, không tự sửa ngầm.
5. **Tham chiếu trong file `.md` chỉ ghi đến chương, bài**: tới giáo trình ghi `Chương 9` (hoặc `Lời nói đầu`), tới bài giảng khác ghi `Bài 3`. Không ghi số mục (`mục 9.2.2`, `Bài 3, mục 3.3.2`) hay số trang (`tr. 158`). Bảng tra cứu ở mục 4 có số trang chỉ để người biên soạn tìm nhanh, không chép vào bài giảng.

---

## 2. Cấu trúc thư mục mỗi bài

```text
lesson_XX/
├── 1_objectives.md     # mục tiêu, chuẩn đầu ra, phân bổ nội dung (chuẩn của bài)
├── 2_<topic>.md        # nội dung lý thuyết chính, đề mục theo các mục CĐR
├── codes/              # notebook demo: 01_<name>.ipynb, 02_<name>.ipynb, …
├── images/             # hình minh họa + script plot_<name>.py sinh hình
└── data/               # dữ liệu
```

### 2.1. File lý thuyết `2_<topic>.md`

- Đề mục theo đúng các mục CĐR trong `1_objectives.md`.
- Ngay sau mỗi đoạn code minh họa, đặt link tới notebook tương ứng (`codes/...`).

### 2.2. Code demo trong `codes/`

- **Viết dạng notebook `.ipynb`**, không viết `.py`. Ngoại lệ duy nhất: ứng dụng phải chạy bằng lệnh riêng (ví dụ app Streamlit chạy bằng `streamlit run`) thì để `.py`.
- Đánh số thứ tự theo trình tự chuyên môn trong bài: `01_<name>.ipynb`, `02_<name>.ipynb`, …
- Xen kẽ ô Markdown giải thích bằng tiếng Việt với ô code.
- Đọc dữ liệu từ `../data/` bằng `Path('..') / 'data'` (notebook chạy với thư mục hiện hành là `codes/`).

### 2.3. Hình trong `images/`

- Mỗi hình có script sinh hình đi kèm, tên bắt đầu bằng `plot_` và trùng tên với hình: `plot_overfitting.py` → `overfitting.png`.

---

## 3. Quy ước đặt tên và ngôn ngữ

### 3.1. Tên file và dữ liệu – tiếng Anh

- Tên mọi file (`.md`, và trong `codes/`, `images/`, `data/`) bằng **tiếng Anh, theo chủ đề**, `snake_case`. Ví dụ: `2_supervised_learning_linear_regression.md`, `2_gradient_descent.md`, `05_house_price_regression.ipynb`, `plot_overfitting.py`, `house_prices.csv`.
- Tên cột dữ liệu tự tạo bằng tiếng Anh (`area`, `bedrooms`, `distance`, `price`). Dữ liệu tải về (ví dụ Kaggle) giữ nguyên tên gốc.

### 3.2. Tên trong code – tiếng Anh, dễ hiểu, đơn giản

- Biến, hàm, lớp, tham số dùng từ thông dụng, ngắn gọn, đọc tên là biết ý nghĩa: `price`, `area`, `loss`, `learning_rate`, `compute_gradient`.
- Theo quy ước Python: `snake_case` cho biến/hàm, `PascalCase` cho lớp.
- Tránh:
  - viết tắt khó đoán (`lr_v2`, `tmp_arr`) và tên quá dài;
  - tên tiếng Việt không dấu (`gia_nha`, `toc_do_hoc`).
- Ký hiệu toán học theo giáo trình được giữ ngắn: `X`, `y`, `w`, `b`, `theta`, `eta`.

### 3.3. Nội dung hiển thị và comment – tiếng Việt có dấu

- Nội dung bài giảng (tiêu đề, văn bản trong `.md`, ô Markdown của notebook) viết tiếng Việt có dấu.
- **Comment trong code** viết tiếng Việt có dấu, không viết không dấu (sai: `# Du lieu chieu cao`).
- Mỗi biến/hàm quan trọng có comment giải thích ý nghĩa, nêu ký hiệu tương ứng trong bài giảng nếu có:

  ```python
  learning_rate = 0.01  # tốc độ học η
  ```

- Chuỗi hiển thị (`print`, nhãn trục, tiêu đề hình) viết tiếng Việt có dấu.

---

## 4. Tra cứu giáo trình theo mục CĐR

Bảng này chỉ dùng khi biên soạn; trong file `.md` của bài chỉ ghi chương (mục 1, nguyên tắc 5). Số trang là số trang trong file PDF (không phải số in trên trang). "—" nghĩa là giáo trình không có, cần nguồn khác (xem mục 1, nguyên tắc 3).

| Bài | Mục CĐR | Tham khảo trong giáo trình | Trang PDF |
| --- | --- | --- | --- |
| 01 | 1.1.1–1.1.3 Khái niệm, phương thức học, ví dụ ứng dụng | Ch.1 Khái niệm học máy; Các bài toán cơ bản; Phân nhóm thuật toán (có học tăng cường) | 11–28 |
| 01 | 1.1.4–1.1.5 Học máy vs Học sâu; nền tảng NLP, CV | Nhắc qua ở Lời nói đầu, Ch.13 | 9, 216 |
| 01 | 1.2 Anaconda, NumPy/Pandas/Matplotlib/Scikit-learn, Jupyter | — | |
| 02 | 2.1.1–2.1.2 Phát biểu dự án, các bước | Ch.1 Quy trình thực hiện một dự án học máy | 36–39 |
| 02 | 2.1.3–2.1.4 Làm sạch dữ liệu, chọn/thiết kế đặc trưng | Ch.1 Dữ liệu; Ch.2 Kỹ thuật xây dựng đặc trưng, chuẩn hóa | 13; 40–60 |
| 02 | 2.2 Tham số, tinh chỉnh, overfitting/underfitting | Ch.1 Tham số mô hình; Thách thức chính; Thiên lệch – phương sai | 28–35 |
| 02 | 2.3 Dự án giá nhà | Ví dụ giá nhà ở Ch.1, Ch.2, Ch.4 | 37–38, 46, 76 |
| 03 | 3.1 Học có giám sát, hồi quy tuyến tính, MSE | Ch.1 Các bài toán cơ bản; Ch.4 Phát biểu bài toán, hàm mất mát, ví dụ Python | 14–20; 75–84 |
| 03 | 3.2 Quá khớp, điều chuẩn L1/L2, Ridge, Lasso | Ch.4 Thảo luận; Hồi quy Ridge và Lasso | 83–89 |
| 03 | 3.2.4 Elastic Net | — | |
| 03 | 3.3 Thực hành giá nhà, đánh giá mô hình | Ch.4 Ví dụ trên Python | 80–83 |
| 04 | 4.1.1–4.1.4 Phân loại, sigmoid, hồi quy Logistic, MLE + GD | Ch.9 Hồi quy logistic | 154–169 |
| 04 | 4.1.5–4.1.6 Ma trận nhầm lẫn, Accuracy/Precision/Recall/F1 | Chỉ có trong phần thực hành Ch.8, Ch.14 | 148, 255 |
| 04 | 4.1.6 ROC-AUC | — | |
| 04 | 4.2 Perceptron, siêu phẳng, cập nhật trọng số PLA | Ch.7 Mô hình nơ-ron Perceptron | 120–131 |
| 04 | 4.3 Dự đoán ung thư phổi (Kaggle) | — (tham khảo phần thực hành Ch.9) | 159–166 |
| 05 | 5.1 Tối ưu, đạo hàm riêng, cập nhật trọng số, learning rate | Ch.3 GD cho hàm một biến, nhiều biến | 61–70 |
| 05 | 5.2 Batch, Stochastic, Mini-batch | Ch.3 Các biến thể của Gradient Descent | 70–74 |
| 05 | 5.3 Tối ưu MSE bằng GD | Ch.3; Ch.4 Xây dựng và tối ưu hàm mất mát | 61–74; 76–80 |
| 06 | 6.1.1–6.1.2 Cấu trúc cây, Entropy, Information Gain | Ch.8 Khái niệm; Độ vẩn đục và độ tinh khiết; Ví dụ | 132–145 |
| 06 | 6.1.3 ID3, C4.5, CART | Ch.8 Các thuật toán xây dựng cây | 138–142 |
| 06 | 6.1.4 Overfitting và cắt tỉa | Ch.8 (nhắc đến pruning) | 141–142 |
| 06 | 6.2 Thực hành | Ch.8 Thực hành bằng Python | 145–151 |
| 07 | 7.1 Siêu phẳng, lề, bài toán tối ưu, kernel | Ch.10 Mô hình máy vector hỗ trợ (kernel chỉ nhắc qua) | 170–181 |
| 07 | 7.2 Thực hành | Ch.10 Thực hành phân loại với SVM | 178–180 |
| 08 | 8.1.1–8.1.3 Học không giám sát, phân cụm, K-means, chọn K | Ch.1 Phân nhóm thuật toán; Ch.11 K-means (Elbow, Silhouette) | 20–28; 182–193 |
| 08 | 8.1.4 Phân cụm phân cấp, dendrogram | Ch.12 Phân cụm phân cấp | 199–215 |
| 08 | 8.2 K-means trên Iris | Ch.11 Thực hành với Python (Iris có ở Ch.5, Ch.12) | 189–198 |
| 09 | 9.1 ANN, hàm kích hoạt, lan truyền xuôi/ngược | Ch.13 Mạng Nơ-ron nhân tạo | 216–247 |
| 09 | 9.2 Học tổ hợp Bagging, Boosting, Stacking | Chỉ có Random Forest ở Ch.14 Dự án 1; còn lại — | 248–254 |
