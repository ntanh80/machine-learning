# Chỉ dẫn biên soạn bài giảng – Học máy K22 KHMT

## Nguyên tắc: Mục tiêu là chuẩn, giáo trình để tham khảo

Khi viết hoặc sửa nội dung một bài (`Bai 01` … `Bai 09`: file nội dung `.md`, notebook, code minh họa):

1. **File `1_objectives.md` (mục tiêu) của bài là chuẩn.** Cấu trúc, phạm vi, thứ tự mục và mức độ (Hiểu/Vận dụng) của nội dung phải bám theo các mục CĐR (ví dụ 3.1.1, 3.1.2, …) trong file này. Không thêm, bớt hay đổi thứ tự mục so với mục tiêu.
2. **Giáo trình chỉ để tham khảo**: `Bai giang/Giáo_trình_Machine_Learning 2026.pdf` (275 trang). Với mỗi mục CĐR, tra phần tương ứng trong giáo trình (bảng bên dưới) để lấy kiến thức, ví dụ, công thức, và ưu tiên dùng thuật ngữ, ký hiệu giống giáo trình cho thống nhất.
3. **Giáo trình không khớp với mục tiêu**: chương trong giáo trình không trùng một-một với bài, và có mục CĐR giáo trình không có. Khi đó vẫn viết đủ theo mục tiêu, dùng nguồn chuẩn khác (sách, tài liệu scikit-learn, …); không cắt mục tiêu cho vừa giáo trình. Phần giáo trình có nhưng mục tiêu không yêu cầu thì không đưa vào.
4. Nếu giáo trình có chỗ sai hoặc mâu thuẫn với mục tiêu, báo lại cho tác giả thay vì tự sửa ngầm.

## Tra cứu giáo trình theo từng mục CĐR

Số trang là số trang trong file PDF (không phải số in trên trang). "—" nghĩa là giáo trình không có, cần nguồn khác.

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

## Cách đọc PDF

Tên file có dấu tiếng Việt lưu ở dạng Unicode tổ hợp (NFD), gõ trực tiếp đường dẫn có thể báo không tìm thấy. Dùng glob:

```python
import glob, pypdf
r = pypdf.PdfReader(glob.glob('Bai giang/*.pdf')[0])
text = "\n".join(r.pages[i].extract_text() for i in range(74, 89))  # trang PDF 75–89 (chỉ số từ 0)
```

Hoặc dùng công cụ Read với tham số `pages` (tối đa 20 trang/lần) để xem cả hình và công thức.

## Quy ước thư mục mỗi bài

- `1_objectives.md` – mục tiêu, chuẩn đầu ra, phân bổ nội dung (chuẩn của bài).
- `2_<topic>.md` – nội dung lý thuyết chính, đề mục theo các mục CĐR. Tên file tiếng Anh theo chủ đề bài, `snake_case`, ví dụ `2_supervised_learning_linear_regression.md`, `2_gradient_descent.md`. Tiêu đề và nội dung bên trong vẫn viết tiếng Việt có dấu.
- `codes/` – code demo Python **viết dạng notebook `.ipynb`** (không viết `.py`), tên file có số thứ tự theo trình tự chuyên môn trong bài: `01_<name>.ipynb`, `02_<name>.ipynb`, … Mỗi notebook chạy độc lập từ đầu đến cuối (Restart & Run All), xen kẽ ô Markdown giải thích bằng tiếng Việt với ô code, đọc dữ liệu từ `../data/` (dùng `Path('..') / 'data'`, vì notebook chạy với thư mục hiện hành là `codes/`). Ngoại lệ duy nhất: ứng dụng phải chạy bằng lệnh riêng (ví dụ app Streamlit chạy bằng `streamlit run`) thì để `.py`. Trong file `2_<topic>.md` đặt link `codes/...` ngay sau đoạn code tương ứng.
- Đặt **tên file** (`.md`, và trong `codes/`, `images/`, `data/`) bằng **tiếng Anh, theo chủ đề**, `snake_case`, ví dụ `05_house_price_regression.ipynb`, `plot_overfitting.py` (script vẽ hình bắt đầu bằng `plot_`, cùng tên với hình nó sinh ra: `overfitting.png`), `house_prices.csv`. Tên cột dữ liệu tự tạo cũng bằng tiếng Anh (`area`, `bedrooms`, `distance`, `price`); dữ liệu tải về (ví dụ Kaggle) giữ nguyên tên gốc.
- Đặt tên trong code (biến, hàm, lớp, tham số) bằng **tiếng Anh, dễ hiểu và đơn giản**: dùng từ thông dụng, ngắn gọn, đọc tên là biết ý nghĩa (`price`, `area`, `loss`, `learning_rate`), tránh viết tắt khó đoán (`lr_v2`, `tmp_arr`) và tên quá dài; theo quy ước Python (`snake_case` cho biến/hàm, `PascalCase` cho lớp), ví dụ hàm `compute_gradient`; không dùng tên tiếng Việt không dấu như `gia_nha`, `toc_do_hoc`. Ký hiệu toán học theo giáo trình được giữ ngắn (`X`, `y`, `w`, `b`, `theta`, `eta`). Mỗi biến/hàm quan trọng có **comment tiếng Việt** giải thích ý nghĩa (nêu ký hiệu tương ứng trong bài giảng nếu có), ví dụ `learning_rate = 0.01  # tốc độ học η`.
- **Comment trong code viết bằng tiếng Việt có dấu** (không viết không dấu như `# Du lieu chieu cao`). Chuỗi hiển thị (`print`, nhãn trục, tiêu đề hình) cũng viết tiếng Việt có dấu.
- `images/` – hình minh họa (kèm script `.py` sinh hình), `data/` – dữ liệu.
- Viết bằng tiếng Việt có dấu.
