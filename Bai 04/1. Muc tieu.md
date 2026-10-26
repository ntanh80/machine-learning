# Bài 4. Hồi quy Logistic và thuật toán PLA

## 1. Mục tiêu bài học

Bài học giới thiệu một mô hình và một thuật toán phân loại quan trọng trong học máy có giám sát: **mô hình hồi quy Logistic** và **thuật toán PLA** (Perceptron Learning Algorithm). Cụ thể, bài học giúp sinh viên:

- Nắm được nguyên lý hoạt động của mô hình hồi quy Logistic và vai trò của hàm sigmoid trong bài toán phân loại nhị phân.
- Phân biệt hồi quy tuyến tính và hồi quy Logistic về mục tiêu dự đoán và dạng đầu ra.
- Tiếp cận thuật toán PLA – một phương pháp học có giám sát kinh điển cho bài toán phân loại tuyến tính.
- Hình thành tư duy phân tích để lựa chọn mô hình phân loại phù hợp với đặc điểm của dữ liệu.

## 2. Chuẩn đầu ra

Sau khi hoàn thành bài học, sinh viên có thể:

1. Trình bày được công thức của mô hình hồi quy Logistic và giải thích cách hàm sigmoid biến đổi đầu ra thành xác suất.
2. Xây dựng được hàm mất mát của hồi quy Logistic từ phương pháp ước lượng hợp lý cực đại (MLE) và tối ưu hàm mất mát bằng Gradient Descent.
3. Mô tả được nguyên lý cập nhật trọng số của thuật toán PLA.
4. Áp dụng hồi quy Logistic và PLA để giải các bài toán phân loại nhị phân cơ bản.
5. Phân tích được ranh giới quyết định (decision boundary) và đánh giá mô hình phân loại bằng các chỉ số phù hợp.

## 3. Nội dung và phân bổ thời gian

| CĐR | Nội dung | Thời lượng | Hình thức | Mức độ |
| --- | --- | --- | --- | --- |
| 4.1 | Mô hình hồi quy Logistic | 2 tiết | Giảng dạy lý thuyết | Hiểu |
| 4.2 | Thuật toán PLA | 1 tiết | Giảng dạy lý thuyết | Hiểu |
| 4.3 | Thực hành mô hình hồi quy Logistic | 1 tiết | Hướng dẫn thực hành, làm bài tập | Vận dụng |

### CĐR 4.1. Hiểu mô hình hồi quy Logistic

- 4.1.1. Phát biểu bài toán phân loại dữ liệu
- 4.1.2. Hàm sigmoid
- 4.1.3. Mô hình hồi quy Logistic
- 4.1.4. Hàm mất mát và tối ưu hàm mất mát (MLE, Gradient Descent)
- 4.1.5. Ma trận nhầm lẫn (Confusion Matrix)
- 4.1.6. Các chỉ số đánh giá: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### CĐR 4.2. Hiểu thuật toán PLA

- 4.2.1. Khái niệm Perceptron
- 4.2.2. Phương trình siêu phẳng phân loại
- 4.2.3. Hàm mất mát và quy tắc cập nhật trọng số

### CĐR 4.3. Vận dụng hồi quy Logistic dự đoán nguy cơ mắc ung thư phổi

- 4.3.1. Thu thập dữ liệu từ Kaggle
- 4.3.2. Tiền xử lý dữ liệu
- 4.3.3. Xây dựng mô hình hồi quy Logistic
- 4.3.4. Đánh giá mô hình
