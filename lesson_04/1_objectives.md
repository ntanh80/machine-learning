# Bài 4. Hồi quy Logistic và thuật toán PLA

## 1. Mục tiêu bài học

Bài học giới thiệu một mô hình và một thuật toán phân loại quan trọng trong học máy có giám sát: **mô hình hồi quy Logistic** và **thuật toán PLA** (Perceptron Learning Algorithm). 

Cụ thể, bài học giúp sinh viên:

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

## 3. Nội dung

### CĐR 1. Hiểu mô hình hồi quy Logistic

- Phát biểu bài toán phân loại dữ liệu
- Hàm sigmoid
- Mô hình hồi quy Logistic
- Hàm mất mát và tối ưu hàm mất mát (MLE, Gradient Descent)
- Ma trận nhầm lẫn (Confusion Matrix)
- Các chỉ số đánh giá: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### CĐR 2. Hiểu thuật toán PLA

- Khái niệm Perceptron
- Phương trình siêu phẳng phân loại
- Hàm mất mát và quy tắc cập nhật trọng số

### CĐR 3. Vận dụng hồi quy Logistic dự đoán nguy cơ mắc ung thư phổi

- Thu thập dữ liệu từ Kaggle
- Tiền xử lý dữ liệu
- Xây dựng mô hình hồi quy Logistic
- Đánh giá mô hình
