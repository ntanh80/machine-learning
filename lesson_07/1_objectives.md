# Bài 7. Máy véc-tơ hỗ trợ (SVM)

## 1. Mục tiêu bài học

Bài học giới thiệu thuật toán SVM (Support Vector Machine – Máy véc-tơ hỗ trợ). Cụ thể, bài học giúp sinh viên:

- Nắm được ý tưởng tìm siêu phẳng phân chia (hyperplane) có biên lớn nhất giữa hai lớp.
- Hiểu khái niệm véc-tơ hỗ trợ (support vectors) và vai trò của chúng trong việc xác định siêu phẳng tối ưu.
- Phân biệt SVM tuyến tính và phi tuyến qua minh họa trực quan trên dữ liệu hai chiều.

## 2. Chuẩn đầu ra

Sau khi hoàn thành bài học, sinh viên có thể:

1. Trình bày được nguyên lý hoạt động của SVM và khái niệm siêu phẳng phân tách.
2. Mô tả được vai trò của véc-tơ hỗ trợ trong việc xác định biên phân chia.
3. Phân biệt được SVM tuyến tính và phi tuyến, cũng như vai trò của hàm kernel.
4. Áp dụng được SVM để giải bài toán phân loại nhị phân trên dữ liệu thực tế.
5. Phân tích được ưu nhược điểm của SVM, đặc biệt với dữ liệu nhiều chiều hoặc không tuyến tính.

## 3. Nội dung và phân bổ thời gian

| CĐR | Nội dung | Thời lượng | Hình thức | Mức độ |
| --- | --- | --- | --- | --- |
| 7.1 | Mô hình SVM | 3 tiết | Giảng dạy lý thuyết | Hiểu |
| 7.2 | Thực hành phân loại bằng SVM | 1 tiết | Hướng dẫn thực hành, làm bài tập | Vận dụng |

### CĐR 7.1. Hiểu về mô hình SVM

- 7.1.1. Siêu phẳng phân chia (hyperplane) và biên tối ưu
- 7.1.2. Bài toán tối ưu và khái niệm lề (margin)
- 7.1.3. Hàm kernel ánh xạ dữ liệu không tuyến tính

### CĐR 7.2. Vận dụng SVM để phân loại dữ liệu

- 7.2.1. Xây dựng mô hình SVM trên bộ dữ liệu có sẵn
