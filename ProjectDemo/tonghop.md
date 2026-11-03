# Tổng hợp và phân tích các hướng nghiên cứu phát hiện giao dịch gian lận trên dữ liệu mất cân bằng

*Detecting fraudulent transactions with imbalanced data: a synthesis of research directions*

## Tóm tắt

Tài liệu này tổng hợp 21 bài báo xuất bản giai đoạn 2023–2026 (17 bài tiếng Anh, 4 bài tiếng Việt) về phát hiện gian lận trong điều kiện mất cân bằng lớp. Các bài được xếp vào bảy hướng tiếp cận: (1) học có giám sát làm nền, (2) tái cân bằng cổ điển, (3) sinh dữ liệu sâu, (4) học nhạy chi phí và hàm mất mát, (5) học trên đồ thị, (6) phát hiện bất thường và học biểu diễn, (7) kết hợp mô hình và mô hình diễn giải được.

Ba kết luận chính được rút ra. Thứ nhất, cách đánh giá ảnh hưởng đến kết quả không kém việc chọn mô hình: cùng XGBoost trên cùng bộ dữ liệu ULB, F1 lớp gian lận được báo cáo dao động từ khoảng 0,81 đến 0,94 tùy giao thức. Thứ hai, tái cân bằng không phải lúc nào cũng có lợi: một số bài cho thấy mô hình cây không tái cân bằng đạt F1 cao hơn hoặc tương đương khi áp dụng SMOTE, và cải thiện lớn chỉ xuất hiện ở số ít bài sinh dữ liệu sâu. Thứ ba, các kết quả gần tuyệt đối trong tập bài báo đều cần được kiểm tra lại, vì nhiều trường hợp gắn với rò rỉ dữ liệu, tập kiểm tra đã cân bằng hoặc chỉ số accuracy.

## 1. Phạm vi và phương pháp

**Nguồn dữ liệu:** toàn bộ 21 tệp PDF trong thư mục `pages/`. Theo `pages/papers.csv`, 17 bài là bản tiền ấn phẩm trên arXiv và 4 bài đăng ở hội nghị, tạp chí hoặc cơ sở dữ liệu khoa học trong nước. Vì vậy phần lớn kết quả chưa thể coi là đã qua phản biện đồng cấp nếu chưa xác minh riêng từng bài. Ba bài tiếng Việt khác được liệt kê trong `papers.csv` không có bản PDF nên không thuộc phạm vi tổng hợp.

**Cách đọc:** nội dung từng bài được đọc trực tiếp từ văn bản trích xuất. Các bảng dạng ảnh quan trọng được đối chiếu trên trang PDF gốc (Bảng 1–3 của [4], Bảng 7 của [5]). Với [10], phần lớn kết quả nằm trong hình, nên chỉ các số liệu ghi ở dạng văn bản được sử dụng.

**Quy ước:** số liệu là số do tác giả báo cáo. Các mục **Nhận xét** là đánh giá của tài liệu này, gồm cả những phép tính lại từ số liệu trong bài; chúng là suy luận, không phải khẳng định của tác giả. Nếu không ghi khác, precision, recall và F1 tính cho lớp gian lận.

### 1.1. Các bộ dữ liệu

| Bộ dữ liệu | Quy mô, tỷ lệ gian lận | Tính chất | Bài sử dụng |
| --- | --- | --- | --- |
| ULB/Kaggle thẻ châu Âu 2013 | 284.807 giao dịch, 492 gian lận (0,172%) | thật, 28 đặc trưng đã qua PCA | [4], [7], [8], [10], [12], [14], [15], [19], [21] |
| IEEE-CIS | khoảng 590.000 giao dịch, 3,5% | thật, đã ẩn danh | [17], [20] |
| PaySim | khoảng 6,36 triệu, 0,129% | mô phỏng | [18] |
| BankSim | 594.643 giao dịch, 7.200 gian lận | mô phỏng | [8] |
| Sparkov | 1.000 khách hàng, 800 đơn vị chấp nhận thẻ | mô phỏng | [14] |
| Kaggle 2023 | 568.630 giao dịch, đã cân bằng 50:50 | đã ẩn danh | [9] |
| Thẻ trực tuyến | 94.682 giao dịch, 2,3% | Kaggle | [1] |
| Dữ liệu viễn thông riêng | không công bố, tối thiểu 5% ở thí nghiệm khó nhất | thật, trên 150 đặc trưng | [5] |
| Medicare | 556.703 yêu cầu thanh toán, 38,1% | hồ sơ y tế | [13] |
| Bảo hiểm ô tô | 39.981 bản ghi | hồ sơ bồi thường | [11] |
| Sichuan, BUPT | IR 0,4735 và 0,0809 | đồ thị viễn thông | [2] |
| YelpChi, Amazon, Comp | đồ thị đa quan hệ | Amazon có khoảng 10% gian lận | [16] |
| Dữ liệu không thuộc lĩnh vực gian lận | UCI (IR 1,87–2,78); Olympic, điều tra dân số Mỹ | kiểm chứng phương pháp | [3], [6] |

## 2. Phân nhóm các hướng tiếp cận

Tiêu chí phân nhóm là **vị trí can thiệp chính** của nghiên cứu trong quy trình học: dữ liệu, hàm mục tiêu, cấu trúc biểu diễn hay bước kết hợp. Mỗi bài được xếp vào một hướng chính; phần giao thoa được nêu ở phần phân tích.

| Hướng | Vị trí can thiệp | Bài chính |
| --- | --- | --- |
| H1. Học có giám sát làm nền | chọn mô hình, không đặt tái cân bằng làm trọng tâm | [5], [9], [17] |
| H2. Tái cân bằng cổ điển | phân phối dữ liệu huấn luyện, bằng nội suy hoặc lấy mẫu | [1], [3], [7], [13], [15], [18] |
| H3. Sinh dữ liệu sâu | phân phối dữ liệu huấn luyện, bằng mô hình sinh | [6], [12], [14], [21] |
| H4. Học nhạy chi phí, hàm mất mát | hàm mục tiêu | [11] |
| H5. Học trên đồ thị | biểu diễn quan hệ và truyền thông điệp | [2], [16] |
| H6. Phát hiện bất thường, học biểu diễn | mô hình hóa hành vi bình thường hoặc trích đặc trưng | [8], [10] |
| H7. Kết hợp mô hình, mô hình diễn giải được | bước tổ hợp điểm hoặc cấu trúc mô hình | [4], [19], [20] |

## 3. Tổng hợp theo từng hướng

### 3.1. H1: học có giám sát làm nền

**Ý tưởng cốt lõi:** xác định năng lực của các bộ phân lớp phổ biến trước khi bổ sung kỹ thuật xử lý mất cân bằng.

**Phương pháp và kết quả:**

- [17] so sánh Logistic Regression, Random Forest, LightGBM và GRU trên IEEE-CIS. F1 có trọng số đạt 0,93–0,97, nhưng F1 lớp gian lận chỉ đạt 0,18 (Logistic Regression), 0,53 (Random Forest), 0,50 (LightGBM) và 0,37 (GRU). GRU có recall cao nhất (0,54) với precision 0,28.
- [5] khảo sát XGBoost trên dữ liệu riêng của doanh nghiệp viễn thông. Với 100.000 mẫu, F1 của mô hình đã tinh chỉnh giảm từ 0,8931 (50% mẫu dương) xuống 0,5790 (5% mẫu dương). Tinh chỉnh bằng tìm kiếm ngẫu nhiên chỉ cải thiện nhất quán ở quy mô 100.000 mẫu; ở 1.000 mẫu và 5% dương, F1 giảm từ 0,4000 xuống 0,2542. Lấy mẫu cân bằng tập huấn luyện không tạo cải thiện nhất quán.
- [9] so sánh Decision Tree, Random Forest, GBM và XGBoost trên Kaggle 2023, với accuracy từ 99,92% đến 99,985%; Random Forest dẫn đầu.

**Nhận xét:** [9] dùng dữ liệu đã cân bằng cả khi huấn luyện lẫn kiểm tra (Class có trung bình 0,5, tập kiểm tra 50:50), nên kết quả không phản ánh bài toán mất cân bằng dù bài trình bày như vậy. [17] chọn siêu tham số theo F1 có trọng số, chỉ số bị lớp đa số chi phối; điều này có thể là một nguyên nhân khiến kết quả trên lớp gian lận thấp. Trong [17], nhận định GRU có F1 “tương đương” mô hình cây cũng không khớp với số liệu 0,37 so với 0,53.

### 3.2. H2: tái cân bằng cổ điển

**Ý tưởng cốt lõi:** thay đổi phân phối lớp trong tập huấn luyện bằng SMOTE, giảm mẫu lớp đa số hoặc kết hợp hai cách.

**Phương pháp và kết quả:**

- [15] so sánh năm mô hình với ba chiến lược, đánh giá trên tập kiểm tra giữ nguyên phân phối gốc. Giảm mẫu về 1:1 cho recall tới 0,9593 nhưng precision rất thấp (0,0501 với Logistic Regression, 0,0831 với Random Forest). Phương pháp lai đưa tỷ lệ gian lận lên khoảng 1–2,5% cho kết quả cân bằng nhất. Recall tăng ở cả năm mô hình, ví dụ MLP từ 0,659 lên 0,813 và Logistic Regression từ 0,634 lên 0,781; F1 tăng nhẹ ở tất cả, Random Forest chỉ từ 0,858 lên 0,862.
- [7] kiểm soát rò rỉ dữ liệu trên XGBoost. SMOTE trước khi chia tập cho F1 99,97%. Khi chỉ áp dụng trên tập huấn luyện, F1 là 92,56%, thấp hơn mô hình không tái cân bằng (94,00%). Cấu hình tốt nhất là CGAN kết hợp `scale_pos_weight` (95,00%).
- [18] dùng Random Forest, SMOTE và GridSearchCV trên PaySim, F1 tăng từ 0,87 lên 0,91, recall tăng từ 0,79 lên 0,90 và precision giảm từ 0,97 xuống 0,92.
- [1] dùng Random Forest và SMOTE, báo cáo accuracy và F1 khoảng 98%.
- [13] kết hợp SMOTE, chọn đặc trưng và năm mô hình trên Medicare. Random Forest đạt accuracy 98,8%, F1 98,4%, AUC 1,000 trên tập kiểm định.
- [3] đề xuất BLSMOTE-LOF: sinh mẫu tại vùng biên rồi loại nhiễu bằng hệ số ngoại lai cục bộ. So với dữ liệu gốc, AUC tăng từ 70,7% lên 75,01% trên Pima và từ 51,61% lên 62,28% trên Haberman.

**Nhận xét:**

- [15] nêu rõ tỷ lệ tối ưu được chọn dựa trên precision và recall “trên tập kiểm tra gốc”, vì vậy mức cải thiện của phương pháp lai có khả năng lạc quan.
- [18] chọn mô hình theo kết quả trên tập kiểm tra và thực hiện kiểm định chéo trên dữ liệu đã SMOTE, nên mẫu tổng hợp có thể chứa thông tin của fold kiểm định.
- Ở [1], tổng ma trận nhầm lẫn là 175.124, đúng bằng 2 × 87.562, tức toàn bộ dữ liệu sau SMOTE. Điều này gợi ý kết quả 98% được đo trên dữ liệu đã cân bằng thay vì tập kiểm tra độc lập. Số liệu 87.562 + 2.052 = 89.614 cũng không khớp tổng 94.682 mà bài nêu.
- [13] mô tả gian lận chiếm dưới 1% yêu cầu thanh toán, nhưng bảng thống kê của chính bài cho tỷ lệ 38,1%. Bài nêu cơ chế học thích nghi trong đóng góp nhưng không có thí nghiệm huấn luyện lại. AUC 1,000 cần được kiểm tra rò rỉ, đặc biệt nếu nhãn gắn ở cấp nhà cung cấp mà dữ liệu không được chia theo nhà cung cấp.
- [3] chỉ kiểm chứng trên dữ liệu có IR dưới 3, xa mức mất cân bằng của bài toán gian lận. Mô tả thuật toán cũng chưa nhất quán: giả mã giữ các phần tử có lof nhỏ hơn ngưỡng, còn phần diễn giải lại loại các phần tử đó.

### 3.3. H3: sinh dữ liệu sâu

**Ý tưởng cốt lõi:** học phân phối của lớp gian lận để sinh mẫu đa dạng hơn phép nội suy tuyến tính của SMOTE.

**Phương pháp và kết quả:**

- [14] đề xuất CPAC, bộ phân lớp dựa trên nguyên mẫu và chú ý, gắn vào bộ mã hóa của VAE-GAN để định hình không gian ẩn theo cả hai lớp. Bài lập luận rằng huấn luyện bộ sinh chỉ trên lớp gian lận dẫn tới mẫu gần trùng lặp và bộ phân lớp quá tự tin. Tốt nhất với XGBoost: precision 94,67%, recall 92,85%, F1 93,74%.
- [12] kết hợp GAN với khối mã hóa Transformer, sinh 5.000 mẫu từ 394 mẫu gian lận. Trên bốn bộ phân lớp, AUC đạt 0,9939–0,9963 và F1 đạt 0,97–0,99.
- [21] đề xuất Q-SYNTH: mạch lượng tử tham số hóa làm bộ sinh, mạng cổ điển làm bộ phân biệt. KS trung vị bằng 0,069, tốt hơn GAN cổ điển (0,185) và kém SMOTE (0,035). Bộ phát hiện độc lập cho AUC 0,475, gần mức ngẫu nhiên. Với QNN, F1 đạt 0,9524, so với 0,9615 của GAN và 0,9355 của SMOTE.
- [6] đề xuất DGGAN sinh dữ liệu nhân khẩu học, kết hợp sinh mẫu trong lúc huấn luyện theo cấp số nhân để giảm sụp đổ chế độ. Điểm trung bình đạt 0,88 và 0,92 về độ tương đồng.

Bằng chứng ngược chiều đến từ [20]: bổ sung CTGAN làm giảm hiệu năng mọi mô hình, ví dụ AUC của LightGBM giảm từ 0,9343 xuống 0,8988 và AUPRC của tổ hợp tốt nhất giảm từ 0,6694 xuống 0,6120. Tác giả giới hạn kết luận cho cấu hình đặc trưng số đã dùng. [7] ngược lại ghi nhận CGAN cải thiện khi kết hợp với trọng số lớp.

**Nhận xét:**

- Mức tăng của [14] khiêm tốn hơn cách diễn đạt. XGBoost không tái cân bằng đã đạt F1 93,00%, trong khi Random Forest với SMOTE hoặc VAE-GAN thông thường đạt 94,56%, cao hơn con số 93,74%. Tập kiểm tra chỉ có 49 mẫu gian lận, nên một mẫu tương ứng khoảng 2 điểm phần trăm recall; khác biệt 90,81% và 92,85% do đó nằm ở mức xấp xỉ một mẫu. Các so sánh với hai nghiên cứu của Shi và cộng sự, Ahmed và cộng sự dùng bản tái lập của nhóm tác giả [14], không phải mã gốc.
- Kết quả của [12] vượt xa các bài còn lại trên cùng bộ dữ liệu và cần được kiểm chứng độc lập. Số liệu trong văn bản và trong bảng không khớp (Logistic Regression: precision 0,957 và recall 0,908 so với 0,98 và 0,97). Bài cũng không làm rõ thành phần tập kiểm tra ở kịch bản “tập GAN cân bằng”.
- [21] chọn đặc trưng trên toàn bộ dữ liệu có nhãn trước khi chia tập, làm việc trên 4 thành phần PCA, chạy trên bộ mô phỏng và đánh giá chính trên tập cân bằng 984 + 984. Do đó bài chứng minh tính khả thi hơn là lợi thế thực tế của phần cứng lượng tử.
- [6] không đo hiệu năng phát hiện gian lận và không dùng dữ liệu gian lận, nên chưa kết luận được về lợi ích cho bài toán này.

### 3.4. H4: học nhạy chi phí và hàm mất mát

**Ý tưởng cốt lõi:** giữ nguyên dữ liệu, thay đổi mức phạt để lỗi trên lớp hiếm có trọng số lớn hơn.

**Phương pháp và kết quả:**

- [11] đề xuất huấn luyện ba giai đoạn cho LSTM: hàm thay thế lồi để khởi tạo, focal loss không lồi trung gian và focal loss chuẩn. Trên dữ liệu bảo hiểm ô tô, F1 đạt 0,4149 và AUC 0,6303, tốt hơn các lịch huấn luyện đơn lẻ. Bài có SHAP để diễn giải.
- [7] dùng `scale_pos_weight` trong cấu hình tốt nhất; [19] dùng trọng số lớp cho các mô hình so sánh.
- [2] tự động học ma trận chi phí bằng gradient (xem mục 3.5). Khi bỏ mô-đun chi phí trên BUPT, recall gian lận giảm từ 0,9539 xuống 0,7012 và recall lớp Courier từ 0,8203 xuống 0,2592.

**Nhận xét:** từ accuracy, precision và recall trong [11] có thể suy ra tỷ lệ dương trong dữ liệu đánh giá khoảng 37–45%, gần với 44,4% của mức tái cân bằng 80% mà bài mô tả. Điều này gợi ý tập đánh giá đã được tái cân bằng, trái với mục tiêu đánh giá trong điều kiện mất cân bằng cực độ. Bài cũng kết hợp cả SMOTE và focal loss, nên hiệu ứng riêng của hàm mất mát chưa được tách bạch.

### 3.5. H5: học trên đồ thị

**Ý tưởng cốt lõi:** gian lận có cấu trúc quan hệ; kẻ gian ngụy trang bằng cách kết nối với người dùng bình thường. Mất cân bằng vì vậy tồn tại cả trong vùng lân cận của từng nút, không chỉ ở tỷ lệ lớp toàn cục.

**Phương pháp và kết quả:**

- [2] đề xuất CSGNN gồm bộ lấy mẫu láng giềng bằng học tăng cường và bộ học nhạy chi phí. AUC đạt 0,9539 trên Sichuan và 0,9698 trên BUPT. Mức đóng góp của các mô-đun tăng theo độ mất cân bằng: bỏ một mô-đun làm giảm khoảng 1–3 điểm trên Sichuan, còn trên BUPT giảm 3–7 điểm khi bỏ bộ lấy mẫu và 17–27 điểm khi bỏ mô-đun chi phí.
- [16] đề xuất MimbFD với hai mô-đun: lan truyền nhãn toàn cục để giảm mất cân bằng topo, và giải tương quan biểu diễn để giảm mất cân bằng lớp. AUC đạt 96,62% trên Amazon, 92,41% trên YelpChi và 65,12% trên Comp.

**Nhận xét:** ở [2], recall lớp bình thường trên BUPT giảm từ 0,9603 xuống 0,7208 khi bật mô-đun chi phí, nghĩa là mức tăng recall gian lận đi kèm chi phí báo động giả đáng kể. Ở [16], GAGA vẫn nhỉnh hơn MimbFD trên YelpChi về AUC (92,99 so với 92,41) và F1 (80,15 so với 80,08). Các bộ dữ liệu đều không phải giao dịch tài chính thuần túy và [2] có cạnh được xây từ độ tương đồng đặc trưng, nên khả năng chuyển giao sang giao dịch thẻ còn cần kiểm chứng.

### 3.6. H6: phát hiện bất thường và học biểu diễn

**Ý tưởng cốt lõi:** mô hình hóa hành vi bình thường rồi coi sai lệch là dấu hiệu gian lận, giảm phụ thuộc vào nhãn gian lận.

**Phương pháp và kết quả:**

- [10] dùng autoencoder sâu trên dữ liệu ULB, AUC đạt 95,33%; chọn đặc trưng bằng thuật toán dơi nhị phân, loại 11 đặc trưng, nâng AUC lên 96,21%. Bài cảnh báo Random Forest với SMOTE gần như lý tưởng có thể là dấu hiệu quá khớp.
- [8] kết hợp autoencoder và Isolation Forest. So với Isolation Forest đơn lẻ, F1 tăng từ 0,3115 lên 0,3749 trên ULB và từ 0,3949 lên 0,6593 trên BankSim.

**Nhận xét:** [8] kết luận hiệu quả “gần như tuyệt đối” 99,77%, nhưng đó là accuracy; cùng bảng kết quả cho precision 0,3644 và recall 0,3861 trên ULB, tức phần lớn gian lận bị bỏ sót. Bài cũng gán lớp bình thường vào ô True Positive, khiến bảng dễ bị đọc sai. Đây là minh họa rõ nhất cho nguy cơ dùng accuracy. Hướng này có giá trị khi thiếu nhãn, song các kết quả hiện có chưa đạt độ chính xác của mô hình có giám sát trên cùng dữ liệu.

### 3.7. H7: kết hợp mô hình và mô hình diễn giải được

**Ý tưởng cốt lõi:** khai thác sai số bổ sung giữa nhiều mô hình, hoặc đạt hiệu năng cao mà vẫn giải thích được quyết định.

**Phương pháp và kết quả:**

- [20] áp dụng Phân tích tổ hợp (CFA) trên IEEE-CIS, tìm kiếm 480 cấu hình trên tập kiểm định. Tổ hợp Random Forest, XGBoost và LightGBM có trọng số đa dạng đạt AUC 0,9405, AUPRC 0,6699, F1 0,6373, so với 0,9343; 0,6480; 0,6093 của LightGBM. Khoảng tin cậy bootstrap của các chênh lệch đều không chứa 0. Stacking kém hơn mô hình đơn tốt nhất; TabNet đa dạng nhất nhưng không được chọn vì yếu.
- [19] tối ưu Explainable Boosting Machine bằng phương pháp Taguchi, không tái cân bằng, chọn 18 đặc trưng. ROC-AUC đạt 0,983, vượt mốc 0,975 trước đó; mô hình cung cấp giải thích toàn cục và cục bộ.
- [4] thử bốn hướng tiền xử lý, rồi Voting, Stacking và logic mờ. SMOTE-ENN làm precision trên tập kiểm tra giảm mạnh, nên nhóm tác giả không tái cân bằng. Logic mờ trên hai biến V4, V17 đạt precision 0,75 và recall 0,73, đồng thời cho mức độ khả năng gian lận thay vì nhãn cứng.

**Nhận xét:**

- [20] là bài có giao thức chặt chẽ nhất trong nhóm: tách rõ huấn luyện, kiểm định và kiểm tra, có khoảng tin cậy, có chỉ số theo ngân sách cảnh báo. Mức tăng khiêm tốn và ổn định này đáng tin hơn những mức tăng lớn không có khoảng tin cậy. Hạn chế là chia ngẫu nhiên phân tầng, chưa chia theo thời gian.
- [19] kết luận EBM vượt trội, nhưng với 18 đặc trưng Random Forest có F1 cao hơn (0,837 so với 0,831) và XGBoost có recall cao hơn (0,822 so với 0,763). EBM chỉ dẫn đầu về ROC-AUC, chỉ số không nhạy với tỷ lệ lớp hiếm. Không thấy mô tả tập kiểm tra độc lập cho toàn bộ quá trình chọn đặc trưng và siêu tham số.
- [4] kết luận Voting Logistic Regression và XGBoost cao hơn mọi mô hình đơn, nhưng F1 của tổ hợp là 0,7798, thấp hơn XGBoost đơn lẻ là 0,8778 trong Bảng 1 của chính bài (mức tăng nằm ở recall 0,8673 so với 0,8061, đổi lại precision giảm từ 0,9634 xuống 0,7083). Bài cũng diễn giải khoảng cách precision giữa tập huấn luyện đã cân bằng và tập kiểm tra gốc là quá khớp, trong khi khác biệt này phần lớn đến từ thay đổi tỷ lệ lớp. Trọng số Voting được chọn khi chỉ có hai tập huấn luyện và kiểm tra.

## 4. Phân tích ưu, nhược điểm

| Hướng | Ưu điểm | Nhược điểm | Điều kiện áp dụng phù hợp |
| --- | --- | --- | --- |
| H1. Có giám sát làm nền | đơn giản, nhanh, là mốc so sánh mạnh trên dữ liệu bảng [7], [17], [20] | suy giảm mạnh khi tỷ lệ dương giảm [5]; phụ thuộc nhãn và chỉ nhận ra mẫu đã biết | luôn cần thực hiện đầu tiên; có nhãn đủ và đặc trưng tốt |
| H2. Tái cân bằng cổ điển | dễ cài đặt, tăng recall [15], [18] | giảm precision mạnh khi cân bằng hoàn toàn [15]; dễ rò rỉ nếu áp dụng sai thứ tự [7]; không cải thiện nhất quán [4], [5], [7] | mô hình nhạy với phân phối lớp (Logistic Regression, MLP); tỷ lệ mục tiêu nhẹ, chọn trên tập kiểm định |
| H3. Sinh dữ liệu sâu | mẫu đa dạng hơn nội suy; có thể đo độ trung thực [14], [21] | tốn tài nguyên, khó huấn luyện; kết quả mâu thuẫn [12], [20]; với ít mẫu gian lận, mô hình sinh tự nó thiếu dữ liệu | dữ liệu có cấu trúc rõ, có đủ mẫu gian lận, cần chia sẻ dữ liệu tổng hợp |
| H4. Nhạy chi phí, hàm mất mát | không đổi dữ liệu, gắn trực tiếp với chi phí kinh doanh [2], [7] | khó xác định chi phí; nhiều siêu tham số; focal loss có thể làm tối ưu kém ổn định [11] | có ước lượng chi phí báo động giả và bỏ sót; cần giữ phân phối thật |
| H5. Đồ thị | khai thác quan hệ, phát hiện ngụy trang và nhóm gian lận [2], [16] | cần dữ liệu quan hệ; chi phí tính toán lớn; có thể gặp lỗi bộ nhớ [16] | có định danh thực thể và liên kết rõ: viễn thông, tài khoản trung gian, đánh giá |
| H6. Bất thường, biểu diễn | giảm phụ thuộc nhãn, có khả năng với mẫu mới [8], [10] | precision và recall thấp hơn có giám sát [8]; khó chọn kiến trúc và ngưỡng | thiếu nhãn, trễ nhãn, làm tín hiệu bổ sung cho mô hình có giám sát |
| H7. Kết hợp, diễn giải được | mức tăng ổn định [20]; minh bạch, phù hợp yêu cầu tuân thủ [19] | tốn chi phí suy luận; rủi ro quá khớp khi tìm kiếm lớn; cần mô hình thành phần đủ mạnh [20] | hệ thống đã có nhiều mô hình mạnh; lĩnh vực yêu cầu giải thích |

## 5. So sánh và đối chiếu

### 5.1. So sánh các hướng theo cùng tiêu chí

Mức đánh giá là định tính, dựa trên bằng chứng trong 21 bài, không phải đo lường trực tiếp.

| Tiêu chí | H1 | H2 | H3 | H4 | H5 | H6 | H7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Giữ nguyên phân phối huấn luyện | có | không | không | có | có | có | tùy thành phần |
| Cần nhãn gian lận | cao | cao | cao | cao | trung bình | thấp | cao |
| Chi phí tính toán | thấp | thấp | cao | trung bình | cao | trung bình | trung bình |
| Rủi ro phương pháp luận chính | chỉ số gây hiểu nhầm | rò rỉ, chọn tỷ lệ trên tập kiểm tra | mẫu không thực, khó tái lập | chi phí chủ quan | chuyển giao lĩnh vực | ngưỡng, precision thấp | quá khớp khi tìm kiếm |
| Khả năng diễn giải | trung bình | trung bình | thấp | trung bình | thấp | thấp | cao |
| Mức nhất quán của bằng chứng | cao | trung bình | thấp | thấp | trung bình | thấp | trung bình |

### 5.2. Kết quả tiêu biểu theo giao thức

Bảng chỉ ghi nhận kết quả tiêu biểu của từng bài; các dòng khác dữ liệu hoặc khác cách đánh giá **không so sánh trực tiếp với nhau**.

| Bài | Dữ liệu | Phương pháp | Đánh giá | Kết quả |
| --- | --- | --- | --- | --- |
| [20] | IEEE-CIS | CFA 3 mô hình cây | kiểm tra độc lập, có bootstrap | AUPRC 0,6699; F1 0,6373 |
| [17] | IEEE-CIS | Random Forest | tách chính thức | F1 0,53; AUC 0,91 |
| [15] | ULB | MLP + lai | test gốc, tỷ lệ chọn trên test | F1 0,800; recall 0,813 |
| [15] | ULB | Random Forest + lai | test gốc, tỷ lệ chọn trên test | F1 0,862 |
| [7] | ULB | XGBoost + CGAN + trọng số | test sau khi chia, không tái cân bằng test | F1 95,00% |
| [14] | ULB | XGBoost + VAE-GAN + CPAC | test 49 mẫu gian lận | F1 93,74%; AUC 98,47% |
| [19] | ULB | EBM | kiểm định chéo | ROC-AUC 0,983; F1 0,831 |
| [21] | ULB | QNN + Q-SYNTH | tập cân bằng | F1 0,9524 |
| [18] | PaySim | RF + SMOTE + HPO | test gốc, dữ liệu mô phỏng | F1 0,91 |
| [16] | Amazon | MimbFD | đồ thị, chia 4:2:4 | AUC 96,62%; F1 91,77% |
| [11] | Bảo hiểm | LSTM + focal loss | 10-fold, nghi tập đánh giá đã tái cân bằng | F1 0,4149; AUC 0,6303 |
| [8] | ULB | AE + Isolation Forest | test giữ phân phối | F1 0,3749; recall 0,3861 |

### 5.3. Cùng XGBoost, cùng dữ liệu ULB, khác giao thức

| Bài | Cấu hình | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| [7] | không tái cân bằng | 96,50% | 91,50% | 94,00% |
| [14] | không tái cân bằng | 95,44% | 90,81% | 93,00% |
| [4] | không tái cân bằng, 80:20 | 96,34% | 80,61% | 87,78% |
| [15] | không tái cân bằng, 75:25 | 94,0% | 76,4% | 84,3% |
| [19] | 18 đặc trưng, `scale_pos_weight` = 10 | 80,6% | 82,2% | 81,4% |

Chênh lệch hơn 12 điểm F1 cho cùng một thuật toán lớn hơn phần lớn mức cải thiện mà các phương pháp đề xuất công bố. Vì vậy mức tăng vài điểm giữa hai bài khác nhau chưa đủ để kết luận phương pháp nào tốt hơn; lựa chọn đáng tin cậy chỉ có được khi so sánh trong cùng một giao thức, như [15] và [20] đã làm.

### 5.4. Các mâu thuẫn chính

- **SMOTE có hữu ích không:** [15] và [18] ghi nhận tăng recall và F1; [4], [5] và [7] ghi nhận không cải thiện hoặc giảm. Điểm chung hợp lý là SMOTE giúp mô hình đơn giản nhiều hơn mô hình cây và đổi precision lấy recall. [15] cho thấy Logistic Regression và MLP hưởng lợi nhất.
- **Mô hình sinh sâu có vượt SMOTE không:** [12] và [14] khẳng định có, [21] cho thấy đánh đổi giữa độ trung thực và hiệu năng, còn [20] cho thấy CTGAN gây hại. Chưa có đồng thuận, và kết luận phụ thuộc mạnh vào cách biểu diễn đặc trưng.
- **Random Forest hay gradient boosting:** [9], [15] và [17] nghiêng về Random Forest, [7] và [14] nghiêng về XGBoost. Khác biệt thường nhỏ so với biến động do giao thức.
- **Đa dạng mô hình có đủ không:** [20] cho thấy mô hình đa dạng nhất không được chọn; đa dạng chỉ có ích khi đi kèm năng lực dự đoán.

## 6. Kết luận

### 6.1. Các phát hiện chính

1. **Mô hình cây tăng cường là mốc chuẩn khó vượt:** trên dữ liệu bảng, mọi phương pháp mới cần so sánh với XGBoost hoặc LightGBM đã tinh chỉnh [7], [17], [20].
2. **Giao thức đánh giá quyết định độ tin cậy:** SMOTE trước khi chia tập nâng F1 từ khoảng 93% lên 99,97% [7]; giảm mẫu 1:1 cho kết quả tốt trên test cân bằng nhưng Random Forest chỉ đạt precision 0,0831 trên test giữ phân phối gốc [15].
3. **Accuracy và ROC-AUC không đủ:** accuracy 99,77% có thể đi cùng recall 0,39 [8]; F1 có trọng số 0,97 có thể đi cùng F1 lớp gian lận 0,53 [17]. AUPRC, precision và recall theo ngân sách cảnh báo phản ánh tốt hơn [5], [20].
4. **Không có phương pháp tái cân bằng tốt nhất phổ quát:** hiệu quả phụ thuộc vào mô hình, tỷ lệ mục tiêu và dữ liệu. Mức tái cân bằng nhẹ, giữ dữ liệu vẫn mất cân bằng, có cơ sở tốt hơn cân bằng 1:1 [15].
5. **Mức cải thiện đáng tin thường khiêm tốn:** bài có kiểm định thống kê ghi nhận mức tăng khoảng 2–3 điểm AUPRC và F1 [20].

### 6.2. Vấn đề còn tồn tại

- **Thiếu chia theo thời gian:** hầu hết bài chia ngẫu nhiên, không phản ánh việc mô hình dự đoán tương lai; biến đổi khái niệm được nhắc đến nhiều nhưng ít được kiểm chứng [10], [13], [20].
- **Phụ thuộc vào ULB:** 9 trên 21 bài dùng bộ dữ liệu từ 2013 với đặc trưng PCA ẩn danh, chỉ khoảng 49–123 mẫu gian lận trong tập kiểm tra, không thể diễn giải nghiệp vụ [9], [19].
- **Thiếu kiểm định thống kê:** hầu hết bài không có khoảng tin cậy; trên tập kiểm tra nhỏ, chênh lệch một vài mẫu có thể tạo ra thứ hạng khác nhau [14].
- **Sai sót phương pháp luận phổ biến:** chọn cấu hình trên tập kiểm tra [15], [18]; đánh giá trên dữ liệu đã cân bằng [9], [21] hoặc có dấu hiệu như vậy [1], [11]; kết luận không khớp bảng số liệu [4], [8], [12], [13], [19].
- **Chưa gắn với chi phí thực tế:** rất ít bài đánh giá theo chi phí báo động giả, ngân sách điều tra hoặc giá trị tiền bị gian lận [17], [20].
- **Thiếu dữ liệu tiếng Việt và bối cảnh Việt Nam:** các bài tiếng Việt đều dùng dữ liệu công khai quốc tế; [8] nêu rõ hạn chế không tiếp cận được dữ liệu thực tại Việt Nam.
- **Khả năng tái lập hạn chế:** dữ liệu riêng [5], [11] và thiếu mã nguồn khiến so sánh phải dựa vào bản tái lập [14].

### 6.3. Hướng nghiên cứu tiếp theo

- **Khung đánh giá chuẩn:** chia theo thời gian, giữ nguyên tập kiểm tra, chỉ tái cân bằng tập huấn luyện trong từng fold, chọn cấu hình trên tập kiểm định, báo cáo AUPRC, precision và recall tại ngân sách cảnh báo, kèm khoảng tin cậy bootstrap.
- **So sánh trực tiếp các hướng trong cùng giao thức:** đánh giá SMOTE lai, trọng số lớp, mô hình sinh và tổ hợp trên nhiều bộ dữ liệu, với XGBoost hoặc LightGBM làm mốc.
- **Mô hình sinh theo ngữ cảnh:** tiếp tục hướng huấn luyện có thông tin của cả hai lớp [14], xử lý đúng biến phân loại lực lượng lớn [20], và kiểm tra đồng thời độ trung thực và hiệu năng [21].
- **Kết hợp đồ thị với mô hình cây:** dùng đặc trưng đồ thị hoặc điểm bất thường làm đầu vào cho mô hình tăng cường, khai thác cả quan hệ và độ ổn định trên dữ liệu bảng.
- **Tối ưu theo chi phí và diễn giải được:** đưa chi phí nghiệp vụ vào hàm mục tiêu và ngưỡng quyết định, kết hợp SHAP hoặc EBM [11], [19].
- **Học thích nghi:** huấn luyện lại định kỳ và theo dõi biến đổi khái niệm, đánh giá trên dòng dữ liệu theo thời gian [13], [18], [19].

## Tài liệu tham khảo

[1] A. H. M. Aburbeian, H. I. Ashqar, “Credit Card Fraud Detection Using Enhanced Random Forest Classifier for Imbalanced Data,” arXiv:2303.06514, 2023.

[2] X. Hu, H. Chen, H. Chen, S. Liu, X. Li, S. Zhang, Y. Wang, X. Xue, “Cost Sensitive GNN-based Imbalanced Learning for Mobile Social Network Fraud Detection,” arXiv:2303.17486, 2023.

[3] Nguyễn Thị Phương Nga, Nguyễn Thị Lan Anh, “Nâng cao hiệu suất bài toán phân lớp dữ liệu mất cân bằng với BLSMOTE-LOF,” Cơ sở dữ liệu khoa học Đại học Huế, 2023.

[4] Trần Đức Thuận, Nguyễn Đình Thuân, “Phát hiện gian lận trong thẻ tín dụng bằng cách sử dụng học máy và logic mờ,” Kỷ yếu Hội nghị FAIR lần thứ XVI, DOI 10.15625/vap.2023.0041, 2023.

[5] G. Velarde, A. Sudhir, S. Deshmane, A. Deshmunkh, K. Sharma, V. Joshi, “Evaluating XGBoost for Balanced and Imbalanced Data: Application to Fraud Detection,” arXiv:2303.15218, 2023.

[6] S. Wang, T. Tricco, X. Jiang, C. Robertson, J. Hawkin, “Synthetic Demographic Data Generation for Card Fraud Detection Using GANs,” arXiv:2306.17109, 2023.

[7] S. Kabane, “Impact of Sampling Techniques and Data Leakage on XGBoost Performance in Credit Card Fraud Detection,” arXiv:2412.07437, 2024.

[8] Ngô Thuỳ Linh, Nguyễn Dương Hùng, “Phát hiện gian lận thẻ tín dụng sử dụng mô hình học sâu Autoencoder kết hợp thuật toán Isolation forest,” Tạp chí Kinh tế - Luật & Ngân hàng, số 266, 2024.

[9] Nguyễn Minh Nhật, Đào Lê Kiều Oanh, “Mức độ hiệu quả của các mô hình học máy tree-based trong phát hiện giao dịch gian lận thẻ tín dụng,” Tạp chí Khoa học Thương mại, số 195, 2024.

[10] S. Verma, J. Dhar, “Credit Card Fraud Detection: A Deep Learning Approach,” arXiv:2409.13406, 2024.

[11] F. Boabang, S. A. Gyamerah, “An Enhanced Focal Loss Function to Mitigate Class Imbalance in Auto Insurance Fraud Detection with Explainable AI,” arXiv:2508.02283, 2025.

[12] K. U. Emaan, “Improving Credit Card Fraud Detection through Transformer-Enhanced GAN Oversampling,” arXiv:2509.19032, 2025.

[13] D. Farahmandazad, K. Danesh, “ML-Driven Approaches to Combat Medicare Fraud: Advances in Class Imbalance Solutions, Feature Engineering, Adaptive Learning, and Business Impact,” arXiv:2502.15898, 2025.

[14] C. Giusti, L. Guarnera, M. Casu, S. Battiato, “Fraud is Not Just Rarity: A Causal Prototype Attention Approach to Realistic Synthetic Oversampling,” arXiv:2507.14706, 2025.

[15] I. Popova, H. A. A. Gardi, “Credit Card Fraud Detection,” arXiv:2509.15044, 2025.

[16] Y. Song, Y. Wei, Y. Lu, Q. Sun, M. Shao, L. Wang, C. Hu, X. Li, X. Fu, “Mitigating Message Imbalance in Fraud Detection with Dual-View Graph Representation Learning,” arXiv:2507.06469, 2025.

[17] C. Wang, C. Nie, Y. Liu, “Evaluating Supervised Learning Models for Fraud Detection: A Comparative Study of Classical and Deep Architectures on Imbalanced Transaction Data,” arXiv:2505.22521, 2025.

[18] R. Batsyas, R. Yaduwanshi, “Fraud Detection System for Banking Transactions,” arXiv:2604.07952, 2026.

[19] R. E. Fazel, A. Bakhtiary, S. A. Bigdeli, “Improving Credit Card Fraud Detection with an Optimized Explainable Boosting Machine,” arXiv:2602.06955, 2026.

[20] X. Han, C. Wu, “Validation-Stage Combinatorial Fusion Analysis for Imbalanced Credit-Card Fraud Detection,” arXiv:2606.10393, 2026.

[21] A. Innan, M. El Alami, N. Innan, M. Shafique, M. Bennai, “Q-SYNTH: Hybrid Quantum-Classical Adversarial Augmentation for Imbalanced Fraud Detection,” arXiv:2605.21164, 2026.
