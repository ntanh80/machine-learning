from sklearn.feature_extraction.text import CountVectorizer

# Dữ liệu viết thường, không có dấu câu để dễ tách bằng split().
van_ban = [
    "tôi thích học máy",
    "tôi thích python",
    "học máy học python",
]

# Cách 1: Python thuần
# Tách theo khoảng trắng (tiếng Việt có thể tách thành các âm tiết).
cac_cau = [cau.split() for cau in van_ban]
tu_dien = sorted(set(" ".join(van_ban).split()))

ma_tran = []
for cau in cac_cau:
    hang = []
    for tu in tu_dien:
        hang.append(cau.count(tu))
    ma_tran.append(hang)

print("Cách 1: Python thuần")
print("Từ điển:", tu_dien)
for hang in ma_tran:
    print(hang)

# Cách 2: Scikit-learn
bo_dem = CountVectorizer()
ma_tran_sklearn = bo_dem.fit_transform(van_ban).toarray()

print("Từ điển:", bo_dem.get_feature_names_out().tolist())
print(ma_tran_sklearn)

# Mỗi hàng là một câu, mỗi cột là số lần xuất hiện của một từ.

# Cách 3: TF-IDF (trọng số từ thay cho số lần xuất hiện)
# Từ có trong ít câu được tăng trọng số; mỗi vector được chuẩn hóa L2.
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
ma_tran_tfidf = tfidf.fit_transform(van_ban).toarray()

print("\nCách 3: TF-IDF")
print("Từ điển:", tfidf.get_feature_names_out().tolist())
print(ma_tran_tfidf.round(3))

# Cách 4: Hashing - băm từ vào vector cố định 8 chiều.
# Có thể có nhiều từ chung một cột; đây là biểu diễn bằng băm,
# không phải embedding học ngữ nghĩa.
from sklearn.feature_extraction.text import HashingVectorizer

hashing = HashingVectorizer(n_features=8, alternate_sign=False)
ma_tran_hash = hashing.transform(van_ban).toarray()
print("\nCách 4: Hashing (8 chiều)")
print(ma_tran_hash.round(3))

# Cách 5: LSA - dùng SVD nén TF-IDF thành embedding 2 chiều.
from sklearn.decomposition import TruncatedSVD

lsa = TruncatedSVD(n_components=2, random_state=42)
embedding_lsa = lsa.fit_transform(ma_tran_tfidf)
print("\nCách 5: LSA/SVD (2 chiều)")
print(embedding_lsa.round(3))

# Cách 6: NMF - biểu diễn mỗi câu bằng 2 trọng số chủ đề không âm.
from sklearn.decomposition import NMF

nmf = NMF(n_components=2, init="nndsvda", random_state=42, max_iter=500)
embedding_nmf = nmf.fit_transform(ma_tran_tfidf)
print("\nCách 6: NMF (2 chiều)")
print(embedding_nmf.round(3))

# Mỗi hàng vẫn là một câu; cột LSA/NMF là đặc trưng ẩn, không phải từ.
# Chỉ có 3 câu nên đây là minh họa cách chạy, chưa học ngữ nghĩa tốt.
