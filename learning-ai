# 📚 Tài liệu ôn thi — Python · NumPy · Pandas
### Chuẩn bị cho kỳ thi Vingroup AI thực chiến

---

## MỤC LỤC

1. [Python cơ bản](#1-python-cơ-bản)
2. [NumPy](#2-numpy)
3. [Pandas](#3-pandas)
4. [SQL cơ bản (bonus)](#4-sql-cơ-bản-bonus)

---

## 1. PYTHON CƠ BẢN

### 1.1 Kiểu dữ liệu & cấu trúc dữ liệu

#### List
```python
a = [1, 2, 3, 4, 5]

a[0]        # 1  — lấy phần tử đầu
a[-1]       # 5  — lấy phần tử cuối
a[1:3]      # [2, 3]  — slice từ index 1 đến 2 (không gồm 3)
a[::-1]     # [5, 4, 3, 2, 1]  — đảo ngược
a.append(6) # thêm vào cuối
a.pop()     # xóa và trả về phần tử cuối
len(a)      # độ dài
```

#### Dictionary
```python
d = {"name": "An", "age": 22}

d["name"]           # "An"
d.get("score", 0)   # 0 — trả về 0 nếu key không tồn tại
d.keys()            # dict_keys(['name', 'age'])
d.values()          # dict_values(['An', 22])
d.items()           # dict_items([('name', 'An'), ('age', 22)])
d["score"] = 95     # thêm key mới
```

#### Set & Tuple
```python
s = {1, 2, 3, 2}   # {1, 2, 3} — tự loại bỏ trùng
t = (1, 2, 3)       # tuple — không thay đổi được (immutable)
```

---

### 1.2 Vòng lặp & điều kiện

```python
# for với enumerate (lấy cả index và giá trị)
for i, v in enumerate(["a", "b", "c"]):
    print(i, v)     # 0 a, 1 b, 2 c

# zip — ghép 2 danh sách
for x, y in zip([1, 2, 3], ["a", "b", "c"]):
    print(x, y)     # 1 a, 2 b, 3 c

# List comprehension — cực kỳ phổ biến trong đề thi
squares = [x**2 for x in range(5)]        # [0, 1, 4, 9, 16]
evens   = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]

# Dict comprehension
d = {k: k**2 for k in range(5)}           # {0:0, 1:1, 2:4, 3:9, 4:16}
```

---

### 1.3 Hàm

```python
# Hàm bình thường
def add(a, b=0):
    return a + b

add(3)      # 3
add(3, 4)   # 7

# Lambda — hàm ẩn danh (thường dùng với map/filter/sorted)
square = lambda x: x**2
square(5)   # 25

# map — áp dụng hàm lên từng phần tử
list(map(lambda x: x*2, [1, 2, 3]))   # [2, 4, 6]

# filter — lọc phần tử thỏa điều kiện
list(filter(lambda x: x > 2, [1, 2, 3, 4]))  # [3, 4]

# sorted với key
data = [{"name": "B", "score": 80}, {"name": "A", "score": 90}]
sorted(data, key=lambda x: x["score"], reverse=True)
# [{"name": "A", "score": 90}, {"name": "B", "score": 80}]
```

---

### 1.4 Các hàm built-in hay gặp trong đề

| Hàm | Mô tả | Ví dụ |
|-----|-------|-------|
| `sum(iterable)` | Tổng | `sum([1,2,3])` → 6 |
| `max(iterable)` | Giá trị lớn nhất | `max([1,5,3])` → 5 |
| `min(iterable)` | Giá trị nhỏ nhất | `min([1,5,3])` → 1 |
| `abs(x)` | Giá trị tuyệt đối | `abs(-5)` → 5 |
| `round(x, n)` | Làm tròn | `round(3.14159, 2)` → 3.14 |
| `range(start, stop, step)` | Dãy số | `list(range(0,10,2))` → [0,2,4,6,8] |
| `type(x)` | Kiểu dữ liệu | `type(3.14)` → float |
| `isinstance(x, type)` | Kiểm tra kiểu | `isinstance(3, int)` → True |
| `str(x)` / `int(x)` / `float(x)` | Ép kiểu | `int("42")` → 42 |

---

### 1.5 String hay gặp

```python
s = "  Hello, World!  "

s.strip()           # "Hello, World!" — bỏ khoảng trắng 2 đầu
s.lower()           # "  hello, world!  "
s.upper()           # "  HELLO, WORLD!  "
s.replace("o", "0") # "  Hell0, W0rld!  "
s.split(",")        # ['  Hello', ' World!  ']
",".join(["a","b"]) # "a,b"
s.startswith("  H") # True
"World" in s        # True
len(s)              # 18
s[7:12]             # "World"
f"Hello {42}"       # "Hello 42" — f-string
```

---

## 2. NUMPY

```python
import numpy as np
```

### 2.1 Tạo mảng

```python
np.array([1, 2, 3])         # array([1, 2, 3])
np.zeros((3, 4))            # ma trận 3x4 toàn số 0
np.ones((2, 3))             # ma trận 2x3 toàn số 1
np.eye(3)                   # ma trận đơn vị 3x3
np.arange(0, 10, 2)         # array([0, 2, 4, 6, 8])
np.linspace(0, 1, 5)        # array([0, .25, .5, .75, 1.]) — 5 điểm đều
np.random.rand(3, 3)        # ma trận ngẫu nhiên [0,1)
np.random.randn(3, 3)       # ma trận ngẫu nhiên phân phối chuẩn
np.random.randint(0, 10, (3,3))  # số nguyên ngẫu nhiên
np.random.seed(42)          # cố định seed để tái lập kết quả
```

---

### 2.2 Thuộc tính mảng

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

a.shape     # (2, 3) — 2 hàng, 3 cột
a.ndim      # 2 — số chiều
a.size      # 6 — tổng số phần tử
a.dtype     # dtype('int64') — kiểu dữ liệu
a.T         # chuyển vị — shape (3, 2)
```

---

### 2.3 Indexing & Slicing

```python
a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

a[0]        # array([1, 2, 3]) — hàng đầu
a[1, 2]     # 6 — hàng 1, cột 2
a[:, 1]     # array([2, 5, 8]) — toàn bộ cột 1
a[0:2, 1:]  # array([[2,3],[5,6]]) — hàng 0-1, cột 1 trở đi

# Boolean indexing — lọc theo điều kiện
a[a > 5]    # array([6, 7, 8, 9])
a[a % 2 == 0]  # array([2, 4, 6, 8]) — số chẵn
```

---

### 2.4 Biến đổi hình dạng

```python
a = np.arange(12)       # array([0, 1, 2, ..., 11])

a.reshape(3, 4)         # ma trận 3x4
a.reshape(2, -1)        # 2 hàng, tự tính cột (= 6)
a.flatten()             # làm phẳng thành 1D
a.ravel()               # tương tự flatten nhưng trả về view (tiết kiệm bộ nhớ)

# Thêm chiều
a = np.array([1, 2, 3])
a[:, np.newaxis]        # shape (3,1) — cột vector
a[np.newaxis, :]        # shape (1,3) — hàng vector
```

---

### 2.5 Phép toán

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Phép toán element-wise (từng phần tử)
a + b       # array([5, 7, 9])
a * b       # array([4, 10, 18])
a ** 2      # array([1, 4, 9])
a / b       # array([0.25, 0.4, 0.5])

# Broadcasting — mở rộng tự động
a + 10      # array([11, 12, 13])
a * 2       # array([2, 4, 6])

# Nhân ma trận
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
A @ B       # nhân ma trận: array([[19,22],[43,50]])
np.dot(A, B)  # tương tự A @ B
```

---

### 2.6 Hàm thống kê

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

np.sum(a)           # 21 — tổng tất cả
np.sum(a, axis=0)   # array([5, 7, 9]) — tổng theo cột
np.sum(a, axis=1)   # array([6, 15]) — tổng theo hàng

np.mean(a)          # 3.5
np.mean(a, axis=0)  # array([2.5, 3.5, 4.5])

np.std(a)           # độ lệch chuẩn
np.var(a)           # phương sai
np.min(a)           # 1
np.max(a)           # 6
np.argmin(a)        # 0 — index của giá trị nhỏ nhất (flatten)
np.argmax(a)        # 5 — index của giá trị lớn nhất (flatten)
np.cumsum(a)        # array([ 1,  3,  6, 10, 15, 21]) — tổng tích lũy
np.median(a)        # 3.5
np.percentile(a, 75) # phân vị thứ 75
```

---

### 2.7 Hàm toán học

```python
np.sqrt(a)          # căn bậc hai
np.exp(a)           # e^x
np.log(a)           # ln(x)
np.log2(a)          # log2(x)
np.abs(a)           # giá trị tuyệt đối
np.clip(a, 2, 5)    # giới hạn giá trị trong [2, 5]
np.where(a > 3, 1, 0)  # if a>3 then 1 else 0
```

---

### 2.8 Ghép và chia mảng

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

np.concatenate([a, b], axis=0)   # ghép theo hàng (stack dọc): shape (4,2)
np.concatenate([a, b], axis=1)   # ghép theo cột (stack ngang): shape (2,4)
np.vstack([a, b])   # tương tự axis=0
np.hstack([a, b])   # tương tự axis=1

np.split(a, 2, axis=0)  # chia thành 2 theo hàng
```

---

### 2.9 Đại số tuyến tính (hay ra trong đề AI)

```python
A = np.array([[1, 2], [3, 4]])

np.linalg.det(A)        # định thức: -2.0
np.linalg.inv(A)        # ma trận nghịch đảo
np.linalg.norm(A)       # chuẩn (norm) của ma trận
np.linalg.eig(A)        # trị riêng và vectơ riêng
np.linalg.svd(A)        # phân rã SVD: U, S, Vt
np.linalg.solve(A, b)   # giải hệ Ax = b
```

---

## 3. PANDAS

```python
import pandas as pd
```

### 3.1 Tạo DataFrame & Series

```python
# Series — mảng 1 chiều có nhãn
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
# a    10
# b    20
# c    30

# DataFrame — bảng dữ liệu 2 chiều
df = pd.DataFrame({
    "name":  ["An", "Bình", "Chi", "Dũng"],
    "age":   [22, 25, 23, 24],
    "score": [85, 92, 78, 88],
    "city":  ["HN", "HCM", "HN", "HCM"]
})
```

---

### 3.2 Xem nhanh dữ liệu

```python
df.head(3)       # 3 hàng đầu
df.tail(2)       # 2 hàng cuối
df.shape         # (4, 4) — số hàng, số cột
df.info()        # kiểu dữ liệu và null count
df.describe()    # thống kê mô tả (mean, std, min, max, ...)
df.columns       # Index(['name', 'age', 'score', 'city'])
df.dtypes        # kiểu dữ liệu từng cột
df.isnull().sum()  # đếm giá trị null theo cột
df.nunique()     # số giá trị duy nhất theo cột
```

---

### 3.3 Chọn cột & hàng

```python
df["name"]           # chọn 1 cột → Series
df[["name", "age"]]  # chọn nhiều cột → DataFrame

# .loc — dùng nhãn (label)
df.loc[0]            # hàng có index = 0
df.loc[0:2]          # hàng index 0, 1, 2 (gồm cả 2)
df.loc[0, "name"]    # "An" — hàng 0, cột "name"
df.loc[:, "age"]     # toàn bộ cột age

# .iloc — dùng số thứ tự (integer position)
df.iloc[0]           # hàng đầu tiên
df.iloc[0:2]         # hàng 0, 1 (không gồm 2)
df.iloc[0, 1]        # hàng 0, cột 1 = 22
df.iloc[:, 1:3]      # tất cả hàng, cột 1 đến 2

# Lọc theo điều kiện (Boolean indexing)
df[df["score"] > 85]              # hàng có score > 85
df[df["city"] == "HN"]            # người ở Hà Nội
df[(df["age"] > 22) & (df["score"] > 80)]  # kết hợp nhiều điều kiện
df[df["city"].isin(["HN", "HCM"])]  # city thuộc danh sách
```

---

### 3.4 Thêm / xóa cột

```python
df["passed"] = df["score"] >= 80   # thêm cột mới
df["bonus"]  = df["score"] * 0.1   # tính từ cột khác

df.drop("bonus", axis=1, inplace=True)   # xóa cột
df.drop(0, axis=0, inplace=True)         # xóa hàng index 0
```

---

### 3.5 Xử lý giá trị null (NaN)

```python
df.isnull()             # True/False từng ô
df.isnull().sum()       # đếm null theo cột
df.notnull()            # ngược lại

df.dropna()             # xóa hàng có bất kỳ null nào
df.dropna(subset=["score"])  # xóa hàng có null ở cột score
df.fillna(0)            # điền null bằng 0
df["score"].fillna(df["score"].mean(), inplace=True)  # điền bằng trung bình
```

---

### 3.6 Sắp xếp

```python
df.sort_values("score")                    # tăng dần
df.sort_values("score", ascending=False)   # giảm dần
df.sort_values(["city", "score"], ascending=[True, False])  # nhiều cột
df.sort_index()          # sắp xếp theo index
```

---

### 3.7 Thống kê & tổng hợp

```python
df["score"].mean()      # trung bình
df["score"].median()    # trung vị
df["score"].std()       # độ lệch chuẩn
df["score"].max()       # max
df["score"].min()       # min
df["score"].sum()       # tổng
df["score"].count()     # số giá trị không null
df["score"].value_counts()  # tần suất từng giá trị

# Tổng hợp nhiều hàm cùng lúc
df["score"].agg(["mean", "std", "min", "max"])
```

---

### 3.8 GroupBy — nhóm và tổng hợp

```python
# Tổng hợp theo nhóm
df.groupby("city")["score"].mean()
# city
# HCM    90.0
# HN     81.5

# Nhiều phép tổng hợp cùng lúc
df.groupby("city")["score"].agg(["mean", "max", "count"])

# Nhóm theo nhiều cột
df.groupby(["city", "passed"])["score"].mean()

# .transform() — giữ nguyên shape, thêm kết quả về từng hàng
df["avg_city"] = df.groupby("city")["score"].transform("mean")
```

---

### 3.9 Apply — áp dụng hàm tùy chỉnh

```python
# Áp dụng lên Series (theo cột)
df["score"].apply(lambda x: "Giỏi" if x >= 85 else "Khá")

# Áp dụng lên DataFrame theo hàng (axis=1)
df.apply(lambda row: row["score"] * 1.1 if row["city"] == "HN" else row["score"], axis=1)

# Áp dụng lên toàn DataFrame (từng phần tử)
df[["age", "score"]].applymap(lambda x: x * 2)
```

---

### 3.10 Merge & Join — ghép bảng

```python
df1 = pd.DataFrame({"id": [1, 2, 3], "name": ["An", "Bình", "Chi"]})
df2 = pd.DataFrame({"id": [1, 2, 4], "score": [90, 85, 78]})

# inner join — chỉ lấy id chung (1, 2)
pd.merge(df1, df2, on="id", how="inner")

# left join — giữ tất cả df1, điền NaN nếu không có trong df2
pd.merge(df1, df2, on="id", how="left")

# right join — giữ tất cả df2
pd.merge(df1, df2, on="id", how="right")

# outer join — giữ tất cả
pd.merge(df1, df2, on="id", how="outer")
```

---

### 3.11 Concat — ghép theo chiều

```python
# Ghép theo hàng (stack dọc)
pd.concat([df1, df2_same_cols], axis=0, ignore_index=True)

# Ghép theo cột (stack ngang)
pd.concat([df1, df2], axis=1)
```

---

### 3.12 Pivot Table

```python
# Tạo bảng tổng hợp dạng pivot
df.pivot_table(
    values="score",       # cột cần tổng hợp
    index="city",         # hàng
    aggfunc="mean"        # phép tính
)

# Pivot table nhiều giá trị
df.pivot_table(values="score", index="city", columns="passed", aggfunc="mean")
```

---

### 3.13 Xử lý string trong Pandas

```python
df["name"].str.lower()          # chuyển thành chữ thường
df["name"].str.upper()          # chữ hoa
df["name"].str.strip()          # bỏ khoảng trắng
df["name"].str.contains("An")   # True/False nếu chứa chuỗi
df["name"].str.startswith("A")  # bắt đầu bằng "A"
df["name"].str.replace("i", "y")  # thay chuỗi
df["name"].str.len()            # độ dài
df["name"].str.split(",")       # tách chuỗi
```

---

### 3.14 Xử lý datetime

```python
df["date"] = pd.to_datetime(df["date"])  # ép kiểu sang datetime

df["date"].dt.year          # năm
df["date"].dt.month         # tháng
df["date"].dt.day           # ngày
df["date"].dt.dayofweek     # thứ trong tuần (0=Thứ 2)
df["date"].dt.hour          # giờ

# Tính khoảng cách ngày
(df["end"] - df["start"]).dt.days
```

---

### 3.15 Đọc/ghi file

```python
df = pd.read_csv("data.csv")               # đọc CSV
df = pd.read_csv("data.csv", sep=";")      # tách bằng dấu ;
df = pd.read_csv("data.csv", index_col=0)  # cột đầu làm index
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")

df.to_csv("output.csv", index=False)
df.to_excel("output.xlsx", index=False)
```

---

## 4. SQL CƠ BẢN (Bonus)

> SQL hay xuất hiện trong đề thi data nói chung. Biết SQL + Pandas giúp đọc code nhanh hơn.

### 4.1 Cấu trúc câu SELECT

```sql
SELECT cot1, cot2
FROM bang
WHERE dieu_kien
GROUP BY cot1
HAVING dieu_kien_nhom
ORDER BY cot1 DESC
LIMIT 10;
```

---

### 4.2 Các hàm thường gặp

```sql
-- Thống kê
SELECT COUNT(*), AVG(score), SUM(score), MAX(score), MIN(score)
FROM students;

-- Nhóm
SELECT city, AVG(score)
FROM students
GROUP BY city
HAVING AVG(score) > 80;

-- Sắp xếp
SELECT * FROM students ORDER BY score DESC LIMIT 5;
```

---

### 4.3 JOIN

```sql
-- INNER JOIN
SELECT s.name, g.score
FROM students s
INNER JOIN grades g ON s.id = g.student_id;

-- LEFT JOIN
SELECT s.name, g.score
FROM students s
LEFT JOIN grades g ON s.id = g.student_id;
```

---

### 4.4 Subquery & Window Functions

```sql
-- Subquery — chọn sinh viên có điểm cao hơn trung bình
SELECT * FROM students
WHERE score > (SELECT AVG(score) FROM students);

-- Window Function — xếp hạng trong từng nhóm city
SELECT name, city, score,
       RANK() OVER (PARTITION BY city ORDER BY score DESC) AS rank
FROM students;
```

---

## 📌 BẢNG SO SÁNH NHANH: Pandas vs SQL

| Thao tác | Pandas | SQL |
|----------|--------|-----|
| Lọc | `df[df["score"]>80]` | `WHERE score > 80` |
| Nhóm | `df.groupby("city")["score"].mean()` | `GROUP BY city` |
| Sắp xếp | `df.sort_values("score")` | `ORDER BY score` |
| Ghép bảng | `pd.merge(df1, df2, on="id")` | `JOIN ... ON` |
| Chọn cột | `df[["name","score"]]` | `SELECT name, score` |
| Số hàng | `df.shape[0]` | `COUNT(*)` |
| Giới hạn | `df.head(10)` | `LIMIT 10` |

---

## ⚠️ LƯU Ý KHI LÀM BÀI THI

1. **Đọc kỹ output** — đề thường cho code sẵn, yêu cầu viết kết quả ra. Tập chạy tay.
2. **Chú ý axis** — `axis=0` là theo hàng (dọc), `axis=1` là theo cột (ngang).
3. **Phân biệt `loc` vs `iloc`** — loc dùng nhãn, iloc dùng số thứ tự.
4. **Broadcasting NumPy** — mảng shape (3,) + (3,1) → kết quả shape (3,3).
5. **Inplace** — `df.sort_values(..., inplace=True)` mới thay đổi df gốc.
6. **NaN truyền nhiễm** — bất kỳ phép tính nào với NaN đều cho NaN.

---

*Tài liệu tổng hợp phục vụ ôn thi kỳ thi Vingroup AI thực chiến (Khóa 2 & 3 — 2026)*
