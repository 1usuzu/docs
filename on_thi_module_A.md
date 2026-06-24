# 📐 Module A — Nền tảng Toán học & Tư duy Định lượng
### Chuẩn bị kỳ thi Vingroup AI thực chiến

---

## MỤC LỤC

1. [Đại số tuyến tính](#1-đại-số-tuyến-tính-linear-algebra)
2. [Giải tích & Tối ưu hóa](#2-giải-tích--tối-ưu-hóa)
3. [Xác suất & Thống kê](#3-xác-suất--thống-kê)

---

## 1. Đại số tuyến tính (Linear Algebra)

### 1.1 Vector

```
Vector a = [a₁, a₂, ..., aₙ]  — mảng số có hướng trong không gian n chiều
```

**Các phép toán cơ bản:**

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Cộng / trừ vector (element-wise)
a + b           # [5, 7, 9]
a - b           # [-3, -3, -3]

# Nhân vô hướng (dot product)
np.dot(a, b)    # 1×4 + 2×5 + 3×6 = 32
a @ b           # tương tự dot product

# Chuẩn (norm) của vector
np.linalg.norm(a)         # L2 norm = √(1²+2²+3²) = √14 ≈ 3.742
np.linalg.norm(a, ord=1)  # L1 norm = |1|+|2|+|3| = 6
```

**Cosine similarity** — đo góc giữa 2 vector (dùng nhiều trong NLP):

```
cos(θ) = (a · b) / (‖a‖ × ‖b‖)

Kết quả ∈ [-1, 1]:
  1  → cùng hướng (giống nhau hoàn toàn)
  0  → vuông góc (không liên quan)
 -1  → ngược hướng (trái ngược nhau)
```

```python
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

---

### 1.2 Ma trận

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])   # ma trận 2×3 (2 hàng, 3 cột)

A.shape     # (2, 3)
A.T         # chuyển vị — shape (3, 2): hàng ↔ cột
```

**Nhân ma trận AB:** cột A = hàng B

```python
A = np.array([[1, 2], [3, 4]])  # 2×2
B = np.array([[5, 6], [7, 8]])  # 2×2

A @ B
# [1×5+2×7, 1×6+2×8]   = [19, 22]
# [3×5+4×7, 3×6+4×8]   = [43, 50]
# Kết quả: [[19, 22], [43, 50]]
```

> ⚠️ **Lưu ý:** AB ≠ BA (nhân ma trận không giao hoán). Tính chất: (AB)ᵀ = BᵀAᵀ

---

### 1.3 Định thức & Ma trận nghịch đảo

**Định thức:**

```
Ma trận 2×2:  det([[a, b], [c, d]]) = ad − bc

Ví dụ:  A = [[3, 2], [1, 4]]
         det(A) = 3×4 − 2×1 = 10
```

```python
A = np.array([[3, 2], [1, 4]])
np.linalg.det(A)    # 10.0
```

**Ma trận nghịch đảo:**

```
A⁻¹ tồn tại khi và chỉ khi det(A) ≠ 0
A × A⁻¹ = A⁻¹ × A = I  (ma trận đơn vị)
```

```python
np.linalg.inv(A)    # ma trận nghịch đảo

# Giải hệ phương trình Ax = b
b = np.array([5, 3])
x = np.linalg.solve(A, b)   # tốt hơn A⁻¹ @ b về mặt số học
```

---

### 1.4 Trị riêng & Vectơ riêng (Eigenvalue / Eigenvector)

**Định nghĩa:**

```
Av = λv

λ: trị riêng (eigenvalue)
v: vectơ riêng (eigenvector) tương ứng — không thay đổi hướng khi nhân với A
```

**Cách tìm:**

```
Bước 1: Giải  det(A − λI) = 0  → tìm λ  (phương trình đặc trưng)
Bước 2: Với mỗi λ, giải  (A − λI)v = 0  → tìm v
```

**Ví dụ:**

```
A = [[2, 1], [0, 3]]

det(A − λI) = det([[2−λ, 1], [0, 3−λ]])
            = (2−λ)(3−λ) − 0
            = λ² − 5λ + 6 = 0
→ λ₁ = 2, λ₂ = 3
```

```python
eigenvalues, eigenvectors = np.linalg.eig(A)
# eigenvalues:  [2., 3.]
# eigenvectors: mỗi CỘT là 1 eigenvector
```

**Ứng dụng trong AI:**
- **PCA:** chọn các eigenvector ứng với eigenvalue lớn nhất → giảm chiều dữ liệu
- **Eigenvalue lớn** = hướng có nhiều variance nhất trong dữ liệu

---

### 1.5 Phân rã SVD (Singular Value Decomposition)

```
A = U × Σ × Vᵀ

U:  ma trận trực giao m×m  (left singular vectors)
Σ:  ma trận đường chéo m×n (singular values ≥ 0, sắp giảm dần)
Vᵀ: ma trận trực giao n×n  (right singular vectors)
```

```python
U, S, Vt = np.linalg.svd(A)
# S: vector chứa singular values (đường chéo của Σ)
```

**Xấp xỉ hạng thấp (Low-rank approximation):**

```python
k = 2  # giữ lại k singular values lớn nhất
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
# Giữ được phần lớn thông tin, giảm nhiễu
```

**Ứng dụng:** giảm chiều, hệ gợi ý (recommendation), NLP (LSA), nén ảnh

---

### 1.6 Chuẩn (Norm) — Bảng tổng kết

| Chuẩn | Công thức | Ý nghĩa & Ứng dụng |
|-------|-----------|---------------------|
| **L1 (Manhattan)** | ‖x‖₁ = Σ\|xᵢ\| | Ít nhạy outlier. Dùng trong Lasso (nhiều w → 0) |
| **L2 (Euclidean)** | ‖x‖₂ = √(Σxᵢ²) | Khoảng cách thông thường. Dùng trong Ridge |
| **L∞ (Chebyshev)** | ‖x‖∞ = max\|xᵢ\| | Giá trị lớn nhất trong vector |
| **Frobenius** | ‖A‖F = √(ΣΣaᵢⱼ²) | Norm của ma trận |
| **Cosine** | a·b / (‖a‖·‖b‖) | Đo góc giữa 2 vector, dùng NLP |

---

## 2. Giải tích & Tối ưu hóa

### 2.1 Đạo hàm cơ bản

```
f'(x) = lim[h→0] (f(x+h) − f(x)) / h

Ý nghĩa: tốc độ thay đổi của f tại điểm x, độ dốc đường tiếp tuyến
```

**Bảng đạo hàm hay gặp:**

| Hàm f(x) | Đạo hàm f'(x) | Ghi nhớ |
|----------|----------------|---------|
| `c` (hằng số) | `0` | Hằng số không đổi |
| `xⁿ` | `n·xⁿ⁻¹` | Hạ mũ, giảm 1 |
| `eˣ` | `eˣ` | Đặc biệt: tự đạo hàm |
| `aˣ` | `aˣ · ln(a)` | |
| `ln(x)` | `1/x` | |
| `log_a(x)` | `1/(x·ln(a))` | |
| `sin(x)` | `cos(x)` | |
| `cos(x)` | `−sin(x)` | Chú ý dấu âm |
| `tan(x)` | `1/cos²(x)` | |

**Các quy tắc đạo hàm:**

```
Chain Rule (quy tắc dây chuyền):
  [f(g(x))]' = f'(g(x)) · g'(x)

  Ví dụ: [sin(x²)]' = cos(x²) · 2x

Product Rule (quy tắc tích):
  [f(x)·g(x)]' = f'(x)·g(x) + f(x)·g'(x)

  Ví dụ: [x²·sin(x)]' = 2x·sin(x) + x²·cos(x)

Quotient Rule (quy tắc thương):
  [f(x)/g(x)]' = (f'g − fg') / g²
```

---

### 2.2 Chain Rule trong Backpropagation

**Chain Rule nhiều biến:**

```
Nếu L = f(g(h(x))):

dL/dx = (∂L/∂f) × (∂f/∂g) × (∂g/∂h) × (∂h/∂x)
```

**Ví dụ mạng nơ-ron 2 lớp:**

```
Input x → [Linear: z₁ = w₁x + b₁] → [ReLU: a₁ = max(0, z₁)]
         → [Linear: z₂ = w₂a₁ + b₂] → [Sigmoid: ŷ = σ(z₂)] → Loss L

Backprop:
  ∂L/∂w₂ = (∂L/∂ŷ) × (∂ŷ/∂z₂)              ← lớp cuối
  ∂L/∂w₁ = (∂L/∂ŷ) × (∂ŷ/∂z₂) × (∂z₂/∂a₁) × (∂a₁/∂z₁) × (∂z₁/∂w₁)
```

> 💡 **Ý nghĩa:** Backprop chính là chain rule áp dụng ngược từ output về input qua từng lớp.

---

### 2.3 Đạo hàm riêng & Gradient

**Đạo hàm riêng:** giữ các biến khác cố định, đạo hàm theo 1 biến

```
f(x, y) = x² + 3xy + y²

∂f/∂x = 2x + 3y   (coi y là hằng số)
∂f/∂y = 3x + 2y   (coi x là hằng số)
```

**Gradient:** vector gồm tất cả đạo hàm riêng

```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

Hướng của ∇f = hướng tăng nhanh nhất của f
−∇f           = hướng giảm nhanh nhất  ← đây là hướng ta muốn đi khi minimize!
```

---

### 2.4 Gradient Descent — Thuật toán tối ưu cốt lõi

**Công thức cập nhật:**

```
w ← w − η × ∂L/∂w

η (eta): learning rate — bước nhảy mỗi lần cập nhật
∂L/∂w:  gradient của loss theo trọng số w
```

**Ví dụ trực quan:**

```
f(w) = w²  →  f'(w) = 2w

w₀ = 4, η = 0.1

Bước 1: w₁ = 4 − 0.1 × (2×4) = 4 − 0.8 = 3.2
Bước 2: w₂ = 3.2 − 0.1 × (2×3.2) = 3.2 − 0.64 = 2.56
...     dần dần hội tụ về w = 0  (điểm minimum của f(w) = w²)
```

**So sánh các biến thể:**

| Biến thể | Data mỗi bước | Ưu điểm | Nhược điểm |
|----------|--------------|---------|------------|
| **Batch GD** | Toàn bộ dataset | Ổn định, hội tụ đảm bảo | Chậm với data lớn |
| **SGD** | 1 sample | Nhanh, thoát local min | Nhiễu, không ổn định |
| **Mini-batch GD** | 32–256 samples | Cân bằng tốc độ & ổn định | Cần chọn batch size |
| **Adam** | Mini-batch | Hội tụ nhanh, ít tune lr | Tốn bộ nhớ hơn SGD |
| **RMSprop** | Mini-batch | Tốt cho RNN, non-stationary | - |

**Vấn đề của learning rate:**

```
η quá lớn → không hội tụ, dao động  (diverge)
η quá nhỏ → hội tụ rất chậm
η vừa     → hội tụ nhanh, ổn định  ✓
```

---

### 2.5 Hàm kích hoạt — Công thức & Đạo hàm

| Hàm | Công thức | Đạo hàm | Đặc điểm |
|-----|-----------|---------|----------|
| **Sigmoid** | σ(x) = 1/(1+e⁻ˣ) | σ(x)·(1−σ(x)) | Output ∈ (0,1). Vanishing gradient khi x lớn/nhỏ. |
| **Tanh** | (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | 1 − tanh²(x) | Output ∈ (−1,1). Tốt hơn sigmoid nhưng vẫn vanish. |
| **ReLU** | max(0, x) | 0 nếu x<0; 1 nếu x>0 | Đơn giản, tính nhanh. Dying ReLU khi x<0 mãi. |
| **Leaky ReLU** | max(αx, x) với α nhỏ | α nếu x<0; 1 nếu x>0 | Sửa Dying ReLU. α thường = 0.01. |
| **Softmax** | eˣⁱ / Σeˣʲ | Phức tạp — dùng với cross-entropy | Output là phân phối xác suất, tổng = 1. |

**Ví dụ Sigmoid:**

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

sigmoid(0)    # 0.5
sigmoid(2)    # 0.880
sigmoid(-2)   # 0.119
sigmoid(100)  # ≈ 1.0  ← saturate
```

---

### 2.6 Tích phân

**Tích phân bất định:**

```
∫xⁿ dx = xⁿ⁺¹/(n+1) + C
∫eˣ dx = eˣ + C
∫(1/x) dx = ln|x| + C
∫sin(x) dx = −cos(x) + C
∫cos(x) dx = sin(x) + C
```

**Tích phân xác định — diện tích dưới đường cong:**

```
∫[a,b] f(x)dx = F(b) − F(a)   (F là nguyên hàm của f)

Ví dụ: ∫[0,2] x² dx = [x³/3]₀² = 8/3 − 0 = 8/3 ≈ 2.667
```

**Ứng dụng xác suất:**

```
Nếu X là biến ngẫu nhiên liên tục với hàm mật độ f(x):
P(a ≤ X ≤ b) = ∫[a,b] f(x)dx

Điều kiện: ∫(-∞,+∞) f(x)dx = 1  (tổng xác suất = 1)
```

---

### 2.7 Overfitting & Regularization

**Bias-Variance Tradeoff:**

```
Total Error = Bias² + Variance + Noise

Bias cao (Underfitting):  mô hình quá đơn giản, không học được pattern
Variance cao (Overfitting): học thuộc train data, kém trên data mới
```

**Regularization — thêm penalty vào loss function:**

```
L1 (Lasso):   L_total = L + λ·Σ|wᵢ|
  → Nhiều trọng số → 0 (feature selection tự động)
  → Tạo ra mô hình sparse (thưa)

L2 (Ridge):   L_total = L + λ·Σwᵢ²
  → Giữ trọng số nhỏ, ít loại bỏ feature
  → Ổn định hơn L1

Elastic Net:  L_total = L + λ₁·Σ|wᵢ| + λ₂·Σwᵢ²
  → Kết hợp cả L1 và L2
```

```python
# Ví dụ với sklearn
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0)    # alpha = λ (hệ số regularization)
lasso = Lasso(alpha=0.1)
```

---

## 3. Xác suất & Thống kê

### 3.1 Xác suất cơ bản

```
P(A) ∈ [0, 1]              — xác suất luôn không âm và ≤ 1
P(Ω) = 1                   — toàn bộ không gian mẫu
P(A') = 1 − P(A)           — xác suất bổ sung

P(A ∪ B) = P(A) + P(B) − P(A ∩ B)    — phép hợp

Độc lập:  P(A ∩ B) = P(A) × P(B)

Xác suất có điều kiện:  P(A|B) = P(A ∩ B) / P(B)
```

---

### 3.2 Định lý Bayes

```
P(A|B) = P(B|A) × P(A) / P(B)
```

| Thành phần | Tên | Ý nghĩa |
|------------|-----|---------|
| P(A) | **Prior** | Xác suất tiên nghiệm — trước khi quan sát B |
| P(B\|A) | **Likelihood** | Xác suất quan sát B nếu A đúng |
| P(B) | **Evidence** | Xác suất quan sát B (tổng hóa) |
| P(A\|B) | **Posterior** | Xác suất hậu nghiệm — sau khi quan sát B |

**Ví dụ thực tế:**

```
Bệnh X xuất hiện ở 1% dân số. Test dương tính nếu có bệnh: 99%. Test dương tính nhầm: 5%.
Hỏi: Test dương tính thì xác suất thực sự có bệnh là bao nhiêu?

P(Bệnh) = 0.01,  P(+|Bệnh) = 0.99,  P(+|KhôngBệnh) = 0.05

P(+) = P(+|Bệnh)×P(Bệnh) + P(+|KhôngBệnh)×P(KhôngBệnh)
     = 0.99×0.01 + 0.05×0.99 = 0.0594

P(Bệnh|+) = P(+|Bệnh) × P(Bệnh) / P(+)
           = 0.99 × 0.01 / 0.0594 ≈ 0.167  →  chỉ ~16.7%!
```

---

### 3.3 Các phân phối xác suất quan trọng

#### Bernoulli & Binomial

```
Bernoulli(p):  X ∈ {0, 1}
  P(X=1) = p,  P(X=0) = 1−p
  E[X] = p,    Var(X) = p(1−p)

Binomial(n, p):  X = số lần thành công trong n phép thử
  P(X=k) = C(n,k) × pᵏ × (1−p)ⁿ⁻ᵏ
  E[X] = np,   Var(X) = np(1−p)
```

#### Phân phối chuẩn (Normal / Gaussian)

```
X ~ N(μ, σ²)

f(x) = (1/√(2πσ²)) × exp(−(x−μ)²/(2σ²))

μ: mean (trung bình) — tâm của hình chuông
σ: std dev          — độ rộng của hình chuông
```

**Quy tắc 68-95-99.7:**

```
68.27%  dữ liệu nằm trong  [μ − 1σ,  μ + 1σ]
95.45%  dữ liệu nằm trong  [μ − 2σ,  μ + 2σ]
99.73%  dữ liệu nằm trong  [μ − 3σ,  μ + 3σ]
```

```python
from scipy import stats

# Xác suất P(-1 ≤ X ≤ 1) với X ~ N(0,1)
stats.norm.cdf(1) - stats.norm.cdf(-1)   # ≈ 0.6827

# Z-score: chuẩn hóa về N(0,1)
z = (x - mu) / sigma
```

#### Phân phối Poisson

```
X ~ Poisson(λ)  — đếm sự kiện trong khoảng thời gian/không gian

P(X=k) = λᵏ × e⁻λ / k!
E[X] = λ,   Var(X) = λ

Ví dụ: Số email spam đến trong 1 giờ ~ Poisson(5)
```

---

### 3.4 Thống kê mô tả

```python
import numpy as np
data = [2, 4, 4, 4, 5, 5, 7, 9]

# Đo xu hướng trung tâm
mean   = np.mean(data)      # 5.0   — trung bình
median = np.median(data)    # 4.5   — trung vị (bền với outlier hơn)

# Đo độ phân tán
var = np.var(data)          # 4.0   — phương sai σ²
std = np.std(data)          # 2.0   — độ lệch chuẩn σ

# Đo mối quan hệ
np.corrcoef(x, y)           # correlation matrix ∈ [-1, 1]
np.cov(x, y)                # covariance matrix
```

**Bảng so sánh Mean vs Median:**

```
Data = [1, 2, 3, 4, 100]

Mean   = (1+2+3+4+100)/5 = 22   ← bị kéo bởi outlier 100
Median = 3                        ← không bị ảnh hưởng

→ Khi dữ liệu có outlier, Median phản ánh trung tâm tốt hơn
```

---

### 3.5 Covariance & Correlation

**Covariance:**

```
Cov(X, Y) = E[(X − μX)(Y − μY)]
           = (1/n) Σ (xᵢ − x̄)(yᵢ − ȳ)

Cov > 0: X và Y cùng tăng / cùng giảm
Cov < 0: X tăng thì Y giảm (ngược chiều)
Cov = 0: không có mối quan hệ tuyến tính
```

**Correlation (Pearson):**

```
ρ(X, Y) = Cov(X, Y) / (σX × σY)   ∈ [−1, 1]

|ρ| = 1.0: tương quan tuyến tính hoàn hảo
|ρ| > 0.7: tương quan mạnh
|ρ| ∈ [0.3, 0.7]: tương quan vừa
|ρ| < 0.3: tương quan yếu
```

> ⚠️ **Correlation ≠ Causation:** Kem que bán nhiều vào mùa hè, đuối nước cũng nhiều vào mùa hè → tương quan cao nhưng kem que không gây đuối nước!

---

### 3.6 Phân phối chuẩn trong ML — Z-score & Chuẩn hóa

**Z-score normalization (Standardization):**

```
z = (x − μ) / σ

Kết quả: mean = 0, std = 1 → phân phối N(0, 1)
```

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Sau khi scale: mean ≈ 0, std ≈ 1 theo từng feature
```

**Min-Max normalization:**

```
x_norm = (x − xmin) / (xmax − xmin)

Kết quả: giá trị ∈ [0, 1]
Nhược điểm: nhạy với outlier
```

---

### 3.7 Maximum Likelihood Estimation (MLE)

```
Ý tưởng: tìm tham số θ sao cho xác suất quan sát được dữ liệu là lớn nhất

θ_MLE = argmax P(data | θ)
       = argmax Σ log P(xᵢ | θ)   (dùng log để dễ tính)
```

**Ví dụ:** Tung đồng xu N lần, k lần ra mặt ngửa. MLE của p là:

```
p_MLE = k / N   (đơn giản là tỉ lệ thực nghiệm)
```

**Trong ML:** Training model = tìm tham số tối đa hóa likelihood của dữ liệu train:
- **Logistic Regression:** maximize log-likelihood = minimize cross-entropy loss
- **Linear Regression (Gaussian noise):** maximize likelihood = minimize MSE

---

## 📌 Bảng ôn nhanh trước khi thi

| Chủ đề | Công thức cần nhớ | Ứng dụng trong AI |
|--------|-------------------|-------------------|
| Dot product | a·b = Σaᵢbᵢ | Tích vô hướng, attention |
| Cosine sim | a·b / (‖a‖·‖b‖) | Semantic similarity, NLP |
| Eigenvalue | Av = λv | PCA, stability analysis |
| SVD | A = UΣVᵀ | Dimensionality reduction |
| Chain rule | dL/dx = dL/df × df/dg × ... | Backpropagation |
| Gradient descent | w ← w − η·∇L | Training mọi mô hình |
| Sigmoid deriv | σ(x)·(1−σ(x)) | Backprop qua sigmoid |
| Bayes | P(A\|B) = P(B\|A)P(A)/P(B) | Naive Bayes, posterior |
| L1 reg | L + λΣ\|w\| | Lasso, feature selection |
| L2 reg | L + λΣw² | Ridge, weight decay |
| Z-score | z = (x−μ)/σ | Feature normalization |

---

## ⚠️ Lưu ý khi làm bài

1. **Toán trong đề AI** thường xuất hiện dưới dạng: "Tính đạo hàm của hàm loss", "Viết công thức backprop", "Tính eigenvalue", "Giải thích tại sao dùng L2 thay L1".
2. **Chú ý đơn vị và chiều**: nhân ma trận (m×n) @ (n×p) = (m×p) — cột trái = hàng phải.
3. **Gradient Descent**: nhiều đề hỏi "tại sao dùng Adam thay SGD?" → Adam có adaptive learning rate + momentum.
4. **Bayes**: hay ra dưới dạng bài toán y tế/spam — nhớ tính P(evidence) đúng cách.
5. **Câu tự luận**: trình bày rõ bước đặt bài toán, viết công thức, giải thích ý nghĩa.

---

*Module A — Tài liệu ôn thi Vingroup AI thực chiến 2026*
