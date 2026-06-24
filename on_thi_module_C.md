# 🤖 Module C — Kiến thức AI & Tư duy Sản phẩm AI
### Chuẩn bị kỳ thi Vingroup AI thực chiến

---

## MỤC LỤC

1. [Machine Learning — Nền tảng](#1-machine-learning--nền-tảng)
2. [Các thuật toán ML quan trọng](#2-các-thuật-toán-ml-quan-trọng)
3. [Metrics đánh giá mô hình](#3-metrics-đánh-giá-mô-hình)
4. [Deep Learning](#4-deep-learning)
5. [NLP & LLM](#5-nlp--llm)
6. [Computer Vision](#6-computer-vision)
7. [Prompt Engineering](#7-prompt-engineering)
8. [Tư duy sản phẩm AI](#8-tư-duy-sản-phẩm-ai)

---

## 1. Machine Learning — Nền tảng

### 1.1 Phân loại bài toán ML

| Loại học | Đặc điểm | Thuật toán tiêu biểu |
|----------|----------|----------------------|
| **Supervised Learning** | Có nhãn y. Học f: X → Y | Linear/Logistic Regression, SVM, Decision Tree, Random Forest, XGBoost, KNN |
| **Unsupervised Learning** | Không có nhãn. Tìm cấu trúc ẩn | K-Means, DBSCAN, PCA, t-SNE, Autoencoders |
| **Semi-supervised** | Ít nhãn + nhiều data không nhãn | Self-training, Label Propagation |
| **Reinforcement Learning** | Agent học qua reward/penalty | Q-Learning, PPO, DQN, A3C |
| **Self-supervised** | Tự tạo nhãn từ data (BERT, GPT) | Masked LM, Contrastive Learning |

---

### 1.2 Quy trình xây dựng mô hình ML

```
1. Thu thập dữ liệu
        ↓
2. EDA (Exploratory Data Analysis)
   — phân phối, tương quan, missing values, outlier
        ↓
3. Tiền xử lý (Preprocessing)
   — xử lý null, encoding, scaling, feature engineering
        ↓
4. Chia dữ liệu: Train / Validation / Test
        ↓
5. Chọn & huấn luyện mô hình
        ↓
6. Đánh giá & tinh chỉnh
   — cross-validation, hyperparameter tuning
        ↓
7. Deploy & Giám sát
   — monitoring, retraining khi data drift
```

---

### 1.3 Tiền xử lý dữ liệu (Preprocessing)

#### Xử lý Missing Values

```python
# Xóa hàng có null
df.dropna()

# Điền bằng mean/median/mode
df["age"].fillna(df["age"].median(), inplace=True)

# Điền bằng giá trị lân cận (time series)
df.fillna(method="ffill")   # forward fill
df.fillna(method="bfill")   # backward fill
```

#### Encoding biến phân loại

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label Encoding: A→0, B→1, C→2  (dùng cho ordinal: thấp/trung/cao)
le = LabelEncoder()
df["grade"] = le.fit_transform(df["grade"])

# One-Hot Encoding: tạo cột nhị phân (dùng cho nominal: màu sắc, thành phố)
pd.get_dummies(df, columns=["city"])
# HN → city_HN=1, city_HCM=0
# HCM → city_HN=0, city_HCM=1
```

#### Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization: mean=0, std=1  (dùng với SVM, Linear, PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)   # fit trên train
X_test_scaled = scaler.transform(X_test)   # chỉ transform trên test!

# Min-Max: giá trị ∈ [0, 1]  (dùng với Neural Network, KNN)
mm = MinMaxScaler()
X_norm = mm.fit_transform(X)
```

> ⚠️ **Quan trọng:** Chỉ `fit` scaler trên **train set**, rồi `transform` cả train lẫn test. Không fit trên test — gây **data leakage**!

---

### 1.4 Train / Validation / Test Split

```
Thường dùng: 70% Train — 15% Validation — 15% Test
Hoặc:        80% Train — 20% Test  (với k-fold cross-validation)

Train:       học tham số mô hình (w, b)
Validation:  tinh chỉnh hyperparameter, chọn mô hình tốt nhất
Test:        đánh giá cuối cùng — chỉ dùng 1 lần!
```

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify giữ tỉ lệ class
)
```

---

### 1.5 Cross-Validation

```
k-Fold CV:
  ┌───┬───┬───┬───┬───┐
  │ 1 │ 2 │ 3 │ 4 │ 5 │  ← k=5 folds
  └───┴───┴───┴───┴───┘
  Lần 1: [TEST][trn][trn][trn][trn]
  Lần 2: [trn][TEST][trn][trn][trn]
  ...
  Score = trung bình 5 lần đánh giá
```

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Khi nào dùng:**
- `k-Fold`: data cân bằng
- `Stratified k-Fold`: data mất cân bằng (imbalanced)
- `Leave-One-Out (LOO)`: dataset rất nhỏ

---

## 2. Các thuật toán ML quan trọng

### 2.1 Linear Regression

```
Mô hình: ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ  =  wᵀx + b

Loss:    MSE = (1/n) Σ(yᵢ − ŷᵢ)²

Tối ưu:  Gradient Descent  hoặc  Closed-form: w = (XᵀX)⁻¹Xᵀy
```

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso

lr = LinearRegression()
lr.fit(X_train, y_train)
lr.coef_       # trọng số w
lr.intercept_  # hệ số chặn b

# Với regularization
ridge = Ridge(alpha=1.0)    # L2
lasso = Lasso(alpha=0.1)    # L1 — có thể đẩy nhiều w về 0
```

---

### 2.2 Logistic Regression

```
Mô hình: P(y=1|x) = σ(wᵀx + b) = 1 / (1 + e^(−(wᵀx+b)))

Loss:    Cross-Entropy = −(1/n) Σ [yᵢ·log(ŷᵢ) + (1−yᵢ)·log(1−ŷᵢ)]

Quyết định: ŷ = 1 nếu P(y=1|x) ≥ 0.5  (có thể điều chỉnh ngưỡng)
```

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=1.0)   # C = 1/λ (regularization strength)
clf.fit(X_train, y_train)

clf.predict(X_test)          # nhãn dự đoán (0 hoặc 1)
clf.predict_proba(X_test)    # xác suất → cột 0: P(y=0), cột 1: P(y=1)
```

---

### 2.3 Decision Tree

**Ý tưởng:** Phân chia dữ liệu dựa trên feature tốt nhất theo từng bước.

**Tiêu chí phân chia:**

```
Gini Index:  G = 1 − Σ pᵢ²
  G = 0: node thuần khiết (chỉ 1 class)
  G = 0.5: phân bổ đều 2 class  ← tệ nhất

Entropy:     H = −Σ pᵢ · log₂(pᵢ)
Information Gain = H(parent) − Σ (|child|/|parent|) × H(child)
  → Chọn feature có Information Gain lớn nhất
```

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    max_depth=5,        # giới hạn độ sâu — tránh overfit
    min_samples_leaf=5, # tối thiểu 5 sample ở leaf
    criterion="gini"    # hoặc "entropy"
)
```

**Ưu/nhược:**
- ✅ Dễ giải thích, không cần scaling, xử lý được mixed data
- ❌ Dễ overfit (cần prune), không ổn định (thay đổi ít data → tree khác nhiều)

---

### 2.4 Random Forest

```
Ý tưởng: Ensemble nhiều Decision Tree độc lập
          → Kết hợp dự đoán bằng majority vote (classification) hoặc mean (regression)

Kỹ thuật:
  1. Bagging: mỗi tree train trên bootstrap sample (lấy mẫu có hoàn lại)
  2. Random feature: mỗi split chỉ xét √p feature ngẫu nhiên (p = tổng features)
  → Giảm correlation giữa các tree → giảm variance tổng thể
```

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,    # số cây
    max_features="sqrt", # số feature xét mỗi split
    max_depth=10,
    random_state=42
)

# Feature importance
rf.feature_importances_  # mảng độ quan trọng của từng feature
```

---

### 2.5 XGBoost / Gradient Boosting

```
Ý tưởng: Boosting — các tree được xây tuần tự
          Mỗi tree mới học để sửa lỗi của tất cả tree trước

  Tree₁ → dự đoán → tính residual (sai số)
  Tree₂ → học từ residual của Tree₁
  Tree₃ → học từ residual tổng hợp
  ...
  Kết quả cuối = Σ η × treeᵢ(x)    (η = learning rate)
```

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,     # shrinkage — giảm contribution mỗi tree
    max_depth=6,
    subsample=0.8,         # bagging fraction
    colsample_bytree=0.8,  # feature fraction
    reg_alpha=0.1,         # L1 regularization
    reg_lambda=1.0,        # L2 regularization
)
```

**So sánh Random Forest vs XGBoost:**

| | Random Forest | XGBoost |
|--|---------------|---------|
| Training | Parallel (song song) | Sequential (tuần tự) |
| Bias-Variance | Giảm variance | Giảm cả bias lẫn variance |
| Tốc độ train | Nhanh hơn | Chậm hơn một chút |
| Overfit | Ít hơn DT | Cần tune cẩn thận |
| Tabular data | Tốt | **Tốt nhất** (benchmark winner) |

---

### 2.6 SVM (Support Vector Machine)

```
Ý tưởng: Tìm hyperplane phân chia 2 class với margin lớn nhất

Margin = 2 / ‖w‖  →  Maximize margin = Minimize ‖w‖²

Support vectors: các điểm nằm trên hoặc gần nhất với margin
```

**Kernel trick** — xử lý dữ liệu phi tuyến:

```
Linear kernel:   K(x,z) = xᵀz              — dữ liệu tuyến tính
Polynomial:      K(x,z) = (xᵀz + c)ᵈ
RBF (Gaussian):  K(x,z) = exp(−γ‖x−z‖²)   — phổ biến nhất, xử lý phi tuyến
```

```python
from sklearn.svm import SVC

svm = SVC(
    C=1.0,        # trade-off: C lớn → ít margin violation → dễ overfit
    kernel="rbf", # linear, poly, rbf, sigmoid
    gamma="scale" # bandwidth của RBF kernel
)
```

---

### 2.7 K-Means Clustering

```
Thuật toán:
  1. Khởi tạo K centroids ngẫu nhiên
  2. Gán mỗi điểm vào cluster có centroid gần nhất (L2 distance)
  3. Cập nhật centroid = mean của các điểm trong cluster
  4. Lặp 2–3 đến khi hội tụ

Objective: Minimize Σᵢ Σₓ∈Cᵢ ‖x − μᵢ‖²
```

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42, n_init=10)
km.fit(X)
km.labels_          # nhãn cluster của từng điểm
km.cluster_centers_ # tọa độ centroid
km.inertia_         # tổng bình phương khoảng cách → dùng Elbow method
```

**Chọn K — Elbow Method:**

```python
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)
# Vẽ đồ thị → chọn K tại "khuỷu tay" (điểm inertia giảm chậm lại)
```

---

### 2.8 PCA (Principal Component Analysis)

```
Ý tưởng: Chiếu dữ liệu lên các hướng có variance lớn nhất
          (là các eigenvector của covariance matrix)

Bước:
  1. Chuẩn hóa dữ liệu (mean=0)
  2. Tính covariance matrix C = (1/n) XᵀX
  3. Tính eigenvalues & eigenvectors của C
  4. Chọn k eigenvectors ứng với k eigenvalues lớn nhất
  5. Chiếu dữ liệu lên k hướng đó
```

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)        # giữ 2 chiều chính
X_pca = pca.fit_transform(X)

pca.explained_variance_ratio_    # % variance giải thích của mỗi component
# [0.72, 0.15] → PC1 giải thích 72%, PC2 giải thích 15% → tổng 87%

# Chọn n_components tự động giữ 95% variance
pca = PCA(n_components=0.95)
```

---

## 3. Metrics đánh giá mô hình

### 3.1 Classification Metrics — Confusion Matrix

```
                    Dự đoán Positive    Dự đoán Negative
Thực tế Positive        TP                   FN        (Recall row)
Thực tế Negative        FP                   TN

TP: dự đoán đúng là Positive   (True Positive)
TN: dự đoán đúng là Negative   (True Negative)
FP: dự đoán sai là Positive    (False Positive) — Type I Error
FN: dự đoán sai là Negative    (False Negative) — Type II Error
```

```python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_pred)
print(classification_report(y_true, y_pred))
```

---

### 3.2 Các Metrics phân loại

**Accuracy:**

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Dùng khi: class cân bằng
Không dùng khi: mất cân bằng (99% class A → predict toàn A đạt 99% accuracy nhưng vô nghĩa)
```

**Precision & Recall:**

```
Precision = TP / (TP + FP)
"Trong số dự đoán là Positive, bao nhiêu % đúng thật?"
→ Quan trọng khi FP tốn kém: spam filter (không muốn email thật vào spam)

Recall = TP / (TP + FN)
"Trong số Positive thật, tìm được bao nhiêu %?"
→ Quan trọng khi FN nguy hiểm: phát hiện bệnh (không muốn bỏ sót ca bệnh)
```

**F1-score:**

```
F1 = 2 × Precision × Recall / (Precision + Recall)

Trung bình điều hòa của P và R
→ Dùng khi data mất cân bằng hoặc cần cân bằng cả P và R
```

**Precision-Recall Tradeoff:**

```
Ngưỡng quyết định (threshold) mặc định = 0.5

Tăng threshold → Precision ↑, Recall ↓  (khắt khe hơn khi phán đúng Positive)
Giảm threshold → Precision ↓, Recall ↑  (dễ phán Positive hơn, bắt được nhiều hơn)

→ Điều chỉnh threshold tùy bài toán: bệnh → threshold thấp (recall cao), spam → threshold cao (precision cao)
```

**AUC-ROC:**

```
ROC Curve: vẽ TPR (Recall) vs FPR theo từng ngưỡng
  TPR = TP / (TP + FN)  = Recall
  FPR = FP / (FP + TN)

AUC (Area Under Curve): diện tích dưới đường ROC
  AUC = 1.0: mô hình hoàn hảo
  AUC = 0.5: random (đường chéo)
  AUC < 0.5: tệ hơn random

Ưu điểm: độc lập với ngưỡng quyết định, tốt cho imbalanced data
```

```python
from sklearn.metrics import roc_auc_score, f1_score

auc = roc_auc_score(y_true, y_prob[:, 1])   # dùng xác suất, không phải nhãn
f1  = f1_score(y_true, y_pred, average="macro")  # macro: trung bình các class
```

---

### 3.3 Regression Metrics

| Metric | Công thức | Ý nghĩa | Khi dùng |
|--------|-----------|---------|---------|
| **MAE** | (1/n)Σ\|yᵢ−ŷᵢ\| | Sai số tuyệt đối trung bình | Bền với outlier |
| **MSE** | (1/n)Σ(yᵢ−ŷᵢ)² | Phạt nặng sai số lớn | Khi outlier quan trọng |
| **RMSE** | √MSE | Cùng đơn vị với y | Dễ giải thích nhất |
| **R²** | 1 − Σ(y−ŷ)²/Σ(y−ȳ)² | % variance giải thích được | Đánh giá tổng thể |
| **MAPE** | (1/n)Σ\|yᵢ−ŷᵢ\|/yᵢ × 100% | Sai số theo % | Khi scale quan trọng |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_true, y_pred)
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_true, y_pred)
```

---

### 3.4 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search: thử mọi combination
param_grid = {
    "max_depth": [3, 5, 7, 10],
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.1, 0.3]
}
grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, scoring="f1")
grid.fit(X_train, y_train)
grid.best_params_   # hyperparameter tốt nhất

# Random Search: nhanh hơn Grid, thường cho kết quả tương đương
random = RandomizedSearchCV(XGBClassifier(), param_grid, n_iter=50, cv=5)
```

---

## 4. Deep Learning

### 4.1 Mạng nơ-ron nhân tạo (ANN)

```
Input layer → Hidden layer(s) → Output layer

Mỗi neuron:   z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b  =  wᵀx + b
              output = activation(z)

Forward pass:  tính output từ input (trái → phải)
Backward pass: tính gradient qua chain rule, cập nhật w (phải → trái)
```

**Các vấn đề khi train deep network:**

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|------------|----------|
| Vanishing gradient | Gradient → 0 khi backprop qua nhiều lớp sigmoid/tanh | Dùng ReLU, ResNet skip connections, batch norm |
| Exploding gradient | Gradient → ∞ | Gradient clipping, careful init |
| Overfitting | Model quá phức tạp | Dropout, L1/L2 reg, early stopping, data augmentation |
| Dying ReLU | Neuron luôn output 0, không học được | Leaky ReLU, ELU, khởi tạo w tốt hơn |

**Batch Normalization:**

```
Sau mỗi lớp: chuẩn hóa activation về mean≈0, std≈1
→ Train ổn định hơn, hội tụ nhanh hơn, giảm phụ thuộc vào learning rate
→ Có tác dụng regularization nhẹ
```

**Dropout:**

```python
# Trong PyTorch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(p=0.5),   # tắt ngẫu nhiên 50% neuron khi training
    nn.Linear(64, 10)
)
# Khi eval(): dropout tự động tắt
model.eval()
```

---

### 4.2 CNN — Convolutional Neural Network

**Kiến trúc cơ bản:**

```
Input (ảnh)
    → Conv Layer (trích xuất feature: cạnh, kết cấu, hình dạng)
    → Activation (ReLU)
    → Pooling Layer (giảm kích thước, giữ feature nổi bật)
    → ... (nhiều block)
    → Flatten
    → Fully Connected Layer
    → Output (softmax cho classification)
```

**Conv Layer — tham số quan trọng:**

```
Filter (Kernel): ma trận nhỏ (3×3, 5×5) trượt qua ảnh
  → Mỗi filter học 1 loại feature khác nhau

Stride: bước trượt của filter (thường = 1 hoặc 2)
Padding: "same" giữ nguyên kích thước, "valid" không padding

Output size = (Input - Filter + 2×Padding) / Stride + 1

Ví dụ: Input 32×32, Filter 3×3, Padding=1, Stride=1
  → Output = (32 - 3 + 2×1)/1 + 1 = 32×32  (same padding)
```

**Pooling Layer:**

```
Max Pooling 2×2, stride=2:
  [1 3 2 4]         [3 4]
  [5 6 1 2]  →  Max → [6 4]   (lấy giá trị lớn nhất trong mỗi vùng 2×2)
  [3 2 4 1]         [3 4]
  [1 0 2 3]
```

**Các kiến trúc CNN nổi tiếng:**

| Kiến trúc | Năm | Đặc điểm nổi bật |
|-----------|-----|-----------------|
| LeNet-5 | 1998 | CNN đầu tiên, nhận diện chữ số |
| AlexNet | 2012 | Đột phá ImageNet, dùng ReLU + Dropout |
| VGG-16/19 | 2014 | Deep nhưng đơn giản: nhiều Conv 3×3 |
| ResNet | 2015 | **Skip connections** — train được mạng 100+ lớp |
| EfficientNet | 2019 | Scale đồng thời depth+width+resolution |
| ViT | 2020 | Vision Transformer — dùng attention thay Conv |

**ResNet — Skip Connection:**

```
Normal block:       x → Conv → BN → ReLU → Conv → BN → output
ResNet block:       x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU → output
                    └────────────────────────────────────────┘  (shortcut)

→ Gradient đi thẳng qua shortcut, không bị vanish
→ Có thể train mạng 50, 101, 152 lớp
```

---

### 4.3 RNN — Recurrent Neural Network

```
Xử lý dữ liệu chuỗi: văn bản, time series, audio, video

Công thức:
  hₜ = f(Wₕ·hₜ₋₁ + Wₓ·xₜ + b)   ← hidden state tại bước t
  ŷₜ = g(Wᵧ·hₜ + bᵧ)             ← output tại bước t

hₜ đóng vai trò "bộ nhớ" — lưu thông tin từ các bước trước
```

**Vấn đề Vanishing Gradient trong RNN:**

```
Gradient backprop qua T bước: ∂L/∂h₁ = ∂L/∂hₜ × (∂hₜ/∂hₜ₋₁)ᵀ⁻¹

Nếu |∂hₜ/∂hₜ₋₁| < 1 → gradient → 0  (vanish, không học được phụ thuộc xa)
Nếu |∂hₜ/∂hₜ₋₁| > 1 → gradient → ∞  (explode)
```

**LSTM — Long Short-Term Memory:**

```
4 gate kiểm soát thông tin:

Forget gate:  fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)   — quên bao nhiêu từ cell state cũ
Input gate:   iₜ = σ(Wᵢ·[hₜ₋₁, xₜ] + bᵢ)   — cập nhật thông tin nào vào cell
Cell update:  C̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)  — thông tin mới
Cell state:   Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ      — bộ nhớ dài hạn
Output gate:  oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)   — output bao nhiêu từ cell
Hidden:       hₜ = oₜ * tanh(Cₜ)
```

**GRU — Gated Recurrent Unit:**

```
Đơn giản hơn LSTM (2 gate thay vì 4):
Reset gate:  rₜ = σ(Wr·[hₜ₋₁, xₜ])   — reset bao nhiêu hidden state cũ
Update gate: zₜ = σ(Wz·[hₜ₋₁, xₜ])   — balance giữa old và new state
→ Kết quả tương đương LSTM, train nhanh hơn
```

---

### 4.4 Transformer

**Self-Attention — cơ chế cốt lõi:**

```
Input: X (sequence of tokens)

Q = X·Wq   (Query  — "câu hỏi" của mỗi token)
K = X·Wk   (Key    — "nhãn" của mỗi token)
V = X·Wv   (Value  — "nội dung" của mỗi token)

Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V

Ý nghĩa:
  QKᵀ → attention score: token nào "liên quan" đến token nào
  / √d_k → scale để tránh softmax bị bão hòa
  softmax → chuẩn hóa thành trọng số attention [0,1]
  × V → lấy giá trị có trọng số
```

**Multi-Head Attention:**

```
Chạy attention h lần với các Wq, Wk, Wv khác nhau
→ Học nhiều loại quan hệ khác nhau đồng thời (cú pháp, ngữ nghĩa, tham chiếu...)
→ Kết quả: Concat(head₁, ..., headₕ) × Wₒ
```

**Kiến trúc Transformer:**

```
BERT (Encoder only):
  → Hiểu ngữ cảnh 2 chiều (bidirectional)
  → Pretrain: Masked Language Model (che 15% token, đoán lại)
  → Dùng cho: classification, NER, QA, embedding

GPT (Decoder only):
  → Sinh văn bản tự hồi quy (trái → phải)
  → Pretrain: dự đoán token tiếp theo
  → Dùng cho: text generation, chatbot, code generation

T5 / BART (Encoder-Decoder):
  → Seq2seq tasks
  → Dùng cho: dịch thuật, tóm tắt, hỏi đáp
```

---

## 5. NLP & LLM

### 5.1 Xử lý văn bản cơ bản

```
Pipeline NLP:
  Raw text
    → Tokenization (tách token)
    → Lowercasing, Remove punctuation
    → Stop word removal (loại "và", "là", "của"...)
    → Stemming / Lemmatization (đưa về dạng gốc)
    → Feature extraction (TF-IDF, Word2Vec, BERT embedding...)
```

**TF-IDF:**

```
TF(t, d)  = số lần từ t xuất hiện trong tài liệu d / tổng số từ trong d
IDF(t)    = log(N / df(t))    N: tổng tài liệu, df(t): số tài liệu chứa t
TF-IDF    = TF × IDF

Ý nghĩa: từ quan trọng = xuất hiện nhiều trong document (TF cao)
          nhưng ít trong corpus (IDF cao) → đặc trưng riêng
```

---

### 5.2 Word Embeddings

```
One-hot encoding: "mèo" = [0,0,1,0,0,...]   → sparse, không có semantic
Word2Vec:         "mèo" = [0.2, -0.5, 0.8, ...]  → dense, có semantic

Tính chất: vector("vua") − vector("đàn ông") + vector("phụ nữ") ≈ vector("nữ hoàng")
```

**Các loại embedding:**

| Loại | Đặc điểm |
|------|----------|
| Word2Vec | Static embedding, 1 từ = 1 vector cố định |
| GloVe | Static, train trên ma trận co-occurrence toàn corpus |
| FastText | Dùng subword — xử lý được từ OOV (out-of-vocabulary) |
| BERT embedding | Contextual — cùng từ, khác ngữ cảnh → vector khác nhau |

---

### 5.3 Large Language Models (LLM)

**Kiến trúc & Training:**

```
Pretraining:
  → Train trên hàng tỉ token văn bản
  → Objective: dự đoán token tiếp theo (Causal LM / GPT)
              hoặc đoán token bị che (Masked LM / BERT)

Fine-tuning:
  → Tiếp tục train trên task cụ thể với data nhỏ hơn
  → Full fine-tuning: cập nhật toàn bộ tham số (tốn kém)
  → LoRA: chỉ thêm ma trận low-rank nhỏ, tiết kiệm hơn nhiều

RLHF (Reinforcement Learning from Human Feedback):
  1. Supervised Fine-tuning (SFT) trên data chất lượng cao
  2. Train Reward Model từ ranking của human
  3. PPO: tối ưu policy để maximize reward
  → Kết quả: model theo instructions, helpful, harmless
```

**Các khái niệm quan trọng:**

| Khái niệm | Giải thích |
|-----------|-----------|
| **Context window** | Số token tối đa trong 1 lần. GPT-4: 128k. Claude: 200k. |
| **Temperature** | Độ ngẫu nhiên. 0 = deterministic. >1 = sáng tạo hơn, ít nhất quán. |
| **Top-p (nucleus)** | Chỉ sample từ top p% xác suất tích lũy. p=0.9 thường dùng. |
| **Top-k** | Chỉ sample từ k token có xác suất cao nhất. |
| **Hallucination** | LLM tạo thông tin sai/bịa đặt với độ tin tưởng cao. |
| **Emergent abilities** | Khả năng xuất hiện đột ngột khi model đạt kích thước nhất định. |
| **In-context learning** | Học từ ví dụ trong prompt mà không cập nhật weight. |

---

### 5.4 RAG — Retrieval-Augmented Generation

```
Vấn đề: LLM có knowledge cutoff, hallucinate khi không biết

Giải pháp RAG:
  1. Encode documents → vector embeddings → lưu vào Vector DB
  2. User query → encode → tìm k documents tương đồng nhất (similarity search)
  3. Đưa documents tìm được vào context của LLM
  4. LLM trả lời dựa trên context thực (giảm hallucination)

Kiến trúc:
  Query → [Retriever] → relevant docs → [Generator (LLM)] → Answer
               ↑
          Vector DB (FAISS, Pinecone, Chroma, Weaviate)
```

---

## 6. Computer Vision

### 6.1 Các bài toán CV cơ bản

| Bài toán | Mô tả | Model tiêu biểu |
|----------|-------|----------------|
| **Image Classification** | Gán nhãn cho ảnh (1 nhãn) | ResNet, EfficientNet, ViT |
| **Object Detection** | Định vị + phân loại nhiều object trong ảnh | YOLO, Faster R-CNN, DETR |
| **Semantic Segmentation** | Gán nhãn cho từng pixel | U-Net, DeepLab, SegFormer |
| **Instance Segmentation** | Phân biệt từng instance cùng class | Mask R-CNN |
| **Image Generation** | Tạo ảnh mới | GAN, Diffusion Models (DALL-E, Stable Diffusion) |
| **Face Recognition** | Nhận diện danh tính | FaceNet, ArcFace |

---

### 6.2 Object Detection — YOLO

```
YOLO (You Only Look Once):
  → Chia ảnh thành grid S×S
  → Mỗi cell dự đoán B bounding boxes và class probabilities
  → Xử lý toàn bộ ảnh 1 lần (1 forward pass) → rất nhanh

Output: [x, y, w, h, confidence, class_probs] cho mỗi bounding box

NMS (Non-Maximum Suppression):
  → Loại bỏ các bounding box overlap (IoU > threshold)
  → Giữ lại box có confidence cao nhất

IoU (Intersection over Union):
  IoU = Area(A ∩ B) / Area(A ∪ B)
  IoU > 0.5: thường coi là detection đúng
```

---

### 6.3 Transfer Learning trong CV

```python
import torchvision.models as models
import torch.nn as nn

# Load pretrained ResNet50 (train trên ImageNet — 1M ảnh, 1000 class)
model = models.resnet50(pretrained=True)

# Freeze tất cả layer — chỉ train classifier mới
for param in model.parameters():
    param.requires_grad = False

# Thay fc layer cho task mới (ví dụ: 5 class)
model.fc = nn.Linear(model.fc.in_features, 5)

# Chỉ layer fc mới có requires_grad=True → train nhanh, ít data hơn
```

**Khi nào dùng gì:**

```
Data nhỏ, task giống ImageNet     → Freeze toàn bộ, chỉ train classifier
Data nhỏ, task khác               → Fine-tune vài layer cuối
Data lớn, task giống              → Fine-tune toàn bộ với lr nhỏ
Data lớn, task rất khác           → Train từ đầu
```

---

## 7. Prompt Engineering

### 7.1 Các kỹ thuật cơ bản

**Zero-shot:**

```
Không cho ví dụ. Dựa hoàn toàn vào kiến thức của model.

Prompt: "Phân loại cảm xúc câu sau: 'Tôi rất vui vì đậu kỳ thi!'
         Trả lời: Tích cực / Tiêu cực / Trung tính"
Output: "Tích cực"
```

**Few-shot:**

```
Cho 2–5 ví dụ mẫu trước câu cần trả lời.

Prompt:
  "Phân loại cảm xúc:
   Câu: 'Hôm nay trời đẹp quá!' → Tích cực
   Câu: 'Tôi bị trượt môn rồi.' → Tiêu cực
   Câu: 'Thời tiết hôm nay bình thường.' → Trung tính
   Câu: 'Tôi vừa nhận được học bổng!' → ?"

Output: "Tích cực"
```

**Chain-of-Thought (CoT):**

```
Thêm "Hãy suy nghĩ từng bước" → model lập luận trước khi kết luận

Prompt: "Một cửa hàng bán 3 loại bánh. Bánh A giá 15k, bánh B giá 20k,
         bánh C giá 12k. Mua 2A, 3B, 1C hết bao nhiêu?
         Hãy tính từng bước."

Output:
  "- 2 bánh A: 2 × 15k = 30k
   - 3 bánh B: 3 × 20k = 60k
   - 1 bánh C: 1 × 12k = 12k
   - Tổng: 30k + 60k + 12k = 102k"
```

---

### 7.2 Các kỹ thuật nâng cao

| Kỹ thuật | Mô tả | Khi dùng |
|----------|-------|---------|
| **Role prompting** | "Bạn là chuyên gia X..." | Cần domain expertise, tone cụ thể |
| **Structured output** | "Trả lời dạng JSON: {key: value}" | Cần parse output tự động |
| **Self-consistency** | Chạy nhiều lần, lấy majority vote | Tăng độ chính xác câu hỏi khó |
| **Tree of Thought** | Khám phá nhiều nhánh suy luận | Bài toán phức tạp, nhiều bước |
| **ReAct** | Xen kẽ Reasoning và Acting (tool use) | Agent gọi tool/API |
| **RAG** | Thêm retrieved context vào prompt | Cần thông tin cập nhật, tránh hallucination |

---

### 7.3 Nguyên tắc viết Prompt tốt

```
1. Rõ ràng & cụ thể:
   Tệ:  "Viết về AI"
   Tốt: "Viết đoạn giới thiệu 200 từ về ứng dụng AI trong y tế,
         dành cho đối tượng là bệnh nhân không có nền tảng kỹ thuật"

2. Định nghĩa output format:
   "Trả lời theo format:
    Ưu điểm: ...
    Nhược điểm: ...
    Kết luận: ..."

3. Cho context đủ:
   Thêm persona, ràng buộc, ví dụ negative (những gì KHÔNG muốn)

4. Chia task phức tạp thành subtask nhỏ hơn

5. Yêu cầu model kiểm tra lại:
   "Sau khi trả lời, hãy kiểm tra lại xem kết quả có đúng không"
```

---

## 8. Tư duy sản phẩm AI

### 8.1 Khung tư duy khi tiếp cận bài toán AI

```
Bước 1: Có cần AI không?
  → Rule-based giải được không? (if-else, regex, lookup table)
  → AI cần khi: pattern phức tạp, data lớn, cần scale, edge case nhiều

Bước 2: Định nghĩa bài toán rõ ràng
  → Input là gì? Output là gì?
  → Metric thành công (business metric vs ML metric)
  → Baseline là gì?

Bước 3: Dữ liệu
  → Có đủ data không? Chất lượng thế nào?
  → Thu thập thêm bao nhiêu? Chi phí?
  → Privacy & compliance issues?

Bước 4: Chọn approach
  → Bắt đầu từ simple model (logistic regression, rule-based)
  → Iterate dần lên phức tạp
  → Tránh over-engineering sớm

Bước 5: Đánh giá & Deploy
  → Offline evaluation (test set)
  → Online evaluation (A/B test)
  → Monitoring sau deploy
```

---

### 8.2 Business Metric vs ML Metric

```
Bài toán: Recommend sản phẩm cho người dùng

ML Metric:         Precision@10 = 0.8  (80% recommendation đúng)
Business Metric:   Click-through rate, Revenue, User retention

→ ML metric tốt ≠ business metric tốt!
→ Ví dụ: recommend sản phẩm đắt tiền có thể giảm CTR nhưng tăng revenue

Luôn theo dõi cả 2 loại metric.
```

---

### 8.3 Trade-offs quan trọng trong sản phẩm AI

| Trade-off | Giải thích |
|-----------|-----------|
| **Accuracy vs Latency** | Model to → chính xác hơn nhưng chậm. Cần cân bằng UX và quality. |
| **Precision vs Recall** | Tùy bài toán: spam (precision), y tế (recall). |
| **Complexity vs Interpretability** | Deep learning mạnh hơn nhưng khó giải thích. Regulated industries cần giải thích được. |
| **Model performance vs Data privacy** | Dùng nhiều data hơn → tốt hơn, nhưng privacy risk tăng. |
| **Automation vs Human oversight** | Tự động hóa giảm chi phí, nhưng high-stakes decisions cần người review. |

---

### 8.4 MLOps — Vận hành mô hình trong thực tế

```
CI/CD cho ML:
  Code change → Test → Build → Deploy (tự động)
  Model change → Evaluate → Shadow deploy → A/B test → Full deploy

Monitoring sau deploy:
  - Data drift: phân phối input thay đổi theo thời gian
  - Concept drift: mối quan hệ X→y thay đổi (model cũ không còn phù hợp)
  - Performance metrics: theo dõi accuracy/F1/latency liên tục

Khi nào retrain:
  - Performance drop dưới ngưỡng
  - Data drift đáng kể
  - Theo lịch (weekly, monthly)
  - Event-triggered (thay đổi business lớn)
```

---

### 8.5 Ví dụ Case Study — Hay gặp trong đề tự luận

**Case 1: Xây dựng hệ thống phát hiện gian lận (Fraud Detection)**

```
Đặc điểm:
  - Imbalanced: 0.1% giao dịch là fraud → không dùng accuracy
  - Metric: F1, AUC-ROC, Precision@high_recall
  - Cost asymmetry: FN (bỏ sót fraud) >> FP (chặn nhầm giao dịch hợp lệ)

Approach:
  1. Oversample minority (SMOTE) hoặc class_weight
  2. XGBoost/Random Forest với custom threshold
  3. Rule-based layer kết hợp ML
  4. Real-time inference (< 100ms)
  5. Giải thích được (SHAP) vì cần audit
```

**Case 2: Chatbot hỗ trợ khách hàng**

```
Approach:
  1. Intent classification → route đến handler đúng
  2. LLM cho open-domain questions + RAG với knowledge base
  3. Fallback to human agent khi confidence thấp
  4. Metrics: resolution rate, escalation rate, CSAT score
  5. Safety: filter toxic input/output, không leak thông tin nhạy cảm
```

---

## 📌 Bảng ôn nhanh trước khi thi

| Chủ đề | Điểm cần nhớ |
|--------|-------------|
| Supervised vs Unsupervised | Có nhãn / không nhãn |
| Bias-Variance | Underfitting (bias) / Overfitting (variance) |
| Precision | TP/(TP+FP) — tránh FP (spam) |
| Recall | TP/(TP+FN) — tránh FN (bệnh) |
| F1 | 2PR/(P+R) — cân bằng cả 2, imbalanced data |
| AUC-ROC | Độc lập ngưỡng, tốt cho imbalanced |
| CNN | Conv → Pool → FC. ResNet giải quyết vanishing gradient |
| RNN/LSTM | Chuỗi. LSTM có forget/input/output gate |
| Transformer | Self-attention = softmax(QKᵀ/√d)×V |
| BERT vs GPT | Encoder (hiểu) vs Decoder (sinh văn bản) |
| RAG | Retrieve docs → thêm vào context → LLM trả lời |
| MLOps | Data drift, concept drift, monitoring, retraining |
| A/B testing | So sánh model mới/cũ trên user thật |

---

## ⚠️ Lưu ý khi làm bài

1. **Câu tự luận về chọn mô hình:** giải thích rõ lý do — data size, interpretability, latency, imbalanced hay không.
2. **Khi hỏi metric:** luôn nêu ngữ cảnh — imbalanced data dùng F1, không dùng accuracy.
3. **Transformer:** nhiều đề hỏi giải thích self-attention, tại sao chia √d_k — để tránh dot product quá lớn → softmax bão hòa.
4. **CNN:** hay hỏi tính output size sau Conv/Pool — nhớ công thức `(Input - Filter + 2P)/S + 1`.
5. **Câu sản phẩm:** luôn đề cập đến business metric, không chỉ ML metric. Nêu trade-off rõ ràng.

---

*Module C — Tài liệu ôn thi Vingroup AI thực chiến 2026*
