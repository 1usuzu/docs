# 🧠 Module D — Tư duy Logic, Đạo đức & Hành vi
### Chuẩn bị kỳ thi Vingroup AI thực chiến

---

## MỤC LỤC

1. [Tư duy Logic & Suy luận](#1-tư-duy-logic--suy-luận)
2. [Phân tích & Giải quyết vấn đề](#2-phân-tích--giải-quyết-vấn-đề)
3. [Đạo đức AI (AI Ethics)](#3-đạo-đức-ai-ai-ethics)
4. [Bias trong AI](#4-bias-trong-ai)
5. [Hành vi & Làm việc nhóm](#5-hành-vi--làm-việc-nhóm)
6. [Tình huống thực tế & Cách trả lời](#6-tình-huống-thực-tế--cách-trả-lời)

---

## 1. Tư duy Logic & Suy luận

### 1.1 Các dạng suy luận cơ bản

#### Deductive Reasoning (Diễn dịch)

```
Từ tiền đề chung → kết luận cụ thể
Nếu tiền đề đúng → kết luận CHẮC CHẮN đúng

Ví dụ:
  Tiền đề 1: Mọi mô hình ML đều cần dữ liệu để huấn luyện.
  Tiền đề 2: Random Forest là mô hình ML.
  Kết luận:  Random Forest cần dữ liệu để huấn luyện. ✓
```

#### Inductive Reasoning (Quy nạp)

```
Từ quan sát cụ thể → quy tắc chung
Kết luận CÓ THỂ đúng, không chắc chắn 100%

Ví dụ:
  Quan sát: 1000 model XGBoost trên tabular data đều cho kết quả tốt.
  Kết luận: XGBoost thường tốt với tabular data.
  ⚠️ Không thể chắc chắn — có thể có ngoại lệ
```

#### Abductive Reasoning (Giả thuyết)

```
Quan sát hiện tượng → chọn giải thích đơn giản nhất có thể

Ví dụ:
  Quan sát: Model accuracy giảm đột ngột sau deploy.
  Giả thuyết tốt nhất: Data drift — phân phối input thay đổi so với lúc train.
  → Kiểm tra phân phối data production vs training.
```

#### Analogical Reasoning (Loại suy)

```
So sánh tình huống tương tự để suy luận

Ví dụ:
  "Transfer learning trong AI giống như người đã biết lái xe máy
   học lái ô tô — dùng lại kỹ năng cân bằng, nhận diện đường,
   chỉ cần học thêm phần mới (chân ga, số tự động...)"

⚠️ Cẩn thận: loại suy có thể sai khi 2 tình huống khác nhau ở điểm then chốt
```

---

### 1.2 Lỗi logic (Logical Fallacies) — Hay gặp trong đề

| Lỗi logic | Mô tả | Ví dụ |
|-----------|-------|-------|
| **Ad Hominem** | Tấn công người nói thay vì lập luận | "Ý kiến của anh ấy sai vì anh ấy không có bằng AI" |
| **Straw Man** | Bóp méo ý kiến đối phương rồi phản bác | "Bạn muốn dùng AI → bạn muốn thay thế toàn bộ nhân viên" |
| **False Dichotomy** | Chỉ đưa ra 2 lựa chọn dù có nhiều hơn | "Hoặc dùng AI hoặc tụt hậu" |
| **Slippery Slope** | A → B → C → thảm họa (không có bằng chứng) | "Cho AI tự động hóa 1 việc → cuối cùng AI kiểm soát tất cả" |
| **Correlation = Causation** | Nhầm tương quan thành nhân quả | "Doanh số tăng cùng lúc deploy AI → AI gây tăng doanh số" |
| **Appeal to Authority** | Chỉ dựa vào người nổi tiếng, không có lý lẽ | "Elon Musk nói AI nguy hiểm nên AI nguy hiểm" |
| **Confirmation Bias** | Chỉ tìm bằng chứng xác nhận ý kiến sẵn có | Chỉ đọc báo cáo nói AI tốt, bỏ qua rủi ro |
| **Sunk Cost Fallacy** | Tiếp tục vì đã đầu tư nhiều, dù không còn hiệu quả | "Đã mất 6 tháng build model này, dù kết quả kém vẫn tiếp tục" |
| **Survivorship Bias** | Chỉ nhìn thành công, bỏ qua thất bại | "Startup AI này thành công → AI startup đều dễ thành công" |
| **Anchoring** | Bị ảnh hưởng quá nhiều bởi thông tin đầu tiên | Ước tính budget AI dựa trên con số nghe đầu tiên |

---

### 1.3 Tư duy phản biện (Critical Thinking)

**Các câu hỏi cần đặt ra khi đánh giá một luận điểm:**

```
1. Bằng chứng ở đâu?
   → Số liệu, nghiên cứu, thực nghiệm — hay chỉ là ý kiến?

2. Nguồn có đáng tin không?
   → Có conflict of interest không? Phương pháp nghiên cứu thế nào?

3. Có giải thích nào khác không?
   → Correlation hay causation? Confounding variables?

4. Kết luận có quá tổng quát không?
   → Sample size bao nhiêu? Đại diện cho ai?

5. Giả định ẩn là gì?
   → Luận điểm này dựa trên giả định nào mà không nói ra?
```

---

### 1.4 Dạng bài logic hay gặp trong thi

#### Dạng 1: Suy luận từ điều kiện

```
Đề: Nếu model có accuracy > 90% VÀ latency < 100ms → deploy được.
    Model A: accuracy = 95%, latency = 80ms.
    Model B: accuracy = 92%, latency = 120ms.
    Model C: accuracy = 88%, latency = 70ms.
    → Model nào deploy được?

Giải:
    Model A: 95% > 90% ✓  AND  80ms < 100ms ✓  → Deploy được ✓
    Model B: 92% > 90% ✓  BUT  120ms > 100ms ✗  → Không deploy
    Model C: 88% < 90% ✗  → Không cần xét điều kiện 2 → Không deploy
```

#### Dạng 2: Tìm kết luận đúng/sai

```
Đề: "Tất cả mô hình supervised learning đều cần labeled data.
     Linear Regression không cần labeled data."
     → Kết luận này đúng hay sai? Tại sao?

Giải: SAI.
     Linear Regression là supervised learning → cần labeled data (y).
     Tiền đề 2 mâu thuẫn với tiền đề 1.
     → Kết luận rút ra từ 2 tiền đề mâu thuẫn không thể tin cậy.
```

#### Dạng 3: Tìm lỗ hổng trong lập luận

```
Đề: "Công ty X dùng AI tuyển dụng, tỉ lệ nhân viên giỏi tăng 30%.
     → AI tuyển dụng luôn tốt hơn người."

Lỗ hổng:
  1. Correlation ≠ Causation: tỉ lệ nhân viên giỏi tăng có thể do nhiều nguyên nhân.
  2. Survivorship bias: chỉ thấy công ty thành công, không thấy công ty thất bại.
  3. Định nghĩa "nhân viên giỏi" là gì? Ai đánh giá? Bias trong label?
  4. Thời gian quan sát bao lâu? 30% tăng sau bao lâu?
  5. Baseline: tỉ lệ nhân viên giỏi trước đó là bao nhiêu?
```

---

## 2. Phân tích & Giải quyết vấn đề

### 2.1 Framework MECE

```
MECE = Mutually Exclusive, Collectively Exhaustive
(Loại trừ lẫn nhau, Bao phủ toàn bộ)

Dùng để phân tích vấn đề không bị chồng chéo và không bỏ sót.

Ví dụ — Phân tích nguyên nhân model kém:
  ├── Vấn đề dữ liệu
  │     ├── Không đủ data
  │     ├── Data chất lượng thấp (nhiễu, sai nhãn)
  │     └── Data drift (production khác training)
  ├── Vấn đề mô hình
  │     ├── Underfitting (model quá đơn giản)
  │     ├── Overfitting (model quá phức tạp)
  │     └── Hyperparameter chưa tối ưu
  └── Vấn đề triển khai
        ├── Preprocessing khác nhau train vs production
        ├── Feature engineering không nhất quán
        └── Version mismatch
```

---

### 2.2 Framework giải quyết vấn đề (6 bước)

```
Bước 1 — XÁC ĐỊNH VẤN ĐỀ
  • Vấn đề thực sự là gì? (Đừng nhầm triệu chứng với nguyên nhân)
  • Ai bị ảnh hưởng? Mức độ nghiêm trọng thế nào?
  • Khi nào bắt đầu xảy ra? Điều gì thay đổi?

Bước 2 — THU THẬP THÔNG TIN
  • Cần data gì để hiểu vấn đề?
  • Đã có thông tin gì rồi? Còn thiếu gì?
  • Ai là người hiểu vấn đề này nhất?

Bước 3 — PHÂN TÍCH NGUYÊN NHÂN GỐC RỄ (Root Cause)
  • 5 Whys: hỏi "Tại sao?" 5 lần để tìm nguyên nhân gốc
  • Fishbone diagram: liệt kê các nhóm nguyên nhân (người, quy trình, công cụ, data)

Bước 4 — ĐỀ XUẤT PHƯƠNG ÁN
  • Ít nhất 2–3 phương án khác nhau
  • Với mỗi phương án: chi phí, thời gian, rủi ro, tác động

Bước 5 — QUYẾT ĐỊNH & THỰC HIỆN
  • Chọn phương án tối ưu dựa trên tiêu chí rõ ràng
  • Chia thành các bước nhỏ, định mốc kiểm tra (milestone)
  • Xác định người chịu trách nhiệm

Bước 6 — ĐÁNH GIÁ KẾT QUẢ
  • Đo lường bằng metric đã định trước
  • Vấn đề có được giải quyết không?
  • Học được gì? Điều chỉnh nếu cần
```

---

### 2.3 Kỹ thuật 5 Whys — Tìm nguyên nhân gốc rễ

```
Vấn đề: Model production có accuracy giảm 15% so với test set.

Why 1: Tại sao accuracy giảm?
  → Phân phối data production khác data training.

Why 2: Tại sao phân phối khác?
  → Data training thu thập từ 2022, production là data 2024.

Why 3: Tại sao không cập nhật data?
  → Không có pipeline tự động thu thập data mới.

Why 4: Tại sao không có pipeline đó?
  → Không ai được assign task xây dựng monitoring & retraining.

Why 5: Tại sao không có ai phụ trách?
  → Khi deploy, team chỉ tập trung vào launch, không có kế hoạch hậu deploy.

Root cause: Thiếu quy trình MLOps — không có monitoring và retraining plan từ đầu.
Giải pháp: Xây dựng data pipeline + monitoring dashboard + retrain trigger.
```

---

### 2.4 Ra quyết định dưới áp lực & sự mơ hồ

**Khi thiếu thông tin:**

```
1. Xác định rõ: biết gì, không biết gì, cần biết gì để quyết định
2. Đặt ra giả định tường minh (explicit assumptions)
3. Chọn phương án có thể reversible (có thể sửa được nếu sai)
4. Bắt đầu nhỏ, fail fast, iterate nhanh
5. Định thời điểm review lại quyết định
```

**Trade-off matrix:**

```
                  Impact Cao    Impact Thấp
Effort Thấp   |  Quick wins  |  Fill-ins   |   ← Làm ngay
Effort Cao    |  Major proj  |  Avoid      |   ← Cần cân nhắc kỹ
```

---

## 3. Đạo đức AI (AI Ethics)

### 3.1 Sáu nguyên tắc đạo đức AI cốt lõi

#### 1. Fairness (Công bằng)

```
AI không được phân biệt đối xử dựa trên:
  giới tính, chủng tộc, độ tuổi, tôn giáo, khuyết tật, quốc tịch...

Các định nghĩa "công bằng" (thường mâu thuẫn nhau):

Demographic Parity:
  Tỉ lệ kết quả tích cực bằng nhau giữa các nhóm
  P(ŷ=1 | nhóm A) = P(ŷ=1 | nhóm B)

Equalized Odds:
  TPR và FPR bằng nhau giữa các nhóm
  P(ŷ=1 | y=1, nhóm A) = P(ŷ=1 | y=1, nhóm B)
  P(ŷ=1 | y=0, nhóm A) = P(ŷ=1 | y=0, nhóm B)

Individual Fairness:
  Người giống nhau → quyết định giống nhau
  d(x₁, x₂) nhỏ → d(f(x₁), f(x₂)) nhỏ

⚠️ Impossibility theorem: không thể thỏa mãn tất cả định nghĩa fairness
   cùng lúc khi base rate khác nhau giữa các nhóm.
   → Cần chọn định nghĩa phù hợp với ngữ cảnh bài toán.
```

#### 2. Transparency (Minh bạch)

```
Người dùng cần biết:
  • Khi nào đang tương tác với AI (không giả vờ là người)
  • AI đã dùng dữ liệu gì của họ
  • AI ra quyết định như thế nào (explainability)
  • Ai sở hữu và vận hành AI này

Quyền được giải thích (Right to Explanation):
  → GDPR Article 22: người dùng có quyền biết logic của quyết định tự động
    ảnh hưởng đến họ (khoản vay, tuyển dụng, bảo hiểm...)
```

#### 3. Accountability (Trách nhiệm)

```
Khi AI gây hại, phải xác định được ai chịu trách nhiệm:
  • Developer: thiết kế mô hình có lỗi?
  • Công ty deploy: dùng AI sai mục đích?
  • Người dùng cuối: dùng AI vô trách nhiệm?

Accountability gap: AI tự động ra quyết định → khó quy trách nhiệm
Giải pháp: Human-in-the-loop cho quyết định quan trọng
            Audit trail: log đầy đủ quyết định của AI
            Clear ownership: ai chịu trách nhiệm ở mỗi giai đoạn
```

#### 4. Privacy (Riêng tư)

```
Các nguyên tắc:
  • Data minimization: chỉ thu thập data thực sự cần thiết
  • Purpose limitation: data chỉ dùng đúng mục đích đã nêu
  • Storage limitation: không giữ data lâu hơn cần thiết
  • Informed consent: người dùng biết và đồng ý trước

Kỹ thuật bảo vệ privacy:
  • Differential Privacy: thêm nhiễu vào data/model để không identify cá nhân
  • Federated Learning: train model tại thiết bị người dùng, không gửi data về
  • Data anonymization: xóa hoặc mã hóa thông tin nhận dạng
  • Synthetic data: tạo data giả giữ nguyên thống kê nhưng không reveal cá nhân
```

#### 5. Safety (An toàn)

```
AI không được gây hại:
  • Trực tiếp: AI điều khiển xe tự lái, robot phẫu thuật
  • Gián tiếp: AI gợi ý nội dung cực đoan, deepfake

Các biện pháp:
  • Red teaming: cố tình tấn công AI để tìm điểm yếu trước khi deploy
  • Adversarial robustness: model không bị đánh lừa bởi input cố ý sai
  • Graceful degradation: khi không chắc → trả về "không biết" thay vì trả lời sai
  • Kill switch: con người có thể tắt AI ngay lập tức
```

#### 6. Human Oversight (Giám sát của con người)

```
Con người phải giữ quyền kiểm soát tối thượng:
  • Với quyết định high-stakes (y tế, tư pháp, tài chính lớn): bắt buộc human review
  • AI nên trình bày tùy chọn và lý do, không chỉ đưa ra 1 quyết định duy nhất
  • Người dùng có quyền từ chối / override quyết định của AI
  • Con người chịu trách nhiệm cuối cùng, không phải AI

Mức độ tự động hóa (Levels of Automation):
  1. Human does everything
  2. AI gợi ý, human quyết định
  3. AI quyết định, human veto được
  4. AI quyết định, chỉ inform human
  5. AI quyết định hoàn toàn
  → Chọn level phù hợp với stakes và độ tin cậy của AI
```

---

### 3.2 AI Ethics trong thực tế — Các trường hợp nổi tiếng

#### COMPAS (Recidivism Prediction)

```
Bối cảnh: Công cụ dự đoán tái phạm tội, dùng trong hệ thống tư pháp Mỹ.

Vấn đề phát hiện (ProPublica 2016):
  → FPR của người da màu cao gấp đôi người da trắng
     (bị gán nhãn "nguy cơ cao" sai nhiều hơn)
  → FNR của người da trắng cao hơn
     (thoát khỏi nhãn "nguy cơ cao" dù tái phạm)

Phản hồi của công ty:
  → Họ chứng minh Calibration (điểm số = xác suất thực) là công bằng
  → Nhưng Equalized Odds không thỏa mãn

Bài học:
  → "Công bằng" có nhiều định nghĩa, không thể thỏa mãn tất cả
  → AI không nên là quyết định duy nhất trong tư pháp
  → Cần transparency về cách model hoạt động
```

#### Amazon Hiring Tool (2018)

```
Bối cảnh: Amazon xây dựng AI tuyển dụng, train trên CV nhân viên 10 năm qua.

Vấn đề:
  → Model penalize CV có từ "women's" (câu lạc bộ nữ, trường nữ...)
  → Ưu tiên động từ mạnh kiểu nam như "executed", "captured"
  → Học lại historical bias: nhân viên tech trong quá khứ chủ yếu là nam

Kết quả: Amazon hủy dự án, không dùng model này.

Bài học:
  → Historical bias trong data → model học và khuếch đại bias đó
  → Cần audit fairness trước khi deploy
  → Đội ngũ phát triển cần đa dạng (diverse team) để phát hiện blind spot
```

---

### 3.3 Explainable AI (XAI)

#### SHAP (SHapley Additive exPlanations)

```
Dựa trên Shapley values trong lý thuyết trò chơi:
  → Phân phối "công lao" của từng feature vào dự đoán cuối

SHAP value của feature i = contribution của feature i vào dự đoán,
                            tính trung bình trên mọi thứ tự kết hợp

Tính chất tốt:
  • Local accuracy: tổng SHAP values = dự đoán của model
  • Missingness: feature không có mặt → SHAP = 0
  • Consistency: feature quan trọng hơn → SHAP cao hơn (hoặc thấp hơn đều nhất quán)
```

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Waterfall plot: giải thích 1 dự đoán cụ thể
shap.waterfall_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# Summary plot: tổng quan tất cả features
shap.summary_plot(shap_values, X_test)
```

#### LIME (Local Interpretable Model-agnostic Explanations)

```
Ý tưởng: với mỗi điểm cần giải thích,
  tạo nhiều sample lân cận → train linear model đơn giản
  → Linear model xấp xỉ behavior của black-box model tại vùng lân cận đó

Ưu điểm: model-agnostic (dùng được với mọi model)
Nhược điểm: không ổn định, kết quả có thể khác nhau mỗi lần chạy
```

#### Attention Visualization (Transformer)

```
Trong BERT/GPT: attention weights cho thấy
  token nào đang "chú ý" đến token nào khi tạo ra output

Ví dụ: "Con mèo ngồi trên [MASK]"
  → Token "[MASK]" attend nhiều đến "mèo" và "ngồi"
  → Gợi ý model dự đoán: "chiếu", "ghế", "bàn"...

⚠️ Lưu ý: attention ≠ explanation hoàn toàn (vẫn đang được nghiên cứu)
```

---

## 4. Bias trong AI

### 4.1 Các loại Bias và nguồn gốc

```
Vòng đời dữ liệu và bias xuất hiện ở đâu:

  Thế giới thực
       ↓  [Historical bias: bất bình đẳng đã tồn tại trong xã hội]
  Thu thập data
       ↓  [Sampling bias: không đại diện, Coverage bias: thiếu nhóm]
  Gán nhãn (Labeling)
       ↓  [Labeler bias: định kiến của người gán nhãn]
  Tiền xử lý
       ↓  [Aggregation bias: gộp nhóm khác nhau làm một]
  Training
       ↓  [Algorithmic bias: objective function không capture fairness]
  Đánh giá
       ↓  [Evaluation bias: benchmark không đại diện]
  Deploy
       ↓  [Deployment bias: dùng model sai mục đích ban đầu]
  Feedback loop
       ↓  [Model kém → quyết định tệ → data mới bị ảnh hưởng → model càng tệ]
```

---

### 4.2 Bảng tổng hợp các loại Bias

| Loại Bias | Định nghĩa | Ví dụ thực tế |
|-----------|-----------|--------------|
| **Historical bias** | Data phản ánh bất bình đẳng lịch sử | CV tuyển dụng chủ yếu của nam → model ưu tiên nam |
| **Sampling bias** | Sample không đại diện cho population | Train nhận diện khuôn mặt chủ yếu trên người da trắng |
| **Measurement bias** | Cách đo lường không nhất quán giữa nhóm | Cảnh sát tuần tra nhiều ở khu A → bắt nhiều ở A → AI nghĩ A nguy hiểm hơn |
| **Aggregation bias** | Gộp các nhóm khác nhau làm một | Model tiểu đường train chung nam và nữ dù triệu chứng khác nhau |
| **Label bias** | Người gán nhãn có định kiến | Annotator gán nhãn "hung hăng" cho câu văn của người da màu nhiều hơn |
| **Confirmation bias** | Chỉ tìm data xác nhận hypothesis | Chỉ thu thập case study thành công của AI |
| **Automation bias** | Tin tưởng AI quá mức, không kiểm tra | Bác sĩ theo AI 100% mà không dùng clinical judgment |
| **Feedback loop bias** | Model tệ → quyết định tệ → data training xấu hơn | Recommender system → echo chamber → cực đoan hóa |

---

### 4.3 Giải pháp giảm Bias

**Ở giai đoạn Data:**

```python
# 1. Kiểm tra phân phối data theo nhóm
df.groupby("gender")["label"].value_counts(normalize=True)

# 2. Oversample minority (SMOTE)
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE().fit_resample(X, y)

# 3. Reweighting samples
sample_weights = compute_sample_weight("balanced", y)

# 4. Data augmentation cho nhóm thiểu số
# 5. Thu thập thêm data từ nhóm bị thiếu đại diện
```

**Ở giai đoạn Model:**

```python
# 1. Fairness constraint trong objective function
# Adversarial debiasing: thêm discriminator để penalize unfair predictions

# 2. Fairness-aware algorithms
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

mitigator = ExponentiatedGradient(estimator, constraints=DemographicParity())
mitigator.fit(X_train, y_train, sensitive_features=A_train)

# 3. Calibration theo nhóm
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(base_model, method="isotonic")
```

**Ở giai đoạn Evaluation:**

```python
from fairlearn.metrics import MetricFrame, demographic_parity_difference

# Tính metrics theo từng nhóm
metric_frame = MetricFrame(
    metrics={"accuracy": accuracy_score, "f1": f1_score},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test  # nhóm dân số (giới tính, chủng tộc...)
)

print(metric_frame.by_group)
# Xem accuracy và f1 của từng nhóm có chênh lệch không

# Demographic Parity Difference: càng gần 0 càng công bằng
dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=A_test)
```

---

### 4.4 Privacy-Preserving AI

#### Federated Learning

```
Thay vì gửi data về server trung tâm:
  1. Server gửi model hiện tại về từng thiết bị
  2. Mỗi thiết bị train trên data local của mình
  3. Thiết bị gửi gradient (không phải data) về server
  4. Server aggregate gradients → cập nhật model global
  5. Lặp lại

Ưu điểm: Data không rời thiết bị → bảo vệ privacy
Nhược điểm: Communication overhead, không giải quyết hoàn toàn (gradient có thể reveal data)

Ứng dụng: Google Keyboard (GBoard), FL cho y tế
```

#### Differential Privacy

```
Thêm nhiễu ngẫu nhiên vào data hoặc gradient:
  → Kết quả vẫn hữu ích ở mức aggregate
  → Không thể identify được cá nhân từ output

Định nghĩa toán học:
  Cơ chế M thỏa ε-differential privacy nếu:
  P(M(D) ∈ S) ≤ eᵉ × P(M(D') ∈ S)
  với mọi D, D' chỉ khác nhau 1 bản ghi, mọi tập output S

  ε nhỏ → privacy mạnh hơn, accuracy thấp hơn
  ε lớn → privacy yếu hơn, accuracy cao hơn

Ứng dụng: Apple (iOS telemetry), US Census Bureau
```

---

## 5. Hành vi & Làm việc nhóm

### 5.1 Giá trị Vingroup tìm kiếm

| Giá trị | Biểu hiện cụ thể |
|---------|-----------------|
| **Chủ động học hỏi** | Tự tìm tài liệu, không chờ được dạy; đọc paper mới, thử framework mới |
| **Tư duy data-driven** | Mọi quyết định cần số liệu; đặt câu hỏi "đo lường thế nào?" trước khi làm |
| **Growth mindset** | Xem thất bại là bài học; không sợ thử cái mới; nhận feedback cởi mở |
| **Collaboration** | Chia sẻ kiến thức; giúp đồng đội; không giữ thông tin |
| **Integrity** | Trung thực về giới hạn năng lực; không che giấu sai sót; giữ cam kết |
| **Impact-focused** | Làm việc hướng đến outcome thực tế, không chỉ output |
| **Bias for action** | Quyết định nhanh khi có đủ thông tin; tránh analysis paralysis |

---

### 5.2 Phương pháp STAR

```
Dùng để trả lời câu hỏi hành vi (behavioral questions):
"Hãy kể về một lần bạn..."

S — Situation (Bối cảnh):
    Mô tả ngữ cảnh, thời điểm, vai trò của bạn.
    "Trong dự án X tại công ty Y, tôi là..."

T — Task (Nhiệm vụ):
    Thách thức hoặc mục tiêu cần đạt được.
    "Tôi được giao nhiệm vụ... với deadline..."

A — Action (Hành động):
    Cụ thể BẠN đã làm gì. Dùng "Tôi" không phải "Chúng tôi".
    "Tôi đã phân tích..., quyết định..., thực hiện..."

R — Result (Kết quả):
    Kết quả cụ thể (số liệu nếu có). Bài học rút ra.
    "Kết quả: accuracy tăng từ 78% lên 91%. Tôi học được..."
```

---

### 5.3 Làm việc nhóm hiệu quả

**Khi bất đồng trong team:**

```
❌ Sai:
  - Im lặng đồng ý dù không đồng ý
  - Tấn công cá nhân thay vì góp ý về ý tưởng
  - Giữ bất đồng riêng, không nêu ra

✓ Đúng:
  - Nêu quan điểm rõ ràng, dùng data và lý lẽ
  - "Tôi thấy approach này có vấn đề ở X vì Y. Thay vào đó, Z có thể..."
  - Lắng nghe phản hồi, sẵn sàng thay đổi ý kiến nếu có lý do tốt
  - Disagree and commit: không đồng ý nhưng vẫn commit khi team quyết định
```

**Khi nhận feedback:**

```
❌ Sai:
  - Defensive ngay lập tức ("Nhưng mà...")
  - Im lặng không phản hồi
  - Nói "tôi biết rồi" nhưng không thay đổi

✓ Đúng:
  - Lắng nghe hết trước khi phản hồi
  - "Cảm ơn feedback. Để tôi hiểu rõ hơn, bạn có thể cho ví dụ cụ thể không?"
  - Xác nhận đã hiểu: "Ý bạn là... đúng không?"
  - Follow up sau khi thực hiện thay đổi
```

**Khi phát hiện sai sót của bản thân:**

```
✓ Đúng:
  1. Nhận trách nhiệm rõ ràng, không đổ lỗi
  2. Báo cáo sớm cho người liên quan
  3. Đề xuất phương án khắc phục cụ thể
  4. Thực hiện và báo cáo tiến độ
  5. Đảm bảo không lặp lại: document, process change
```

---

### 5.4 Giao tiếp kỹ thuật hiệu quả

**Giải thích kỹ thuật cho người không chuyên:**

```
❌ "Model của chúng ta dùng ensemble của gradient boosted trees
    với L2 regularization và SHAP-based feature importance."

✓ "Chúng ta đã xây dựng hệ thống dự đoán có thể cho biết
   khách hàng nào có khả năng rời bỏ, và giải thích lý do cụ thể
   (ví dụ: ít dùng app trong 30 ngày, số support ticket cao).
   Hệ thống chính xác 87%, giúp team sales ưu tiên đúng đối tượng."
```

**Pyramid Principle (khi trình bày):**

```
1. Kết luận/Đề xuất TRƯỚC (không build-up dài dòng)
2. Lý do hỗ trợ (3 điểm chính)
3. Data và bằng chứng cho từng lý do

Ví dụ:
  "Tôi đề xuất dùng XGBoost cho project này. [Kết luận]
   Vì 3 lý do:
   1. Accuracy cao nhất trong benchmark (F1 = 0.91 vs 0.85 của Random Forest)
   2. Training time chấp nhận được (< 5 phút trên dataset hiện tại)
   3. Feature importance dễ giải thích cho stakeholder
   [Chi tiết data benchmark đính kèm]"
```

---

## 6. Tình huống thực tế & Cách trả lời

### 6.1 Tình huống về Đạo đức AI

**Tình huống 1: Phát hiện bias trong model đang chạy production**

```
Đề: Model tín dụng của bạn đang hoạt động tốt (AUC = 0.89).
    Sau audit, phát hiện tỉ lệ chấp thuận khoản vay của phụ nữ
    thấp hơn nam giới 15% với cùng hồ sơ tài chính.
    PM muốn giữ nguyên vì "accuracy tổng thể vẫn cao". Bạn làm gì?

Trả lời tốt:
  1. Xác nhận vấn đề: 15% gap với cùng hồ sơ = rõ ràng là bias, không phải do
     khác biệt tài chính thực sự. Đây là vấn đề pháp lý (Equal Credit Opportunity Act)
     và đạo đức, không chỉ là vấn đề kỹ thuật.

  2. Không đồng ý với PM về "giữ nguyên":
     - Accuracy tổng thể cao không che giấu được disparate impact
     - Rủi ro pháp lý và reputational cho công ty
     - Đây là quyết định liên quan đến Fairness và Accountability

  3. Đề xuất cụ thể:
     - Tạm thời: thêm human review cho các case bị từ chối
     - Ngắn hạn: audit toàn bộ feature, loại bỏ proxy variables (zip code, tên...)
     - Dài hạn: retrain với fairness constraint (Equalized Odds), re-evaluate

  4. Escalate nếu PM không đồng ý: đây là vấn đề đủ nghiêm trọng để báo lên
     Chief AI Officer hoặc Legal team.
```

---

**Tình huống 2: Yêu cầu xây dựng hệ thống giám sát nhân viên**

```
Đề: Sếp yêu cầu bạn xây dựng AI theo dõi năng suất nhân viên
    bằng cách phân tích email, tin nhắn nội bộ, thời gian gõ phím,
    camera tại chỗ ngồi. Bạn xử lý thế nào?

Trả lời tốt:
  1. Làm rõ mục tiêu: "Mục tiêu cụ thể là gì? Tăng năng suất?
     Phát hiện nhân viên không làm việc? Giảm chi phí?"

  2. Nêu lo ngại cụ thể:
     - Privacy: giám sát email/tin nhắn cá nhân vượt quá mức cần thiết
     - Legal: nhiều quốc gia/tỉnh có luật giới hạn giám sát nhân viên
     - Chilling effect: nhân viên lo bị theo dõi → sáng tạo giảm, stress tăng
     - Trust: phá vỡ văn hóa công ty, tăng turnover

  3. Đề xuất alternatives:
     - Đo lường output thực sự (deliverables, quality of work) thay vì activity
     - Survey định kỳ về engagement và blockers
     - 1-on-1 check-ins thường xuyên hơn
     - OKR rõ ràng để nhân viên tự quản lý

  4. Nếu vẫn muốn dùng AI:
     - Chỉ track aggregated metrics (không individual), với sự đồng ý của nhân viên
     - Transparent: nhân viên biết gì đang được track
     - Không dùng camera hay private communications
```

---

**Tình huống 3: Model tốt trên test nhưng kém trong thực tế**

```
Đề: Model phát hiện fraud bạn build có AUC = 0.95 trên test set.
    Sau deploy 1 tháng, fraud team báo cáo model miss nhiều case mới.
    Nguyên nhân và giải pháp?

Trả lời tốt:
  Root cause analysis:
    - Data leakage trong training? → check feature engineering
    - Test set không đại diện? → check distribution
    - Concept drift: fraud patterns thay đổi theo thời gian? (99% khả năng)
    - Adversarial: fraudsters đã adapt để tránh model?

  Giải pháp ngắn hạn:
    - Tăng human review cho high-value transactions
    - Thu thập và label các case mới miss
    - Hạ ngưỡng (threshold) để tăng recall

  Giải pháp dài hạn:
    - Retrain với data mới hơn (sliding window)
    - Xây dựng monitoring dashboard theo dõi model drift
    - Thiết lập automated retraining trigger
    - Thêm rule-based layer để catch pattern fraud mới
    - Feedback loop: fraud team label → data → retrain
```

---

### 6.2 Tình huống về Hành vi & Teamwork

**Tình huống 4: Đồng nghiệp gian lận kết quả**

```
Đề: Bạn phát hiện đồng nghiệp sửa metric trong báo cáo
    để kết quả thực nghiệm trông tốt hơn thực tế.
    Sếp chưa biết. Đồng nghiệp đó là bạn thân của bạn. Bạn làm gì?

Trả lời tốt:
  1. Không im lặng: đây là vấn đề integrity, ảnh hưởng đến quyết định của team
     và công ty dựa trên thông tin sai.

  2. Nói chuyện riêng với đồng nghiệp trước:
     - Không buộc tội, mà hỏi thẳng: "Tôi thấy con số này khác với kết quả
       thực nghiệm. Có chuyện gì không?"
     - Giải thích tác hại: quyết định sai → lãng phí nguồn lực → ảnh hưởng cả team
     - Đưa ra deadline: "Mình cần report lại đúng số trước buổi meeting thứ Sáu"

  3. Nếu đồng nghiệp không sửa:
     - Báo cáo với team lead hoặc manager
     - Tập trung vào vấn đề (sai số liệu), không phải cá nhân
     - "Tôi phát hiện có sự khác biệt trong báo cáo vs kết quả thực,
        cần được làm rõ trước khi dùng để ra quyết định"

  4. Quan trọng: Quan hệ cá nhân không được phép ảnh hưởng đến integrity
     nghề nghiệp. Bảo vệ bạn bằng cách cho họ cơ hội tự sửa, không phải im lặng.
```

---

**Tình huống 5: Áp lực deadline vs chất lượng**

```
Đề: Còn 2 ngày là deadline demo cho khách hàng lớn.
    Bạn phát hiện model có bug quan trọng có thể gây kết quả sai
    trong một số edge case. Fix sẽ mất 3–4 ngày.
    PM nói "cứ demo trước, fix sau". Bạn làm gì?

Trả lời tốt:
  1. Đánh giá mức độ nghiêm trọng:
     - Edge case đó có khả năng xuất hiện trong demo không?
     - Nếu xảy ra, hậu quả với khách hàng thế nào?
     - Có workaround tạm thời không?

  2. Nếu rủi ro chấp nhận được:
     - Demo với disclaimer rõ ràng về limitation
     - Document bug, timeline fix, và cam kết cụ thể

  3. Nếu rủi ro không chấp nhận được:
     - Không demo feature có bug mà không thông báo
     - Escalate cho PM và stakeholder: "Tôi không thoải mái demo tính năng
       này vì [lý do cụ thể]. Tôi đề xuất: hoặc fix trước, hoặc demo phiên
       bản cũ hơn, hoặc thông báo rõ với khách hàng về limitation"
     - Đưa ra phương án thay thế, không chỉ "không được"

  Nguyên tắc: Không bao giờ mislead khách hàng. Có thể delay, có thể
  scope down, nhưng không được giả vờ sản phẩm tốt hơn thực tế.
```

---

**Tình huống 6: Conflict trong team về technical decision**

```
Đề: Team bạn tranh luận về việc dùng model A (accuracy cao, latency chậm)
    hay model B (accuracy thấp hơn một chút, latency nhanh hơn 3x).
    Mọi người không đồng thuận. Bạn là junior member. Bạn làm gì?

Trả lời tốt:
  1. Làm rõ tiêu chí quyết định trước:
     - "Chúng ta cần thống nhất: user experience hay accuracy quan trọng hơn
       với bài toán này? SLA về latency là gì?"

  2. Đề xuất framework quyết định dựa trên data:
     - A/B test cả 2 model với user thật, đo business metric
     - Hoặc: user study về acceptable latency threshold

  3. Chia sẻ quan điểm rõ ràng, dựa trên lý lẽ:
     - "Theo tôi, nếu latency > 2 giây là unacceptable với user (dựa trên
       research về UX), thì Model B là lựa chọn tốt hơn. Accuracy giảm 3%
       ít tác hại hơn là mất user vì chậm."

  4. Sẵn sàng defer cho người có thẩm quyền và kinh nghiệm hơn,
     nhưng vẫn ghi lại quan điểm của mình.

  5. Một khi team đã quyết định: commit 100%, không passive-aggressive.
```

---

## 📌 Bảng ôn nhanh trước khi thi

| Chủ đề | Điểm cần nhớ |
|--------|-------------|
| Deductive | Tiền đề đúng → kết luận chắc chắn đúng |
| Inductive | Quan sát → quy tắc chung (có thể sai) |
| Abductive | Chọn giải thích đơn giản nhất |
| Sunk cost | Không để chi phí đã mất ảnh hưởng quyết định tương lai |
| Correlation ≠ Causation | Cùng xảy ra ≠ cái này gây ra cái kia |
| 6 nguyên tắc AI Ethics | Fairness, Transparency, Accountability, Privacy, Safety, Human Oversight |
| Historical bias | Data lịch sử → model học lại bất bình đẳng |
| Demographic Parity | Tỉ lệ kết quả tích cực bằng nhau giữa các nhóm |
| Equalized Odds | TPR và FPR bằng nhau giữa các nhóm |
| SHAP | Contribution của từng feature vào dự đoán |
| Federated Learning | Train tại thiết bị, không gửi data về |
| Differential Privacy | Thêm nhiễu để không identify được cá nhân |
| STAR method | Situation → Task → Action → Result |
| Pyramid Principle | Kết luận trước, lý do sau, data cuối |
| Disagree & commit | Không đồng ý nhưng vẫn thực hiện khi team quyết định |

---

## ⚠️ Lưu ý khi làm bài

1. **Câu hỏi đạo đức không có đáp án duy nhất** — trình bày được trade-off, nhận diện được các bên bị ảnh hưởng, và đề xuất giải pháp có lý là quan trọng hơn chọn "đúng".

2. **Câu hỏi hành vi:** Luôn dùng ví dụ cụ thể (STAR method). Tránh câu trả lời lý thuyết chung chung.

3. **Khi bất đồng với người trên:** Nêu quan điểm rõ ràng, dùng data và lý lẽ, nhưng tôn trọng. Không im lặng, không passive-aggressive.

4. **Câu tình huống AI Ethics:** Luôn đề cập đến: (a) nhận diện vấn đề rõ ràng, (b) các bên bị ảnh hưởng, (c) giải pháp ngắn hạn và dài hạn, (d) khi nào cần escalate.

5. **Fairness:** Nhớ rằng các định nghĩa fairness có thể mâu thuẫn nhau — không có giải pháp hoàn hảo, cần chọn phù hợp với ngữ cảnh.

6. **Câu logic:** Đọc kỹ, kiểm tra điều kiện từng trường hợp, không đọc lướt.

---

*Module D — Tài liệu ôn thi Vingroup AI thực chiến 2026*
