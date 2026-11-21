
# ==============================
# BÀI 1: XÂY DỰNG MÔ HÌNH LOGISTIC REGRESSION
# ==============================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load dữ liệu
data = load_breast_cancer()

X = data.data
y = data.target

# 2. Chia dữ liệu Train và Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Chuẩn hóa dữ liệu
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Huấn luyện mô hình Logistic Regression
model = LogisticRegression(max_iter=10000)

model.fit(X_train, y_train)

print("Đã huấn luyện mô hình thành công!")


# ==============================
# BÀI 2: ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI
# ==============================

# Dự đoán trên tập Test
y_pred = model.predict(X_test)

# Tính các chỉ số đánh giá
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n===== KẾT QUẢ ĐÁNH GIÁ =====")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)


# ==============================
# BÀI 3: MA TRẬN NHẦM LẪN
# ==============================

# Tạo ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)

print("\n===== CONFUSION MATRIX =====")
print(cm)

# Lấy các thành phần TN, FP, FN, TP
TN, FP, FN, TP = cm.ravel()

print("TN (True Negative):", TN)
print("FP (False Positive):", FP)
print("FN (False Negative):", FN)
print("TP (True Positive):", TP)

# Trực quan hóa ma trận nhầm lẫn
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)

display.plot(cmap="Blues")
plt.title("Confusion Matrix - Breast Cancer")
plt.show()
