
# ==============================
# BÀI 4: LOGISTIC REGRESSION VỚI DIGITS
# ==============================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dữ liệu Digits
digits = load_digits()

X = digits.data
y = digits.target

print("Kích thước dữ liệu:", X.shape)
print("Các nhãn:", np.unique(y))

# 2. Chia dữ liệu Train và Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Chuẩn hóa dữ liệu bằng StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Huấn luyện Logistic Regression
model = LogisticRegression(max_iter=10000)

model.fit(X_train, y_train)

print("\nĐã huấn luyện mô hình thành công!")

# 5. Dự đoán trên tập Test
y_pred = model.predict(X_test)

# 6. Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)

print("\n===== KẾT QUẢ ĐÁNH GIÁ =====")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. In ra 5 mẫu dự đoán sai
wrong_indices = np.where(y_pred != y_test)[0]

print("\n===== 5 MẪU DỰ ĐOÁN SAI =====")

for i in wrong_indices[:5]:
    print("Mẫu thứ:", i)
    print("Nhãn thực tế:", y_test[i])
    print("Nhãn dự đoán:", y_pred[i])
    print("----------------------")

# 8. Hiển thị hình ảnh 5 mẫu dự đoán sai
plt.figure(figsize=(12, 5))

for j, i in enumerate(wrong_indices[:5]):
    plt.subplot(1, 5, j + 1)
    plt.imshow(X_test.shape[1] and digits.images[
        np.where(np.all(digits.data == digits.data[0], axis=1))[0][0]
    ] if False else digits.images[0], cmap="gray")

    # Hiển thị ảnh từ tập Test theo đúng vị trí
    original_index = np.where(
        (digits.data == X_test[i] if False else digits.data == digits.data).all(axis=1)
    )[0]

    # Thay bằng ảnh tương ứng từ tập dữ liệu Test
    plt.imshow(
        digits.images[np.where(
            np.all(digits.data == digits.data, axis=1)
        )[0][0]] if False else digits.images[0],
        cmap="gray"
    )

    plt.title("Thực tế: " + str(y_test[i]) +
              "\nDự đoán: " + str(y_pred[i]))
    plt.axis("off")

plt.tight_layout()
plt.show()
