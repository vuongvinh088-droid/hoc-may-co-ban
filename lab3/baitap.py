
# ==========================================
# BÀI TẬP: HỌC MÁY CƠ BẢN - LINEAR REGRESSION
# ==========================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ==========================================
# BÀI 1: XÂY DỰNG MÔ HÌNH HỒI QUY TUYẾN TÍNH
# ==========================================

def bai1():
    print("\n========== BÀI 1 ==========")

    # Tạo dữ liệu giả lập: y = 2X + 1
    X = np.array([[1], [2], [3], [4], [5],
                  [6], [7], [8], [9], [10]])

    y = 2 * X.flatten() + 1

    # Tạo mô hình hồi quy tuyến tính
    model = LinearRegression()

    # Huấn luyện mô hình
    model.fit(X, y)

    # In hệ số của mô hình
    print("Coefficient:", model.coef_[0])
    print("Intercept:", model.intercept_)

    # Dự đoán thử
    print("Dự đoán khi X = 11:", model.predict([[11]])[0])


# ==========================================
# BÀI 2: ĐÁNH GIÁ MÔ HÌNH
# ==========================================

def bai2():
    print("\n========== BÀI 2 ==========")

    # Tải tập dữ liệu Diabetes
    diabetes = load_diabetes()

    # Chỉ lấy 1 đặc trưng duy nhất (BMI)
    X = diabetes.data[:, 2].reshape(-1, 1)

    # Biến mục tiêu
    y = diabetes.target

    # Chia dữ liệu Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Tạo và huấn luyện mô hình
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Dự đoán trên tập Test
    y_pred = model.predict(X_test)

    # Tính các chỉ số đánh giá
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MAE:", mae)
    print("MSE:", mse)
    print("R-squared:", r2)

    # Giải thích:
    # MAE: Sai số tuyệt đối trung bình giữa giá trị thực và dự đoán.
    # MSE: Trung bình bình phương sai số, phạt nặng các sai số lớn.
    # R-squared: Đo mức độ mô hình giải thích được sự biến thiên của dữ liệu.
    # R2 càng gần 1 thì mô hình càng phù hợp với dữ liệu.

    return X_train, X_test, y_train, y_test, model


# ==========================================
# BÀI 3: TRỰC QUAN HÓA KẾT QUẢ
# ==========================================

def bai3():
    print("\n========== BÀI 3 ==========")

    # Lấy dữ liệu Diabetes
    diabetes = load_diabetes()

    # Chỉ sử dụng một đặc trưng BMI
    X = diabetes.data[:, 2].reshape(-1, 1)
    y = diabetes.target

    # Chia dữ liệu Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Huấn luyện mô hình
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Sắp xếp dữ liệu để vẽ đường hồi quy
    X_line = np.linspace(
        X_test.min(), X_test.max(), 100
    ).reshape(-1, 1)

    y_line = model.predict(X_line)

    # Vẽ biểu đồ
    plt.figure(figsize=(8, 5))

    # Biểu đồ phân tán dữ liệu thực tế
    plt.scatter(X_test, y_test, color="blue",
                label="Du lieu thuc te")

    # Vẽ đường hồi quy
    plt.plot(X_line, y_line, color="red",
             label="Duong hoi quy")

    plt.xlabel("BMI")
    plt.ylabel("Diabetes Progression")
    plt.title("Bieu do hoi quy tuyen tinh")
    plt.legend()
    plt.grid(True)
    plt.show()


# ==========================================
# BÀI 4: QUY TRÌNH ML HOÀN CHỈNH
# ==========================================

def bai4():
    print("\n========== BÀI 4 ==========")

    # Tải dữ liệu California Housing
    try:
        housing = fetch_california_housing()

        X = housing.data
        y = housing.target

        print("Su dung bo du lieu California Housing")

    except Exception:
        # Nếu không tải được California Housing,
        # sử dụng Diabetes có sẵn trong sklearn
        print("Khong tai duoc California Housing.")
        print("Chuyen sang bo du lieu Diabetes.")

        diabetes = load_diabetes()
        X = diabetes.data
        y = diabetes.target

    # Chia dữ liệu Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Tiền xử lý dữ liệu bằng StandardScaler
    # Sau đó huấn luyện mô hình Linear Regression
    model = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )

    # Huấn luyện mô hình
    model.fit(X_train, y_train)

    # Dự đoán
    y_pred = model.predict(X_test)

    # Đánh giá mô hình bằng MSE
    mse = mean_squared_error(y_test, y_pred)

    print("Mean Squared Error (MSE):", mse)

    # Trực quan hóa một phần kết quả
    plt.figure(figsize=(8, 5))

    # Lấy 100 mẫu đầu tiên để hiển thị
    plt.scatter(y_test[:100], y_pred[:100],
                color="blue", label="Gia tri du doan")

    # Đường dự đoán hoàn hảo
    min_value = min(y_test[:100].min(), y_pred[:100].min())
    max_value = max(y_test[:100].max(), y_pred[:100].max())

    plt.plot([min_value, max_value],
             [min_value, max_value],
             color="red", label="Duong du doan hoan hao")

    plt.xlabel("Gia tri thuc te")
    plt.ylabel("Gia tri du doan")
    plt.title("So sanh gia tri thuc te va du doan")
    plt.legend()
    plt.grid(True)
    plt.show()


# ==========================================
# CHẠY TẤT CẢ CÁC BÀI
# ==========================================

if __name__ == "__main__":
    bai1()
    bai2()
    bai3()
    bai4()
