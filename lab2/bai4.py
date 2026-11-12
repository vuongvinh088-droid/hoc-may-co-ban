from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Tiếp tục với tập dữ liệu digits
digits = load_digits()
X, y = digits.data, digits.target

print(f'Dữ liệu gốc digits - X shape: {X.shape}, y shape: {y.shape}')

# 2. Chia tập dữ liệu thành Train và Test (ví dụ: 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Xây dựng một Pipeline đơn giản chứa bước chuẩn hóa (StandardScaler) và model (SVM)
# Vì giá trị pixel nằm trong khoảng 0-16, StandardScaler giúp chuẩn hóa về phân phối chuẩn phù hợp cho SVM
pipeline = Pipeline(
    [
        ('scaler', StandardScaler()),  # Bước 1: Chuẩn hóa dữ liệu
        (
            'model',
            SVC(kernel='rbf', random_state=42),
        ),  # Bước 2: Mô hình SVM
    ]
)

# Huấn luyện mô hình thông qua Pipeline
pipeline.fit(X_train, y_train)

# Đánh giá độ chính xác trên tập test
accuracy = pipeline.score(X_test, y_test)
print(f'\nĐộ chính xác của Pipeline (Scaler + SVM) trên tập Test: {accuracy * 100:.2f}%')