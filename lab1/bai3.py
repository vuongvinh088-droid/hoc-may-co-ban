from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression

# Bước 1: Load dữ liệu Iris
iris = load_iris()
X = iris.data  # Các đặc trưng (features)
y = iris.target  # Nhãn (target)

print(f"Kích thước dữ liệu X: {X.shape}")
print(f"Kích thước nhãn y: {y.shape}")

# Bước 2: Khởi tạo model (Sử dụng LinearRegression làm ví dụ theo yêu cầu)
model = LinearRegression()

# Bước 3: Fit model (Huấn luyện mô hình với dữ liệu)
model.fit(X, y)
print("Đã huấn luyện mô hình thành công!")

# Bước 4: Predict (Dự đoán thử với vài mẫu dữ liệu đầu tiên)
sample_data = X[:5]  # Lấy 5 mẫu đầu tiên
predictions = model.predict(sample_data)

print("Kết quả dự đoán của mô hình:", predictions)
print("Nhãn thực tế tương ứng:", y[:5])