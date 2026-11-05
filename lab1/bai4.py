import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# 1. Tải tập dữ liệu digits từ sklearn
digits = load_digits()

X = digits.data  # Dữ liệu ảnh dạng mảng 1 chiều (64 pixels)
y = digits.target  # Nhãn chữ số tương ứng

# 2. In ra hình dạng (shape) của dữ liệu và nhãn
print(f"Hình dạng của dữ liệu (X shape): {X.shape}")
print(f"Hình dạng của nhãn (y shape): {y.shape}")

# 3. Hiển thị giá trị của mẫu đầu tiên và vẽ trực quan hóa dạng hình ảnh
print(f"Nhãn của mẫu đầu tiên: {y[0]}")
print("Giá trị mảng pixels của mẫu đầu tiên:\n", X[0])

# Vẽ hình ảnh mẫu đầu tiên (chuyển đổi lại từ mảng 1D 64 phần tử về ma trận 8x8)
plt.figure(figsize=(3, 3))
plt.imshow(digits.images[0], cmap='gray')
plt.title(f"Nhãn: {y[0]}")
plt.axis('off')
plt.show()