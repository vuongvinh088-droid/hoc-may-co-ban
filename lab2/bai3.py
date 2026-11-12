from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Sử dụng tập dữ liệu diabetes từ sklearn.datasets
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# In ra kích thước của tập dữ liệu gốc
print(f'Kích thước dữ liệu gốc (X): {X.shape}')
print(f'Kích thước nhãn gốc (y): {y.shape}')

# Sử dụng hàm train_test_split để chia dữ liệu thành 2 tập: train (80%) và test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# In ra kích thước của hai tập sau khi chia
print('\n--- Sau khi chia Train/Test (80/20) ---')
print(f'Kích thước X_train: {X_train.shape}')
print(f'Kích thước X_test: {X_test.shape}')
print(f'Kích thước y_train: {y_train.shape}')
print(f'Kích thước y_test: {y_test.shape}')