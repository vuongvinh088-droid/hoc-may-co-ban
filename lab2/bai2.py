import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Tạo mảng dữ liệu với các cột có đơn vị khác nhau
data = {
    'Tuoi': [20, 30, 40, 50, 60],
    'ThuNhap': [10000000, 30000000, 50000000, 75000000, 100000000],
}
df = pd.DataFrame(data)

print('--- Dữ liệu gốc ---')
print(df)

# Sử dụng StandardScaler (Chuẩn hóa Z-score)
scaler_standard = StandardScaler()
scaled_standard = scaler_standard.fit_transform(df)
df_standard = pd.DataFrame(scaled_standard, columns=['Tuoi_std', 'ThuNhap_std'])

# Sử dụng MinMaxScaler (Chuẩn hóa Min-Max về khoảng [0, 1])
scaler_minmax = MinMaxScaler()
scaled_minmax = scaler_minmax.fit_transform(df)
df_minmax = pd.DataFrame(
    scaled_minmax, columns=['Tuoi_minmax', 'ThuNhap_minmax']
)

# In kết quả sau khi chuẩn hóa để so sánh
print('\n--- Kết quả sau khi dùng StandardScaler ---')
print(df_standard)

print('\n--- Kết quả sau khi dùng MinMaxScaler ---')
print(df_minmax)