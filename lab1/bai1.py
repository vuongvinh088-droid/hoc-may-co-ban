import numpy as np
import pandas as pd

# 1. Tạo mảng Numpy 1 chiều chứa các số từ 1 đến 10
arr = np.arange(1, 11)
print("Mảng Numpy từ 1 đến 10:")
print(arr)
print("-" * 40)

# 2. Tạo DataFrame chứa thông tin của 5 sinh viên
data = {
    'Họ tên': ['Nguyễn Văn A', 'Trần Thị B', 'Lê Văn C', 'Phạm Thị D', 'Hoàng Văn E'],
    'Tuổi': [19, 20, 19, 21, 20],
    'Điểm toán': [8.5, 7.0, 9.0, 6.5, 8.0]
}

df = pd.DataFrame(data)
print("DataFrame thông tin 5 sinh viên:")
print(df)
print("-" * 40)

# 3. Hiển thị thông tin tổng quan và tính điểm trung bình môn toán
print("Thông tin tổng quan của DataFrame (info):")
df.info()
print("-" * 40)

diem_tb_toan = df['Điểm toán'].mean()
print(f"Điểm trung bình môn toán của 5 sinh viên là: {diem_tb_toan}")