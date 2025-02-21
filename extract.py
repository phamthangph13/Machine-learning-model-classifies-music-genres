import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Đọc dữ liệu từ file CSV
data = pd.read_csv("music_features.csv")

# In ra danh sách các cột và vài dòng đầu của dữ liệu
print("Các cột trong dữ liệu:", data.columns.tolist())
print("Dữ liệu mẫu:")
print(data.head())

# 2. Kiểm tra xem cột 'genre' có tồn tại hay không
if 'genre' in data.columns:
    # Ví dụ: vẽ boxplot của đặc trưng mfcc_0 theo từng genre
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='genre', y='mfcc_0', data=data)
    plt.title("Phân bố của mfcc_0 theo thể loại (genre)")
    plt.xlabel("Thể loại")
    plt.ylabel("Giá trị mfcc_0")
    plt.show()
else:
    print("Không tìm thấy cột 'genre' trong dữ liệu.")
    # Loại bỏ cột file_name nếu tồn tại
    features = data.drop(columns=['file_name'], errors='ignore')
    # Chỉ lấy các cột số (numeric) để tính ma trận tương quan
    features_numeric = features.select_dtypes(include=['number'])
    
    print("Các cột số được sử dụng cho ma trận tương quan:", features_numeric.columns.tolist())
    
    correlation_matrix = features_numeric.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Ma trận tương quan của các đặc trưng số")
    plt.show()
