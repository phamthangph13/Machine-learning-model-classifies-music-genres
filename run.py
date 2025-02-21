import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Đọc dữ liệu từ file CSV
data = pd.read_csv('music_features.csv')

# 2. Khám phá dữ liệu
print("Thông tin dữ liệu:")
print(data.info())
print("\nThống kê mô tả:")
print(data.describe())

# 3. Vẽ histogram cho các cột số
data.hist(bins=30, figsize=(15,10))
plt.tight_layout()
plt.show()

# 4. Vẽ Boxplot để phát hiện outliers
plt.figure(figsize=(12,6))
sns.boxplot(data=data)
plt.xticks(rotation=90)
plt.title("Boxplot của các biến")
plt.show()

# 5. Vẽ Heatmap của ma trận tương quan
plt.figure(figsize=(10,8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Ma trận tương quan")
plt.show()
