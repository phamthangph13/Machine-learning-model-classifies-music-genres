import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Đọc dữ liệu từ CSV
df = pd.read_csv('data.csv')
X = df.drop('Label', axis=1)
y = df['Label']

# Mã hóa nhãn
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Huấn luyện mô hình Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Đánh giá mô hình
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Lưu mô hình và LabelEncoder
joblib.dump(model, 'model.pkl')
joblib.dump(le, 'label_encoder.pkl')