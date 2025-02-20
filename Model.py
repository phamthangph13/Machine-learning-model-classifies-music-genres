from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
from extract import extract_features
# Tạo dataset
features = []
labels = []
genres = ['Pop', 'Rock', 'Rap', 'EDM', 'Country']

for genre in genres:
    for filename in os.listdir(f'dataset/{genre}'):
        if filename.endswith('.wav'):
            audio_path = f'dataset/{genre}/{filename}'
            feature_vector = extract_features(audio_path)
            features.append(feature_vector)
            labels.append(genre)

# Chuyển thành DataFrame
df = pd.DataFrame(features)
df['genre'] = labels

# Chia tập huấn luyện và kiểm thử
X = df.drop('genre', axis=1)
y = df['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Đánh giá mô hình
accuracy = model.score(X_test, y_test)
print(f"Độ chính xác: {accuracy:.2f}")


