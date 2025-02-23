import librosa
import numpy as np
import joblib
import pandas as pd

# Hàm trích xuất đặc trưng (tương tự File 1)
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return np.concatenate([mfccs, chroma, contrast, [zcr], [tempo]])

# Tải mô hình và LabelEncoder
model = joblib.load('model.pkl')
le = joblib.load('label_encoder.pkl')

# Đọc file mp3 mới
file_path = 'path_to_new_song.mp3'
features = extract_features(file_path)

# Dự đoán nhãn
features = features.reshape(1, -1)
pred = model.predict(features)
label = le.inverse_transform(pred)[0]

print(f"Thể loại nhạc dự đoán: {label}")