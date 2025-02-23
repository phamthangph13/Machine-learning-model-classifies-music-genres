import os
import librosa
import pandas as pd
import numpy as np

# Đường dẫn đến các folder chứa file mp3
genres = ['Acoustic', 'Country', 'EDM', 'Rap', 'Rock']
data = []

# Hàm trích xuất đặc trưng từ một file mp3
def extract_features(file_path, label):
    y, sr = librosa.load(file_path)
    
    # MFCCs
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
    # Chroma features
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    # Spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    # Zero-crossing rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    features = np.concatenate([mfccs, chroma, contrast, [zcr], [tempo]])
    return np.append(features, label)

# Duyệt qua các folder và file mp3
for genre in genres:
    folder_path = f'path_to_dataset/{genre}'
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mp3'):
            file_path = os.path.join(folder_path, file_name)
            features = extract_features(file_path, genre)
            data.append(features)

# Tạo DataFrame và lưu vào CSV
columns = [f'MFCC_{i+1}' for i in range(20)] + \
          [f'Chroma_{i+1}' for i in range(12)] + \
          [f'Spectral_Contrast_{i+1}' for i in range(7)] + \
          ['Zero_Crossing_Rate', 'Tempo', 'Label']
df = pd.DataFrame(data, columns=columns)
df.to_csv('data.csv', index=False)


