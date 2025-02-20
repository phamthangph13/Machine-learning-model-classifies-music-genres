import librosa
import numpy as np
import os

def extract_features(audio_path):
    # Đọc file âm thanh
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Trích xuất đặc trưng
    # 1. Đặc trưng phổ (Spectral Features)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)  # Shape: (13,)
    
    # 2. Đặc trưng nhịp điệu (Rhythmic Features)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # 3. Đặc trưng năng lượng (Energy Features)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    # 4. Đặc trưng hòa âm (Harmonic Features)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)  # Shape: (12,)
    
    # Ghép các đặc trưng - đảm bảo mỗi giá trị là mảng 1D
    features = np.concatenate([
        mfcc_mean,
        np.array([tempo]),  # Chuyển số vô hướng thành mảng 1D
        np.array([spectral_centroid]),  # Chuyển số vô hướng thành mảng 1D
        chroma
    ])
    
    # Debug: In ra kích thước của mỗi phần
    print(f"mfcc_mean shape: {mfcc_mean.shape}")
    print(f"tempo array shape: {np.array([tempo]).shape}")
    print(f"spectral_centroid array shape: {np.array([spectral_centroid]).shape}")
    print(f"chroma shape: {chroma.shape}")
    
    return features


extract_features("data.mp3")