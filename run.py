import os
import numpy as np
import pandas as pd
import joblib
import librosa
import argparse

def extract_features(file_path):
    """Trích xuất đặc trưng âm thanh từ file audio"""
    try:
        # Tải file âm thanh
        y, sr = librosa.load(file_path, duration=30)
        
        # Trích xuất MFCC (20 đặc trưng)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
        
        # Trích xuất chroma (12 đặc trưng)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        
        # Trích xuất spectral contrast (7 đặc trưng)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        
        # Trích xuất zero crossing rate (1 đặc trưng)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Trích xuất tempo (1 đặc trưng)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # Chuyển tempo thành giá trị scalar để tránh cảnh báo DeprecationWarning
        tempo = float(np.array(tempo).item())
        
        # Kết hợp các đặc trưng thành vector 41 phần tử:
        # 20 (MFCC) + 12 (Chroma) + 7 (Spectral Contrast) + 1 (ZCR) + 1 (Tempo) = 41
        features = np.concatenate([
            mfccs,
            chroma,
            contrast,
            np.array([zcr]),
            np.array([tempo])
        ])
        
        return features
    
    except Exception as e:
        print(f"Lỗi khi trích xuất đặc trưng từ {file_path}: {str(e)}")
        return None

def predict_genre(file_path, model_type='RF'):
    """Dự đoán thể loại nhạc từ file âm thanh"""
    # Xác định đường dẫn mô hình và encoder (nếu có) dựa vào loại mô hình
    if model_type.upper() == 'RF':
        model_path = 'report/RF/rf_model.pkl'
        encoder_path = 'report/RF/rf_label_encoder.pkl'
        scaler_path = None
    elif model_type.upper() == 'LR':
        model_path = 'report/LR/lr_model.pkl'
        # Với LR, không có file label encoder nên dùng mapping dictionary
        encoder_path = None  
        scaler_path = 'report/LR/lr_scaler.pkl'
    else:
        raise ValueError("model_type phải là 'RF' hoặc 'LR'")
    
    # Kiểm tra file mô hình có tồn tại không
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}")
    
    # Với RF, kiểm tra file encoder
    if model_type.upper() == 'RF' and (encoder_path is None or not os.path.exists(encoder_path)):
        raise FileNotFoundError(f"Không tìm thấy file encoder: {encoder_path}")
    
    # Tải mô hình
    model = joblib.load(model_path)
    
    # Với RF, load label encoder; với LR, định nghĩa mapping dictionary
    if model_type.upper() == 'RF':
        label_encoder = joblib.load(encoder_path)
    else:
        # Mapping dictionary cho mô hình LR - chỉnh sửa theo dữ liệu huấn luyện của bạn
        label_mapping = {
             0: 'Acoustic',
             1: 'Country',
             2: 'EDM',
             3: 'Rap',
             4: 'Rock'
            }
    
    # Tải scaler nếu sử dụng cho mô hình LR
    scaler = None
    if model_type.upper() == 'LR' and scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    # Trích xuất đặc trưng từ file âm thanh
    features = extract_features(file_path)
    if features is None:
        return None
    
    # Định dạng lại đặc trưng thành vector 2 chiều
    features = features.reshape(1, -1)
    if scaler is not None:
        features = scaler.transform(features)
    
    # Dự đoán
    if model_type.upper() == 'RF':
        prediction = model.predict(features)
    else:
        # Với LR, làm tròn kết quả dự đoán thành số nguyên
        prediction = np.round(model.predict(features)).astype(int)
    
    # Chuyển đổi dự đoán thành nhãn
    if model_type.upper() == 'RF':
        genre = label_encoder.inverse_transform(prediction)[0]
    else:
        genre = label_mapping.get(prediction[0], "Unknown")
    
    # Tính độ tin cậy nếu mô hình hỗ trợ (chỉ với RF)
    confidence = None
    if model_type.upper() == 'RF' and hasattr(model, 'predict_proba'):
        confidence = np.max(model.predict_proba(features)) * 100
    
    return {
        'genre': genre,
        'confidence': confidence,
        'model_type': model_type
    }

def main():
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Dự đoán thể loại nhạc từ file âm thanh')
    parser.add_argument('file_path', type=str, help='Đường dẫn đến file âm thanh')
    parser.add_argument('--model', type=str, choices=['RF', 'LR'], default='LR',
                        help='Loại mô hình để sử dụng (RF: Random Forest, LR: Linear Regression)')
    args = parser.parse_args()
    
    # Dự đoán thể loại nhạc
    result = predict_genre(args.file_path, args.model)
    
    # In kết quả dự đoán
    if result:
        print("\n==== KẾT QUẢ DỰ ĐOÁN ====")
        print(f"File: {args.file_path}")
        print(f"Thể loại nhạc: {result['genre']}")
        if result['confidence'] is not None:
            print(f"Độ tin cậy: {result['confidence']:.2f}%")
        print(f"Mô hình: {result['model_type']}")
    else:
        print("Không thể dự đoán thể loại nhạc từ file này.")

if __name__ == "__main__":
    main()
