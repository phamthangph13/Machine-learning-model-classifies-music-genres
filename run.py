from extract import extract_features
from Model import model

def predict_genre(audio_path, model):
    # Trích xuất đặc trưng từ file âm thanh
    features = extract_features(audio_path)
    
    # Dự đoán thể loại
    genre = model.predict([features])[0]
    
    return genre

# Sử dụng
song_path = "unknown_song.mp3"
predicted_genre = predict_genre(song_path, model)
print(f"Thể loại dự đoán: {predicted_genre}")