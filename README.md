Dưới đây là câu trả lời hoàn chỉnh cho yêu cầu của bạn, với việc bổ sung các ảnh minh họa và icon để làm cho nội dung dễ hiểu và hấp dẫn hơn. Tôi sẽ sử dụng markdown để định dạng rõ ràng và trực quan.

---

# Machine Learning Model for Music Genre Classification

## Giới thiệu
Dự án **Machine-learning-model-classifies-music-genres** là một hệ thống phân loại thể loại nhạc tự động sử dụng học máy. Hệ thống trích xuất đặc trưng âm thanh từ các file MP3, sau đó áp dụng hai mô hình học máy - **Random Forest** và **Linear Regression (Ridge)** - để dự đoán thể loại nhạc. Các thể loại nhạc được hỗ trợ bao gồm: **Acoustic**, **Country**, **EDM**, **Rap**, và **Rock**.

![Music Genres](https://images.pexels.com/photos/164660/pexels-photo-164660.jpeg?auto=compress&cs=tinysrgb&w=600)  
*Hình 1: Một số thể loại nhạc phổ biến được phân loại trong dự án*

---

## Cấu trúc dự án
Dự án bao gồm 3 file Python chính và 1 file dữ liệu mẫu:

1. **`data_preparation.py`**  
   📥 Trích xuất đặc trưng âm thanh từ file MP3 và lưu vào `data.csv`.

2. **`model_training.py`**  
   🛠️ Huấn luyện và đánh giá hai mô hình học máy, lưu kết quả vào thư mục `report`.

3. **`predict_genre.py`**  
   🔮 Dự đoán thể loại nhạc từ file MP3 mới bằng mô hình đã huấn luyện.

4. **`data.csv`**  
   📊 File chứa đặc trưng âm thanh và nhãn thể loại nhạc mẫu.

---

## Yêu cầu cài đặt
### Thư viện cần thiết
Cài đặt các thư viện Python sau bằng lệnh:
```bash
pip install pandas numpy sklearn librosa matplotlib seaborn scipy joblib argparse
```

### Yêu cầu phần cứng/phần mềm
- 🐍 **Python**: Phiên bản 3.7 trở lên.
- 💻 **Hệ điều hành**: Windows, macOS, hoặc Linux.
- 🧠 **RAM**: Tối thiểu 4GB (khuyến nghị 8GB).
- 💾 **Dung lượng ổ cứng**: Ít nhất 1GB.

---

## Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu
- **Bước 1**: Tạo thư mục chứa file MP3 theo thể loại (ví dụ: `path_to_dataset/Acoustic`, `path_to_dataset/Rock`).
- **Bước 2**: Chỉnh sửa đường dẫn trong `data_preparation.py`:
  ```python
  folder_path = f'path_to_dataset/{genre}'
  ```
- **Bước 3**: Chạy lệnh để tạo `data.csv`:
  ```bash
  python data_preparation.py
  ```

### 2. Huấn luyện mô hình
- Đảm bảo file `data.csv` đã sẵn sàng.
- Chạy lệnh:
  ```bash
  python model_training.py
  ```
- **Kết quả**:  
  - Mô hình được lưu trong `report/RF` (Random Forest) và `report/LR` (Linear Regression).  
  - Báo cáo đánh giá và biểu đồ (confusion matrix, feature importance) được lưu dưới dạng `.txt` và `.png`.  

![Model Training](https://images.pexels.com/photos/270348/pexels-photo-270348.jpeg?auto=compress&cs=tinysrgb&w=600)  
*Hình 2: Quá trình huấn luyện mô hình học máy*

### 3. Dự đoán thể loại nhạc
- Chạy lệnh để dự đoán từ file MP3 mới:
  ```bash
  python predict_genre.py <đường_dẫn_file_mp3> --model <RF hoặc LR>
  ```
  - Ví dụ:
    ```bash
    python predict_genre.py "path/to/song.mp3" --model RF
    ```
- **Kết quả đầu ra**:  
  - 🎵 Thể loại nhạc dự đoán.  
  - ✅ Độ tin cậy (confidence score) nếu dùng Random Forest.

![Prediction](https://images.pexels.com/photos/159376/earphones-ipod-music-earbuds-159376.jpeg?auto=compress&cs=tinysrgb&w=600)  
*Hình 3: Dự đoán thể loại nhạc từ file MP3 mới*

---

## Chi tiết kỹ thuật

### Đặc trưng âm thanh
Hệ thống trích xuất **41 đặc trưng** từ mỗi file MP3:
- 🎤 **MFCCs (20 đặc trưng)**: Hệ số phổ âm thanh.
- 🎹 **Chroma (12 đặc trưng)**: Năng lượng của 12 nốt nhạc.
- 🌈 **Spectral Contrast (7 đặc trưng)**: Độ tương phản phổ.
- ⚡ **Zero Crossing Rate (1 đặc trưng)**: Tỷ lệ vượt qua 0.
- ⏱️ **Tempo (1 đặc trưng)**: Nhịp độ bài hát.

### Mô hình học máy
1. **Random Forest** 🌳  
   - Tối ưu tham số bằng `GridSearchCV`.  
   - Đánh giá: Độ chính xác, ma trận nhầm lẫn.

2. **Linear Regression (Ridge)** 📈  
   - Tối ưu tham số `alpha`.  
   - Đánh giá: RMSE, MAE, R² Score.

---

## Hạn chế và cải tiến
### Hạn chế
- ⚠️ Chỉ hỗ trợ 5 thể loại nhạc cố định.
- 🎧 Yêu cầu file MP3 chất lượng cao.
- ⏳ Thời gian xử lý lâu với dữ liệu lớn.

### Cải tiến tiềm năng
- ➕ Thêm thể loại nhạc mới.
- 🚀 Tối ưu bằng xử lý song song.
- 🧠 Dùng mô hình học sâu (CNN, RNN).

---

## Kết luận
Dự án này cung cấp một giải pháp đơn giản nhưng hiệu quả để phân loại thể loại nhạc bằng học máy. Với các ảnh minh họa và icon, hy vọng hướng dẫn này dễ theo dõi hơn! Nếu cần hỗ trợ, hãy liên hệ qua email: [winnieph13@gmail.com].

--- 

Hy vọng câu trả lời này đáp ứng yêu cầu của bạn!