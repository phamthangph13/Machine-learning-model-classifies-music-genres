import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# ----------------- Random Forest Model -----------------
class RandomForestModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.best_params = None
        
    def prepare_data(self, data_path):
        """Đọc và chuẩn bị dữ liệu cho Random Forest"""
        print("1. Đang chuẩn bị dữ liệu cho Random Forest...")
        df = pd.read_csv(data_path)
        X = df.drop('Label', axis=1)
        y = df['Label']
        y_encoded = self.label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=0.2, 
            random_state=42,
            stratify=y_encoded
        )
        return X_train, X_test, y_train, y_test
    
    def optimize_parameters(self, X_train, y_train):
        """Tối ưu hóa tham số sử dụng GridSearchCV"""
        print("2. Đang tối ưu hóa tham số cho Random Forest...")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=2,
            scoring='accuracy'
        )
        grid_search.fit(X_train, y_train)
        self.best_params = grid_search.best_params_
        print("\nTham số tối ưu:", self.best_params)
        print("Độ chính xác tốt nhất:", grid_search.best_score_)
        return grid_search.best_estimator_
    
    def train(self, X_train, y_train, optimize=True):
        """Huấn luyện mô hình Random Forest"""
        print("3. Đang huấn luyện mô hình Random Forest...")
        if optimize:
            self.model = self.optimize_parameters(X_train, y_train)
        else:
            self.model = RandomForestClassifier(random_state=42)
            self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """Đánh giá mô hình Random Forest và lưu kết quả vào folder RF trong report"""
        print("4. Đang đánh giá mô hình Random Forest...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(
            y_test, 
            y_pred, 
            target_names=self.label_encoder.classes_
        )
        print("\nĐộ chính xác:", accuracy)
        print("\nBáo cáo phân loại:\n", class_report)
        
        # Lưu kết quả đánh giá vào file report cho RF
        rf_report_dir = "report/RF"
        with open(os.path.join(rf_report_dir, "rf_evaluation_report.txt"), "w", encoding="utf-8") as f:
            f.write("Độ chính xác: " + str(accuracy) + "\n")
            f.write("Báo cáo phân loại:\n" + class_report)
        
        # Vẽ và lưu confusion matrix cho RF
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Ma trận nhầm lẫn - Random Forest')
        plt.ylabel('Nhãn thực tế')
        plt.xlabel('Nhãn dự đoán')
        plt.savefig(os.path.join(rf_report_dir, "rf_confusion_matrix.png"))
        plt.close()
        
        # Vẽ và lưu feature importance cho RF
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 đặc trưng quan trọng nhất - Random Forest')
        plt.savefig(os.path.join(rf_report_dir, "rf_feature_importance.png"))
        plt.close()
        
        return accuracy, conf_matrix, class_report
    
    def save_model(self, model_path='report/RF/rf_model.pkl', encoder_path='report/RF/rf_label_encoder.pkl'):
        """Lưu mô hình và encoder vào folder RF trong report"""
        print("5. Đang lưu mô hình Random Forest...")
        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, encoder_path)
        if self.best_params:
            with open('report/RF/rf_best_params.txt', 'w', encoding="utf-8") as f:
                f.write(str(self.best_params))

# ----------------- Linear Regression Model -----------------
class LinearRegressionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()

    def prepare_data(self, data_path):
        """Đọc và chuẩn bị dữ liệu cho Linear Regression với chuẩn hóa dữ liệu và mã hóa nhãn"""
        print("1. Đang chuẩn bị dữ liệu cho Linear Regression...")
        df = pd.read_csv(data_path)
        X = df.drop('Label', axis=1)
        y = df['Label']
        self.feature_names = X.columns
        y_encoded = self.label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

    def optimize_parameters(self, X_train, y_train):
        """Tối ưu hóa tham số regularization cho Linear Regression (sử dụng Ridge)"""
        print("2. Đang tối ưu hóa tham số regularization cho Linear Regression...")
        param_grid = {
            'alpha': [0.01, 0.1, 1, 10, 100]
        }
        ridge = Ridge(random_state=42)
        grid_search = GridSearchCV(
            estimator=ridge,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=2,
            scoring='neg_mean_squared_error'
        )
        grid_search.fit(X_train, y_train)
        self.best_params = grid_search.best_params_
        print("\nTham số tối ưu:", self.best_params)
        print("MSE tốt nhất (âm):", grid_search.best_score_)
        return grid_search.best_estimator_

    def train(self, X_train, y_train, optimize=True):
        """Huấn luyện mô hình Linear Regression"""
        print("3. Đang huấn luyện mô hình Linear Regression...")
        if optimize:
            self.model = self.optimize_parameters(X_train, y_train)
        else:
            self.model = Ridge(random_state=42)
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """Đánh giá mô hình Linear Regression và lưu kết quả vào folder LR trong report"""
        print("4. Đang đánh giá mô hình Linear Regression...")
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print("\nRMSE:", rmse)
        print("MAE:", mae)
        print("R2 Score:", r2)
        
        lr_report_dir = "report/LR"
        # Lưu kết quả đánh giá vào file report cho LR
        with open(os.path.join(lr_report_dir, "lr_evaluation_report.txt"), "w", encoding="utf-8") as f:
            f.write("RMSE: " + str(rmse) + "\n")
            f.write("MAE: " + str(mae) + "\n")
            f.write("R2 Score: " + str(r2) + "\n")
        
        # Vẽ và lưu các biểu đồ cho LR
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=y_pred, y=(y_test - y_pred))
        plt.axhline(0, color='red', linestyle='--')
        plt.title('Residuals vs Dự đoán (Linear Regression)')
        plt.xlabel('Dự đoán')
        plt.ylabel('Residuals')
        plt.savefig(os.path.join(lr_report_dir, "lr_residuals.png"))
        plt.close()
        
        plt.figure(figsize=(10,6))
        sns.histplot(y_test - y_pred, kde=True)
        plt.title('Phân phối Residuals (Linear Regression)')
        plt.xlabel('Residuals')
        plt.ylabel('Tần số')
        plt.savefig(os.path.join(lr_report_dir, "lr_residuals_hist.png"))
        plt.close()
        
        plt.figure(figsize=(10,6))
        stats.probplot(y_test - y_pred, dist="norm", plot=plt)
        plt.title('Q-Q Plot của Residuals (Linear Regression)')
        plt.savefig(os.path.join(lr_report_dir, "lr_qqplot.png"))
        plt.close()
        
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_
        }).sort_values(by='coefficient', key=lambda x: abs(x), ascending=False)
        plt.figure(figsize=(10,6))
        sns.barplot(data=coef_df, x='coefficient', y='feature')
        plt.title('Tác động của từng đặc trưng (Linear Regression)')
        plt.savefig(os.path.join(lr_report_dir, "lr_feature_impact.png"))
        plt.close()
        
        return rmse, mae, r2
    
    def save_model(self, model_path='report/LR/lr_model.pkl', scaler_path='report/LR/lr_scaler.pkl'):
        """Lưu mô hình Linear Regression và bộ chuẩn hóa vào folder LR trong report"""
        print("5. Đang lưu mô hình Linear Regression...")
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        if self.best_params:
            with open(os.path.join("report/LR", "lr_best_params.txt"), 'w', encoding="utf-8") as f:
                f.write(str(self.best_params))

# ----------------- Main Function -----------------
def main():
    # Tạo các folder report, RF và LR nếu chưa tồn tại
    os.makedirs("report/RF", exist_ok=True)
    os.makedirs("report/LR", exist_ok=True)
    
    # ----------------- Random Forest -----------------
    rf_model = RandomForestModel()
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = rf_model.prepare_data('data.csv')
    rf_model.train(X_train_rf, y_train_rf, optimize=True)
    rf_model.evaluate(X_test_rf, y_test_rf)
    rf_model.save_model()
    
    # ----------------- Linear Regression -----------------
    lr_model = LinearRegressionModel()
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = lr_model.prepare_data('data.csv')
    lr_model.train(X_train_lr, y_train_lr, optimize=True)
    lr_model.evaluate(X_test_lr, y_test_lr)
    lr_model.save_model()

if __name__ == "__main__":
    main()
