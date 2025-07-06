import serial
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os
import tkinter as tk
from tkinter import ttk
from tkinter.font import Font
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AlcoholDetector:
    def __init__(self, port='COM13', baudrate=115200, max_time=1500):
        self.port = port
        self.baudrate = baudrate
        self.max_time = max_time
        self.ser = None
        self.raw_data = []
        self.feature_data = {}
        self.predicted_label = None

    def open_serial(self):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
        time.sleep(2)

    def close_serial(self):
        if self.ser and self.ser.is_open:
            self.ser.write(b"False\n")
            self.ser.close()

    def send_command(self, command: bytes):
        if self.ser and self.ser.is_open:
            self.ser.write(command)

    def collect_data(self):
        self.open_serial()
        self.send_command(b"True\n")
        i = 0
        while i <= self.max_time:
            if self.ser.in_waiting > 0:
                data = self.ser.readline().decode().strip()
                if data:
                    try:
                        a_num = float(data)
                        self.raw_data.append(a_num)
                        i += 1
                    except ValueError:
                        pass
        self.send_command(b"False\n")
        self.close_serial()

    def normalize_column(self, column):
        column = np.array(column)
        return (column - np.min(column)) / (np.max(column) - np.min(column))

    def bandpass_filter(self, data, lowcut, highcut, fs=50, order=5):
        nyquist = 0.5 * fs
        b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
        return filtfilt(b, a, data)

    def calculate_features(self):
        normalized = self.normalize_column(self.raw_data)
        filtered = self.bandpass_filter(normalized, 0.8, 2.5)
        peaks, _ = find_peaks(filtered)
        bpm = len(peaks) * 2
        ppi = np.diff(peaks)
        ppi_mean = round(np.mean(ppi), 3) if len(ppi) else None
        pnn50 = (np.sum(np.abs(ppi) > 50) / len(ppi)) * 100 if len(ppi) else None
        slope = np.gradient(filtered)
        diff_max = np.max(slope)
        diff_min = np.min(slope)
        diff_std = np.std(slope)

        self.feature_data = {
            'Bpm': [bpm],
            'PPI av': [ppi_mean],
            'Pnn50(%)': [pnn50],
            'diff_max': [diff_max],
            'diff_min': [diff_min],
            'diff_std': [diff_std],
        }

        df = pd.DataFrame(self.feature_data)
        df.to_excel('calculated_parameters_alcohol.xlsx', index=False)

    def train_model(self, file_path):
        data = pd.read_excel(file_path)
        data_binary = data[data['label'].isin([0, 1])]
        X = data_binary.drop(columns=['label'])
        y = data_binary['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf']
        }

        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        grid.fit(X_train, y_train)

        print(f"Train Accuracy: {accuracy_score(y_train, grid.predict(X_train)):.2%}")
        print(f"Test Accuracy: {accuracy_score(y_test, grid.predict(X_test)):.2%}")
        print(f"Best Params: {grid.best_params_}")

        joblib.dump(grid.best_estimator_, 'best_svm_model.pkl')

    def predict(self):
        model = joblib.load('best_svm_model.pkl')
        df = pd.read_excel('calculated_parameters_alcohol.xlsx')
        X = df.iloc[:, :6]  # 6개 피처 사용
        self.predicted_label = model.predict(X)[0]
        print(f"예측 라벨: {self.predicted_label}")
        self.send_prediction_to_arduino()

    def send_prediction_to_arduino(self):
        try:
            self.open_serial()
            if self.predicted_label == 1:
                self.send_command(b'1')
            else:
                self.send_command(b'0')
            self.close_serial()
        except Exception as e:
            print(f"전송 오류: {e}")

    def reset(self):
        self.close_serial()
        self.raw_data.clear()
        self.feature_data.clear()
        self.predicted_label = None
        print("초기화 완료.")

    def stop_motor(self):
        try:
            self.open_serial()
            self.send_command(b"2\n")
            self.close_serial()
        except Exception as e:
            print(f"모터 정지 실패: {e}")

    def start_all(self):
        self.collect_data()
        self.calculate_features()
        self.predict()

        
class AlcoholApp:
    def __init__(self, root):
        self.detector = AlcoholDetector()
        self.root = root
        self.root.title("음주 판단 프로그램")
        self.root.geometry("1500x800+100+100")
        self.root.configure(bg="#f0f0f0")
        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Large.TButton", font=("Helvetica", 20), padding=10)
        style.configure("TLabel", font=("Helvetica", 12), background="#f0f0f0", padding=10)

        frame = tk.Frame(self.root)
        frame.grid(row=0, column=0, sticky="nsew")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=3)

        # 그래프 초기화
        self.fig, self.ax = plt.subplots(figsize=(4, 2))
        self.line, = self.ax.plot([], '-', label='Raw Data')
        self.ax.set_xlabel('Raw Data')
        self.ax.set_ylabel('Blood Flow Rate')
        self.ax.set_title('Data Floating')
        self.ax.grid(True)
        self.ax.legend()
        self.ax.set_facecolor("#eaeaf2")

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=1, rowspan=4, sticky="nsew", padx=20, pady=20)

        # 버튼 생성
        ttk.Button(frame, text="측정 시작", style="Large.TButton", command=self.start_all).grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        ttk.Button(frame, text="데이터 처리", style="Large.TButton", command=self.process_data).grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        ttk.Button(frame, text="예측 시작", style="Large.TButton", command=self.predict).grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        ttk.Button(frame, text="초기화", style="Large.TButton", command=self.reset).grid(row=3, column=0, sticky="nsew", padx=10, pady=10)
        ttk.Button(frame, text="모터 정지", style="Large.TButton", command=self.detector.stop_motor).grid(row=4, column=0, sticky="nsew", padx=10, pady=10)

        # 라벨
        self.result_label = ttk.Label(frame, text="특징 값", justify="left", anchor="nw")
        self.result_label.grid(row=2, column=3, sticky="w", padx=10, pady=10)

        self.predict_label = ttk.Label(frame, text="예측 값", justify="left", anchor="nw")
        self.predict_label.grid(row=5, column=3, sticky="w", padx=10, pady=10)

    def start_all(self):
        self.detector.collect_data()
        self.update_plot()
        self.process_data()
        self.predict()

    def process_data(self):
        self.detector.calculate_features()
        formatted = "\n\n".join([f"{k}: {v}" for k, v in self.detector.feature_data.items()])
        self.result_label['text'] = formatted

    def predict(self):
        self.detector.predict()
        self.predict_label['text'] = f"예측된 라벨: {self.detector.predicted_label}"

    def update_plot(self):
        raw_data = self.detector.raw_data
        self.line.set_ydata(raw_data)
        self.line.set_xdata(range(len(raw_data)))
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def reset(self):
        self.detector.reset()
        self.line.set_ydata([])
        self.line.set_xdata([])
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        self.result_label['text'] = ""
        self.predict_label['text'] = ""


if __name__ == "__main__":
    root = tk.Tk()
    app = AlcoholApp(root)
    root.mainloop()
