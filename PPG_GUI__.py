import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from tkinter import Label
from tkinter.font import Font
from scipy.signal import butter, filtfilt, find_peaks
import joblib
import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Tk 창 생성
window = tk.Tk()
window.title("음주 판단 프로그램")
window.geometry("1500x800+100+100")
window.resizable(True, True)
window.configure(bg="#f0f0f0")
# ttk 스타일 설정
style = ttk.Style()
style.theme_use("clam")  # 사용할 테마 설정 ('clam', 'alt', 'default', 'classic')
style.configure("TButton", font=("Helvetica", 12), padding=10)
style.configure("TLabel", font=("Helvetica", 12), background="#f0f0f0", padding=10)
# Frame을 생성하고 창 크기 변화에 따라 확장되도록 설정
frame = tk.Frame(window)
frame.grid(row=0, column=0, sticky="nsew")  # grid로 배치

# 창의 행과 열 확장 비율 설정
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

# Frame 내부에 grid 비율 설정
frame.grid_rowconfigure(0, weight=1)
frame.grid_rowconfigure(1, weight=1)
frame.grid_rowconfigure(2, weight=1)
frame.grid_columnconfigure(0, weight=1)
frame.grid_columnconfigure(1, weight=3)

# 그래프 초기화
fig, ax = plt.subplots(figsize=(4, 2))
line, = ax.plot([], '-', label='Raw Data')  # 빈 플롯 생성
ax.set_xlabel('Raw Data')
ax.set_ylabel('blood flow rate')
ax.set_title('Data floating')
ax.grid(True)
ax.legend()

# Tkinter canvas에 Figure 추가
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.draw()
canvas.get_tk_widget().grid(row=0, column=1, rowspan=4, sticky="nsew", padx=20, pady=20)

raw_data = []  # ppg신호 데이터 담을 전역변수
feature_data = [] # 피처 담을 전역변수
predicted_labels = 0
ser = None  # 시리얼 포트 객체를 처음에는 None으로 설정


    
def Serialrun():
    global raw_data, ser
    # 시리얼 포트 연결 설정 (COM11 포트, 보드레이트 115200)
    ser = serial.Serial('COM13', 115200, timeout=1)

    Start_Button['text'] = "시리얼통신 시작"
    ser.write(b"True\n") # 아두이노에 True 전송
    # 데이터 수신 및 플로팅을 위한 변수 초기화
    i = 0
    max_time = 1500



    # 시리얼 통신으로부터 데이터를 읽고 실시간으로 플롯 업데이트
    while i <= max_time:
        if ser.in_waiting > 0:  # 시리얼 버퍼에 데이터가 있는지 확인
            data = ser.readline().decode().strip()  # 문자열로 데이터 읽기
            if data:
                try:
                    a_num = float(data)  # 문자열을 숫자로 변환
                    raw_data.append(a_num)  # 변환된 숫자를 배열에 저장
                    line.set_ydata(raw_data)
                    line.set_xdata(np.arange(1, len(raw_data) + 1))  # XData 업데이트
                    ax.relim()
                    ax.autoscale_view()  # 축 자동 스케일링
                    canvas.draw()  # Tkinter에서 그래프 업데이트
                    canvas.get_tk_widget().update()  # GUI 업데이트
                    i += 1
                except ValueError:
                    pass  # 변환 실패 시 무시

    ser.write(b"False\n")
    # 시리얼 포트 닫기
    ser.close()
    result_label["text"] = "측정 시작"


def normalize_column(column):
    column = np.array(column)
    return (column - np.min(column)) / (np.max(column) - np.min(column))

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def calculate_peak_of_pulse(data):
    peaks, _ = find_peaks(data)
    return peaks * 20  # Return the indices of the peaks

def calculate_ppi_stats(peaks):
    if len(peaks) > 1:
        ppi_intervals = np.diff(peaks)
        ppi_mean = round(np.mean(ppi_intervals), 3)
        ppi_variance = round(np.var(ppi_intervals), 2)
        ppi_std = round(np.std(ppi_intervals), 3)
        return ppi_intervals, ppi_mean, ppi_variance, ppi_std
    else:
        return None, None, None

def calculate_NN50(peaks):
    diff_ppi = np.diff(peaks)
    nn50 = np.sum(np.abs(diff_ppi) > 50)
    return nn50

def calculate_Pnn50(nn50, ppi):
    return (nn50 / len(ppi)) * 100

def calculate_slope(data, time_step=1):
    return np.gradient(data, time_step)

def calculate_slope_stats(slope_data):
    mean_slope = np.mean(slope_data)
    max_slope = np.max(slope_data)
    min_slope = np.min(slope_data)
    std_slope = np.std(slope_data)
    return mean_slope, max_slope, min_slope, std_slope

def process_data():
    global raw_data
    if raw_data:
        # 정규화 및 밴드패스 필터링
        normalized_data = normalize_column(raw_data)
        fs = 50
        lowcut = 0.8
        highcut = 2.5
        filtered_data = bandpass_filter(normalized_data, lowcut, highcut, fs)

        # 피크 계산
        detected_peaks = calculate_peak_of_pulse(filtered_data)

        # PPI 통계 계산
        ppi_intervals, ppi_mean, ppi_var, ppi_std = calculate_ppi_stats(detected_peaks)

        # BPM 계산
        bpm_value = len(detected_peaks) * 2

        # NN50 및 PNN50 계산
        nn50 = calculate_NN50(ppi_intervals)
        pnn50 = calculate_Pnn50(nn50, detected_peaks)

        # 기울기 및 기울기 통계 계산
        slope_data = calculate_slope(filtered_data)
        diff_mean,diff_max,diff_min,diff_std = calculate_slope_stats(slope_data)

        # 1. 데이터 테이블 생성
        feature_data = {
            'Bpm': [bpm_value],
            'PPI av': [ppi_mean],
            #'PPI variance': [ppi_var],
            #'PPI std': [ppi_std],
            #'NN50': [nn50],
            'Pnn50(%)': [pnn50],
            'diff_max': [diff_max],
            'diff_min': [diff_min],
            'diff_std': [diff_std],
        }
 
        # 데이터프레임으로 변환
        df = pd.DataFrame(feature_data)

        # 파일 이름 설정
        file_name = 'calculated_parameters_alcohol.xlsx'

        # 테이블을 엑셀 파일로 저장
        df.to_excel(file_name, index=False)

        # 파일 저장 경로 출력

        full_file_path = os.path.join(os.getcwd(), file_name)
        print(f'파일이 성공적으로 저장되었습니다: {full_file_path}')

        #딕셔너리를 문자열로 변환하면서 항목 간에 한 줄 띄우기
        formatted_feature_data = "\n\n".join([f"{key}: {value}" for key, value in feature_data.items()])

        # 결과를 result_label에 표시
        result_label['text'] = formatted_feature_data


def Python_to_Arduino(predicted_labels):
    """
    Python에서 아두이노로 데이터를 전송하는 함수

    Args:
        port (str): 아두이노가 연결된 COM 포트 (예: 'COM10')
        baudrate (int): 시리얼 통신 속도 (예: 115200)
        data (str): 전송할 데이터 ('1' 또는 '0')
    """
    
    try:
        # 아두이노 연결 설정
        ser = serial.Serial('COM13', 115200, timeout=1)
        
        # 약간의 지연 시간
        time.sleep(2)
        
        # '1' 또는 '0'이 입력되었을 때 아두이노로 전송
        if predicted_labels == 1:
            ser.write(b'1')
        elif predicted_labels == 0:
            ser.write(b'0')
        else:
            print("잘못된 입력입니다. 1 또는 0만 입력하세요.")
        
        # 연결 닫기
        ser.close()
        
    except Exception as e:
        print(f"에러 발생: {e}")

def machinelaering():

            # 데이터 로드
            file_path = "PATH" # 파일 경로 수정
            data = pd.read_excel(file_path)

            # 레이블 0과 1만 선택 (레이블 2는 제외)
            data_binary = data[data['label'].isin([0, 1])]

            # 데이터 나누기
            X = data_binary.drop(columns=['label'])  # label 제외한 특징 데이터 (8개)
            y = data_binary['label']  # 라벨 데이터
            
            # 데이터를 훈련 데이터와 테스트 데이터로 분할 (8:2 비율로)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 더 세밀한 하이퍼파라미터 그리드 설정
            param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000], 
                        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                        'kernel': ['rbf']}

            # GridSearchCV로 세밀한 하이퍼파라미터 탐색
            grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
            grid.fit(X_train, y_train)

            # 학습 데이터에 대한 예측 및 정확도 계산
            y_pred_train = grid.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_pred_train)
            print(f"Train Accuracy: {train_accuracy * 100:.2f}%")

            # 테스트 데이터에 대한 예측 및 정확도 계산
            y_pred_grid = grid.predict(X_test)
            grid_test_accuracy = accuracy_score(y_test, y_pred_grid)
            print(f"Test Accuracy after detailed tuning: {grid_test_accuracy * 100:.2f}%")


            #최적 하이퍼파라미터 출력
            print(f"Best parameters: {grid.best_params_}")

            # 최적 모델 저장
            best_model = grid.best_estimator_
            model_filename = 'best_svm_model.pkl'
            joblib.dump(best_model, model_filename)

# 예측 함수
def predict_drinking_status():

            print(feature_data)
            machinelaering()

            # 저장된 SVM 모델 불러오기
            model = joblib.load('best_svm_model.pkl')

            # 데이터 로드 (엑셀 파일)
            file_name = 'calculated_parameters_alcohol.xlsx'  # 파일 이름을 필요에 맞게 수정하세요.
            data = pd.read_excel(file_name)
            X = data.iloc[:, :8]
            # 데이터 스케일링 (z-score 사용)
  

            # 저장된 SVM 모델을 사용하여 예측
            predicted_labels = model.predict(X)
            
            # 예측 결과 출력
            print('예측된 라벨 (음주: 1, 정상: 0):')
            print(predicted_labels[0])  # 첫 10개 결과 출력

            predict_label['text'] = f'예측된 라벨 : {predicted_labels}'
            Python_to_Arduino(predicted_labels[0])

#초기화 함수
def init():
    global ser
     
    if ser and ser.is_open:  # 시리얼 포트가 열려 있는지 확인
        ser.write(b"False\n")  # 중지 명령 전송
        ser.close()  # 시리얼 포트 닫기
        ser = None


    # 그래프 초기화
    raw_data.clear()  # 기존 데이터를 비웁니다
    feature_data.clear()  # 피처 데이터를 비웁니다
    predicted_labels = 0

    line.set_ydata([])
    line.set_xdata([])
    ax.relim()               # 데이터 리미트 재설정
    ax.autoscale_view()      # 축 자동 스케일링
    canvas.draw() 

    line.set_ydata([])  # 그래프 초기화
    line.set_xdata([])  # 그래프 초기화
    canvas.draw()
    result_label['text'] = ""
    predict_label['text']=""
    print("초기화 완료.")

#모터 정지?
def stop():
      global ser
      if ser is None or not ser.is_open:
        try:
            ser = serial.Serial('COM13', 115200, timeout=1)
            time.sleep(2)  # 연결 후 안정화를 위해 약간의 대기 시간 추가
            print("Serial connection established on COM11")
        except serial.SerialException:
            print("Failed to open COM13. Please check the connection.")
            return  # 연결 실패 시 함수 종료

    # 시리얼 연결이 열려 있다면 명령어 전송
        if ser.is_open:
            ser.write(b"2\n")
        else:
            print("Serial connection is not open.")


def Start():
    Serialrun()
    process_data()
    predict_drinking_status()


# 폰트 스타일 정의
large_font = Font(family="Helvetica", size=20, weight="bold")  # 굵은 글씨체
small_font = Font(family="Helvetica", size=12)

# ttk 스타일 정의
style = ttk.Style()
style.configure("Large.TButton", font=large_font)
style.configure("Small.TButton", font=small_font)
# 버튼과 라벨 추가
# SerialStartBttn = Button(frame, text="측정 시작", overrelief="solid", command=Serialrun)
# SerialStartBttn.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

# 버튼과 라벨 개선
ProcessDataBttn = ttk.Button(frame, text="데이터 처리",style="Large.TButton",command=process_data)
ProcessDataBttn.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

predict_button = ttk.Button(frame, text="예측 시작",style="Large.TButton", command=predict_drinking_status)
predict_button.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)

init_button = ttk.Button(frame, text="초기화",style="Large.TButton", command=init)
init_button.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)

stop_button = ttk.Button(frame, text="모터 정지",style="Large.TButton", command=stop)
stop_button.grid(row=4, column=0, sticky="nsew", padx=10, pady=10)

Start_Button = ttk.Button(frame, text="측정 시작",style="Large.TButton", command=Start)
Start_Button.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

# 결과 출력 라벨
result_label = ttk.Label(frame, text="특징 값", justify="left", anchor="nw")
result_label.grid(row=2, column= 3, sticky="w", padx=10, pady=10)

predict_label = ttk.Label(frame, text="예측 값", justify="left", anchor="nw")
predict_label.grid(row=5, column= 3, sticky="w", padx=10, pady=10)

# 그래프 배경과 제목 스타일링
ax.set_facecolor("#eaeaf2")
ax.set_title("Data Floating", fontsize=14, fontweight="bold")
ax.grid(True, color="#cccccc", linestyle="--", linewidth=0.5)

# 결과 출력 라벨
result_label = Label(frame, text="", justify="left", anchor="nw")
result_label.grid(row=5, column=2, sticky="w", padx=5, pady=5)

predict_label = Label(frame, text="", justify="left", anchor="nw")
predict_label.grid(row=6, column=2, sticky="w", padx=5, pady=5)


window.mainloop()
