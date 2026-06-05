"""
MediaPipe
https://google.github.io/mediapipe/
https://mediapipe.dev/

針對即時的串流媒體，提供跨平台、客製化的機器學習解決方案。
可利用GPU進行加速運算。
提供多樣化的解決方案。
支援行動裝置與嵌入式系統。

MediaPipe Hands
https://google.github.io/mediapipe/solutions/hands.html
"""

"""
安裝
pip install mediapipe --user
pip install opencv-python==4.5.5.64 --user
"""

# 匯入OpenCV及MediaPipe套件
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 建立手部偵測器
base_options = python.BaseOptions(model_asset_path=r'C:\Users\USER\Downloads\Mediapipe_HandLandmark\_Mediapipe_HandLandmark\hands\hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    # 偵測手部數量上限
    num_hands=2,
    # 最小偵測與追蹤為手部的信心程度
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)
detector = vision.HandLandmarker.create_from_options(options)

# 開啟攝影機（加入 cv2.CAP_DSHOW 以相容 Windows 系統，避免無法讀取畫面）
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# 設定鏡頭解析度為 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 判斷攝影機是否處於開啟狀態
empty_frame_count = 0
while cap.isOpened():
    # 如果成功擷取會回傳兩個值，分別存放至success及frame
    # success: True or False，代表是否成功讀取到圖片
    # frame: 讀取到的那一個frame
    success, frame = cap.read()
    
    # 增加讀取frame錯誤的判斷，限制重試次數，避免無限迴圈造成 CPU 滿載
    if not success:
        empty_frame_count += 1
        print(f"Ignoring empty camera frame ({empty_frame_count}/10).")
        if empty_frame_count >= 10:
            print("【錯誤】連續 10 次讀取到空畫面，請確認相機是否被其他程式佔用，或嘗試重插拔 USB。")
            break
        continue
    empty_frame_count = 0
    
    # 透過函數cvtColor將圖片顏色進行轉換，因為OpenCV預設的顏色是BGR，而圖片是RGB
    frame = cv2.flip(frame, 1)  # 水平翻轉
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # 手部偵測
    detection_result = detector.detect(mp_image)
    
    # 如果有檢測到多個手部的landmarks
    if detection_result.hand_landmarks:
        # 針對每一個landmarks(共21個點)進行繪製與連結
        for hand_landmarks in detection_result.hand_landmarks:
            for landmark in hand_landmarks:
                pos = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                cv2.circle(frame, pos, 5, (0, 255, 0), -1)
            
            # 繪製手部骨架連線
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
                (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
                (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
                (0, 13), (13, 14), (14, 15), (15, 16),  # 無名指
                (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
                (5, 9), (9, 13), (13, 17)  # 手掌
            ]
            for connection in connections:
                start_idx, end_idx = connection
                start_pos = (int(hand_landmarks[start_idx].x * frame.shape[1]), 
                           int(hand_landmarks[start_idx].y * frame.shape[0]))
                end_pos = (int(hand_landmarks[end_idx].x * frame.shape[1]), 
                         int(hand_landmarks[end_idx].y * frame.shape[0]))
                cv2.line(frame, start_pos, end_pos, (255, 255, 255), 2)
    
    # 利用imshow來開啟視窗將該擷取的frame顯示出來，'Hands Detection'是視窗上的標題
    cv2.imshow('Hands Detection', frame)
    # 等待使用者按下ESC鍵來停止迴圈
    # waitKey(0): 無限期顯示視窗，直到按任何鍵為止。
    # waitKey(1): 等待keyPress 1毫秒，如果不是按下對應的按鍵，將繼續刷新並使用讀取攝影機的frame
    if cv2.waitKey(1) & 0xFF == 27:
        # 跳出無窮迴圈
        break
        
# 釋放鏡頭資源
cap.release()
# 關閉所有視窗
cv2.destroyAllWindows()