import os
import sys
import json
import math
import subprocess
from collections import Counter, deque
import cv2
import numpy as np

def handle_non_ascii_path():
    """
    Bypass MediaPipe path encoding bug on Windows when directory contains non-ASCII characters.
    It maps the project directory to a virtual drive (e.g., Z:) and re-runs the process.
    """
    if sys.platform != "win32":
        return

    # project_root is the parent of the sign_language_app folder (where .venv is located)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if any(ord(c) > 127 for c in project_root):
        import string
        drive = None
        for letter in string.ascii_uppercase[::-1]:
            candidate = f"{letter}:"
            if not os.path.exists(candidate + "\\"):
                drive = candidate
                break

        if not drive:
            print("Error: No free drive letter found to bypass MediaPipe path bug.")
            sys.exit(1)

        # Map the drive
        subprocess.run(["subst", drive, project_root], shell=True, stdout=subprocess.DEVNULL)

        # Re-build script path on the virtual drive
        relative_script = os.path.relpath(os.path.abspath(__file__), project_root)
        virtual_script = os.path.join(drive, relative_script)
        virtual_python = os.path.join(drive, ".venv", "Scripts", "python.exe")

        if not os.path.exists(virtual_python):
            virtual_python = sys.executable.replace(project_root, drive)

        # Spawn the child process on the virtual drive
        args = [virtual_python, virtual_script] + sys.argv[1:]
        try:
            result = subprocess.run(args)
            returncode = result.returncode
        finally:
            subprocess.run(["subst", drive, "/d"], shell=True, stdout=subprocess.DEVNULL)

        sys.exit(returncode)

# Run bypass check immediately on startup before importing mediapipe/cv2
handle_non_ascii_path()

# 載入 MediaPipe
try:
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
except ImportError:
    print("錯誤：找不到 mediapipe 或 opencv 模組。請確認是否已安裝套件。")
    sys.exit(1)

def normalize_landmarks(landmarks_list):
    """手部關節點正規化（與 extract_dataset.py 演算法完全相同）"""
    wrist = landmarks_list[0]
    middle_mcp = landmarks_list[9]
    
    # 計算掌心大小
    dx = middle_mcp['x'] - wrist['x']
    dy = middle_mcp['y'] - wrist['y']
    dz = middle_mcp['z'] - wrist['z']
    scale = np.sqrt(dx**2 + dy**2 + dz**2)
    if scale == 0:
        scale = 1e-6
        
    normalized = []
    for lm in landmarks_list:
        nx = (lm['x'] - wrist['x']) / scale
        ny = (lm['y'] - wrist['y']) / scale
        nz = (lm['z'] - wrist['z']) / scale
        normalized.extend([nx, ny, nz])
        
    return normalized

def euclidean_distance(v1, v2):
    """計算兩個向量之間的歐式距離"""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(v1, v2)))

def classify_knn(query_vector, samples, k=9):
    """
    K-Nearest Neighbors (KNN) 分類演算法：
    1. 計算輸入特徵向量與所有樣板之間的歐氏距離。
    2. 找出距離最近的 K 個鄰居。
    3. 進行多數表決決定分類，並計算置信度。
    """
    if not samples:
        return "Unknown", 0.0

    distances = []
    for sample in samples:
        dist = euclidean_distance(query_vector, sample["vector"])
        distances.append((dist, sample["label"]))

    # 排序並取出前 K 個最近的鄰居
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    # 多數決表決
    labels = [n[1] for n in neighbors]
    most_common = Counter(labels).most_common(1)[0]
    
    label = most_common[0]
    confidence = most_common[1] / k
    return label, confidence

def main():
    # 取得相對此腳本所在目錄的正確絕對路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.json")

    if not os.path.exists(model_path):
        print(f"錯誤：找不到 KNN 樣板模型檔案 '{model_path}'。")
        print("請先依序執行 downloader.py ➡️ extract_dataset.py ➡️ train_classifier.py 以生成模型。")
        sys.exit(1)

    # 載入模型數據
    print(f"正在載入手語辨識模型: {model_path}")
    with open(model_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)
    samples = model_data["samples"]
    print(f"模型載入成功，共有 {len(samples)} 筆樣板特徵。")

    # 初始化相機與偵測器
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )

    # 辨識歷史緩衝區，用於平滑結果，減少跳動誤判
    history = deque(maxlen=8)
    sentence = []
    last_added_sign = None

    print("\n系統啟動成功！")
    print("鍵盤熱鍵說明：")
    print("  - 按 'q' 鍵：結束程式。")
    print("  - 按 'c' 鍵：清除底部連貫翻譯語句。")

    # 建立可任意拉大縮小的視窗
    cv2.namedWindow("Sign Language Recognition App", cv2.WINDOW_NORMAL)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("無法讀取相機畫面。")
                break

            # 鏡像翻轉
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            # 轉成 RGB 並用 MediaPipe 偵測
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(rgb)

            current_sign = "No hand"
            confidence = 0.0

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                
                # 繪製手部骨骼
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # 轉換為座標 dict 格式
                lm_list = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in landmarks.landmark]
                
                # 取得 63 維正規化特徵向量
                query_vector = normalize_landmarks(lm_list)

                # 使用 KNN 進行分類
                current_sign, confidence = classify_knn(query_vector, samples, k=9)

                # 在手部上方印出當前預測結果
                x_vals = [lm.x for lm in landmarks.landmark]
                y_vals = [lm.y for lm in landmarks.landmark]
                x_pos = max(10, int(min(x_vals) * width))
                y_pos = max(30, int(min(y_vals) * height) - 10)
                
                cv2.putText(
                    frame,
                    f"{current_sign} ({confidence:.0%})",
                    (x_pos, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )
                
                history.append(current_sign)
            else:
                history.append("No hand")

            # 取得歷史穩定手勢
            stable_sign = "No hand"
            if len(history) > 0:
                most_common = Counter(history).most_common(1)[0]
                # 穩定影格數大於等於 5 幀才算穩定手勢
                if most_common[1] >= 5:
                    stable_sign = most_common[0]

            # 更新連貫翻譯語句
            if stable_sign == "No hand":
                last_added_sign = None
            elif stable_sign != "Unknown" and stable_sign != last_added_sign:
                sentence.append(stable_sign)
                last_added_sign = stable_sign

            # 繪製左上角狀態 HUD
            if stable_sign != "No hand":
                cv2.rectangle(frame, (10, 10), (340, 55), (0, 0, 0), cv2.FILLED)
                cv2.putText(
                    frame,
                    f"SIGN: {stable_sign}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            # 繪製底部連貫性手語翻譯條
            if sentence:
                cv2.rectangle(frame, (10, height - 55), (width - 10, height - 15), (0, 0, 0), cv2.FILLED)
                sentence_text = " -> ".join(sentence)
                
                # 自動截斷以防溢出畫面
                max_char_len = int(width / 12)
                if len(sentence_text) > max_char_len:
                    sentence_text = "..." + sentence_text[-max_char_len:]
                    
                cv2.putText(
                    frame,
                    f"SENTENCE: {sentence_text}",
                    (20, height - 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # 顯示影像
            cv2.imshow("Sign Language Recognition App", frame)

            # 按鍵監聽
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence = []
                last_added_sign = None

            # 點擊視窗右上角關閉
            if cv2.getWindowProperty("Sign Language Recognition App", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        hands_detector.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
