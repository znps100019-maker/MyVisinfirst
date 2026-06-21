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

# 在匯入 mediapipe 之前立即執行路徑防呆
handle_non_ascii_path()

# 載入 MediaPipe 與 yt-dlp
try:
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    import yt_dlp
except ImportError:
    print("錯誤：找不到 mediapipe、opencv 或 yt-dlp 模組。請確認是否已安裝套件。")
    sys.exit(1)

def normalize_landmarks(landmarks_list):
    """手部關節點正規化"""
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
    """KNN 表決分類器"""
    if not samples:
        return "Unknown", 0.0

    distances = []
    for sample in samples:
        dist = euclidean_distance(query_vector, sample["vector"])
        distances.append((dist, sample["label"]))

    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    labels = [n[1] for n in neighbors]
    most_common = Counter(labels).most_common(1)[0]
    
    label = most_common[0]
    confidence = most_common[1] / k
    return label, confidence

def get_stream_url(video_input):
    """如果輸入是網路連結，使用 yt-dlp 取得直接串流 URL"""
    if video_input.startswith(("http://", "https://")):
        print(f"正在分析網路影片連結: {video_input}")
        # 取得最低畫質影片的直接下載串流網址
        ydl_opts = {
            'format': 'worst[ext=mp4]/worst',
            'quiet': True,
            'no_warnings': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(video_input, download=False)
                return info['url']
            except Exception as e:
                print(f"無法解析網路影片連結: {e}")
                sys.exit(1)
    return video_input

def main():
    import argparse
    parser = argparse.ArgumentParser(description="辨識影片（本機檔案或網路 YouTube 連結）中的手語。")
    parser.add_argument("--input", required=True, help="影片路徑或網頁連結 (例如: raw_videos/hello/xxx.mp4 或 https://youtu.be/xxx)。")
    parser.add_argument("--model", default="model.json", help="樣板模型 model.json 路徑。")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, args.model)

    if not os.path.exists(model_path):
        print(f"錯誤：找不到樣板模型檔案 '{model_path}'。請先訓練模型。")
        sys.exit(1)

    # 載入樣板
    with open(model_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)
    samples = model_data["samples"]
    print(f"模型載入成功，共有 {len(samples)} 筆樣板特徵。")

    # 取得影片資源 (如果輸入是網址，這會解析成串流連結)
    video_source = get_stream_url(args.input)

    # 初始化相機與偵測器
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"錯誤：無法開啟影片資源: {args.input}")
        sys.exit(1)

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

    print("\n辨識啟動成功！")
    print("鍵盤熱鍵說明：")
    print("  - 按 'q' 鍵：中途退出辨識。")
    print("  - 按 'c' 鍵：清空翻譯語句。")

    cv2.namedWindow("Sign Language Video Recognizer", cv2.WINDOW_NORMAL)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("\n影片播放結束。")
                break

            # 轉向處理（使偵測與相機畫面一致，手勢影像通常需鏡像對稱）
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            # 轉成 RGB 並用 MediaPipe 偵測
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(rgb)

            current_sign = "No hand"
            confidence = 0.0

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                lm_list = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in landmarks.landmark]
                query_vector = normalize_landmarks(lm_list)

                # KNN 分類
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

            cv2.imshow("Sign Language Video Recognizer", frame)

            # 按鍵監聽
            key = cv2.waitKey(10) & 0xFF  # 設定為 10ms，使影片播放順暢
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence = []
                last_added_sign = None

            if cv2.getWindowProperty("Sign Language Video Recognizer", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        hands_detector.close()
        cv2.destroyAllWindows()
        
        # 影片播放完畢後在終端機輸出完整翻譯結果
        print("\n========================================")
        print("🎬 影片手語辨識與翻譯結束。")
        if sentence:
            print(f"📝 最終翻譯句：{' -> '.join(sentence)}")
        else:
            print("📝 沒有辨識出任何有效的手語手勢。")
        print("========================================\n")

if __name__ == "__main__":
    main()
