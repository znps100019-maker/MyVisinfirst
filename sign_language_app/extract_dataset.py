import os
import sys
import json
import subprocess
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
    mp_hands = mp.solutions.hands
except ImportError:
    print("錯誤：找不到 mediapipe 模組。請確認是否已安裝套件。")
    sys.exit(1)

def normalize_landmarks(landmarks_list):
    """
    手部關節點正規化演算法：
    1. 平移不變性：以手腕 (第 0 點) 為原點 (0, 0, 0)，所有關節點座標減去手腕座標。
    2. 比例不變性：計算手腕 (第 0 點) 到中指根部 MCP (第 9 點) 的 3D 距離作為「掌心基準大小」。
       將所有座標除以此基準大小，消除因為手離鏡頭遠近產生的縮放影響。
    """
    wrist = landmarks_list[0]
    middle_mcp = landmarks_list[9]
    
    # 計算掌心大小（手腕到中指根部的歐式距離）
    dx = middle_mcp['x'] - wrist['x']
    dy = middle_mcp['y'] - wrist['y']
    dz = middle_mcp['z'] - wrist['z']
    scale = np.sqrt(dx**2 + dy**2 + dz**2)
    if scale == 0:
        scale = 1e-6
        
    normalized = []
    for lm in landmarks_list:
        # 平移並縮放
        nx = (lm['x'] - wrist['x']) / scale
        ny = (lm['y'] - wrist['y']) / scale
        nz = (lm['z'] - wrist['z']) / scale
        normalized.extend([nx, ny, nz])
        
    # 回傳 63 維特徵向量 (21 個點 * 3D 軸)
    return normalized

def process_video(video_path, hands_detector, frame_interval=15):
    """讀取影片影格，每隔 frame_interval 影格偵測並擷取正規化特徵以加速處理"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
        
    features_list = []
    frame_idx = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue
        frame_idx += 1
            
        # 鏡像翻轉並轉為 RGB
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb)
        
        if results.multi_hand_landmarks:
            # 為了訓練穩定，每次只取畫面中最大或偵測到的第一隻手
            landmarks = results.multi_hand_landmarks[0]
            
            lm_list = []
            for lm in landmarks.landmark:
                lm_list.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
            
            # 正規化為 63 維向量
            feat = normalize_landmarks(lm_list)
            features_list.append(feat)
                
    cap.release()
    return features_list

def main():
    import argparse
    parser = argparse.ArgumentParser(description="從已分類手語影片中擷取正規化手部關節特徵。")
    # 將預設路徑設為與 downloader.py 下載輸出對齊的目錄
    parser.add_argument("--videos-dir", default="raw_videos", help="包含分類影片的目錄。")
    parser.add_argument("--output", default="dataset.json", help="輸出的 JSON 特徵資料庫路徑。")
    parser.add_argument("--limit", type=int, default=0, help="限制處理的影片總數，0 代表不限制。")
    parser.add_argument("--frame-interval", type=int, default=15, help="影格擷取間隔，預設每 15 影格擷取一次。")
    args = parser.parse_args()
    
    # 取得相對此腳本所在目錄的正確絕對路徑
    script_dir = os.path.dirname(os.path.abspath(__file__))
    videos_dir = os.path.join(script_dir, args.videos_dir)
    output_path = os.path.join(script_dir, args.output)
    
    if not os.path.exists(videos_dir):
        print(f"錯誤：找不到影片分類目錄 '{videos_dir}'。請先執行 downloader.py。")
        sys.exit(1)
        
    # 初始化 MediaPipe Hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )
    
    dataset = {}
    
    # 掃描子目錄獲取分類標籤
    categories = [d for d in os.listdir(videos_dir) if os.path.isdir(os.path.join(videos_dir, d))]
    
    print(f"正在從目錄 {videos_dir} 擷取特徵...")
    
    video_count = 0
    limit = args.limit
    
    for category in categories:
        cat_dir = os.path.join(videos_dir, category)
        video_files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if not video_files:
            continue
            
        print(f"\n> 正在擷取手語標籤: '{category}'...")
        dataset[category] = []
        
        for video_file in video_files:
            if limit > 0 and video_count >= limit:
                break
            video_path = os.path.join(cat_dir, video_file)
            print(f"  處理影片 {video_file}... ", end="", flush=True)
            
            features = process_video(video_path, hands_detector, args.frame_interval)
            print(f"完成 (成功擷取 {len(features)} 影格)")
            
            if features:
                dataset[category].extend(features)
                video_count += 1
                
        if limit > 0 and video_count >= limit:
            break
                
    hands_detector.close()
    
    # 將所有特徵矩陣寫入 dataset.json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)
        
    print(f"\n[OK] 特徵資料庫編譯成功，已寫入：{output_path}")
    
if __name__ == "__main__":
    main()
