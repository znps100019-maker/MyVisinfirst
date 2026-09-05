import os
import sys
import json
import cv2

# 將專案根目錄加入路徑，以便載入手勢偵測模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hand_detector import HandSignRecognizer

def main():
    import argparse
    parser = argparse.ArgumentParser(description="從影片中擷取手部關節點以建立手語資料庫。")
    parser.add_argument("--input-dir", default="dataset_videos", help="存放手勢資料夾與影片的目錄。")
    parser.add_argument("--output", default="hand_gesture_dataset.json", help="輸出 JSON 資料庫的檔案路徑。")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_path = args.output

    if not os.path.exists(input_dir):
        print(f"錯誤：找不到影片目錄 '{input_dir}'。")
        print("請建立如下結構的資料夾：")
        print("  dataset_videos/")
        print("    ├── hello/        (資料夾名稱即為標籤)")
        print("    │    └── video1.mp4")
        print("    └── thank_you/")
        print("         └── video2.mp4")
        sys.exit(1)

    # 初始化我們現有的手勢偵測器
    recognizer = HandSignRecognizer()
    dataset = {}

    print(f"正在掃描 '{input_dir}' 目錄下的手語影片...")
    # 獲取所有子目錄（分類名稱即為資料夾名稱）
    categories = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    if not categories:
        print(f"錯誤：在 '{input_dir}' 中沒有找到任何手勢分類子資料夾。請建立子資料夾並放入影片。")
        sys.exit(1)

    for category in categories:
        cat_dir = os.path.join(input_dir, category)
        # 支援多種常見影片格式
        video_files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if not video_files:
            continue

        print(f"\n▶ 正在處理分類：'{category}' (共 {len(video_files)} 部影片)...")
        dataset[category] = []

        for video_file in video_files:
            video_path = os.path.join(cat_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"  無法開啟影片: {video_file}")
                continue

            print(f"  讀取 {video_file}... ", end="", flush=True)
            video_frames_data = []
            frame_idx = 0
            
            while True:
                success, frame = cap.read()
                if not success:
                    break
                
                # 使用偵測器處理每一幀
                detections = recognizer.process(frame)
                
                # 如果該影格有偵測到手部，儲存對應的關節點數據
                if detections:
                    frame_data = {
                        "frame_index": frame_idx,
                        "hands": []
                    }
                    for det in detections:
                        # 擷取 21 個 3D 正規化座標（可用於機器學習訓練，如 LSTM / KNN / SVM）
                        raw_landmarks = []
                        for lm in det["landmarks"].landmark:
                            raw_landmarks.append({
                                "x": round(lm.x, 5),
                                "y": round(lm.y, 5),
                                "z": round(lm.z, 5)
                            })
                        
                        frame_data["hands"].append({
                            "handedness": det["handedness"],   # 左手/右手
                            "fingers": det["fingers"],         # 每隻手指伸直狀態 (True/False)
                            "sign": det["sign"],               # 當前規則辨識出的手勢
                            "joint_points": det["joint_points"],  # 像素座標 (x, y, z)
                            "raw_landmarks": raw_landmarks        # MediaPipe 原始 3D 正規化座標
                        })
                    video_frames_data.append(frame_data)
                
                frame_idx += 1
            
            cap.release()
            print(f"完成 (共 {frame_idx} 幀，其中 {len(video_frames_data)} 幀成功偵測手部)")
            
            dataset[category].append({
                "video_file": video_file,
                "total_frames": frame_idx,
                "frames_with_hands": video_frames_data
            })

    recognizer.close()

    # 寫入 JSON 資料庫檔案
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 手語關節點資料庫建立完成！已儲存至：{output_path}")
    print(f"總共處理手勢分類數：{len(dataset)}")

if __name__ == "__main__":
    main()
