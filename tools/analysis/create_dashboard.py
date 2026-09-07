"""建立影片手勢分類與時間一致率儀表板截圖。"""
import argparse
import cv2
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.detectors.hand_detector import HandSignRecognizer

def create_dashboard(video_path, max_frames=100):
    """建立手勢辨識的即時視覺化"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"無法開啟影片：{video_path}")
    recognizer = HandSignRecognizer()
    
    # 儲存歷史數據
    gesture_history = []
    consistency_history = []
    
    frame_count = 0
    while frame_count < max_frames:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)
        detections = recognizer.process(img)
        
        # 記錄歷史
        stable_status = recognizer.stable_status
        gesture_history.append(stable_status["sign"])
        consistency_history.append(stable_status["consistency"])
        
        # 建立儀表板
        dashboard = np.zeros((600, 800, 3), dtype=np.uint8)
        dashboard.fill(30)  # 深色背景
        
        # 顯示當前手勢
        cv2.putText(dashboard, "Current Gesture:", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(dashboard, stable_status["sign"], (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # 顯示時間一致率，不把它當成模型正確率。
        consistency = stable_status["consistency"]
        status_text = "Confirmed" if stable_status["is_stable"] else "Confirming"
        cv2.putText(dashboard, f"Consistency: {consistency:.2%} ({status_text})", (20, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 繪製置信度條
        bar_width = int(200 * consistency)
        cv2.rectangle(dashboard, (20, 170), (220, 190), (100, 100, 100), -1)
        cv2.rectangle(dashboard, (20, 170), (20 + bar_width, 190), (0, 255, 0), -1)
        
        # 顯示手部數量
        hand_count = len(detections)
        cv2.putText(dashboard, f"Hands Detected: {hand_count}", (20, 230),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 繪製歷史趨勢圖
        if len(gesture_history) > 1:
            chart_height = 200
            chart_width = 500
            chart_x = 250
            chart_y = 50
            
            cv2.rectangle(dashboard, (chart_x, chart_y),
                         (chart_x + chart_width, chart_y + chart_height),
                         (50, 50, 50), -1)
            
            # 繪製置信度趨勢
            for i in range(1, len(consistency_history)):
                x1 = chart_x + (i-1) * chart_width // len(consistency_history)
                y1 = chart_y + chart_height - int(consistency_history[i-1] * chart_height)
                x2 = chart_x + i * chart_width // len(consistency_history)
                y2 = chart_y + chart_height - int(consistency_history[i] * chart_height)
                cv2.line(dashboard, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 顯示歷史手勢
        y_offset = 300
        for i, gesture in enumerate(gesture_history[-10:]):
            cv2.putText(dashboard, f"{i+1}. {gesture}", (20, y_offset + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 合併顯示
        if img.shape[0] != 600:
            img_resized = cv2.resize(img, (int(img.shape[1] * 600 / img.shape[0]), 600))
        else:
            img_resized = img
        
        combined = np.hstack([img_resized, dashboard])
        
        # 儲存儀表板截圖（每 10 幀存一次）
        if frame_count % 10 == 0:
            cv2.imwrite(f"dashboard_frame_{frame_count:03d}.jpg", combined)
            print(f"已儲存儀表板截圖: dashboard_frame_{frame_count:03d}.jpg")
        
        frame_count += 1
    
    cap.release()
    recognizer.close()

    if frame_count == 0:
        raise ValueError(f"影片沒有可讀取的影格：{video_path}")
    
    print(f"\n儀表板建立完成！共儲存 {frame_count // 10} 張截圖")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="要分析的影片路徑")
    parser.add_argument("--max-frames", type=int, default=50)
    args = parser.parse_args()
    try:
        create_dashboard(args.video, max_frames=args.max_frames)
    except (OSError, ValueError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        raise SystemExit(1)
