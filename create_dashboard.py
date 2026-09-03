"""建立即時視覺化儀表板"""
import cv2
import numpy as np
from hand_detector import HandSignRecognizer

def create_dashboard(video_path, max_frames=100):
    """建立手勢辨識的即時視覺化"""
    cap = cv2.VideoCapture(video_path)
    recognizer = HandSignRecognizer()
    
    # 儲存歷史數據
    gesture_history = []
    confidence_history = []
    
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
        confidence_history.append(stable_status["confidence"])
        
        # 建立儀表板
        dashboard = np.zeros((600, 800, 3), dtype=np.uint8)
        dashboard.fill(30)  # 深色背景
        
        # 顯示當前手勢
        cv2.putText(dashboard, "Current Gesture:", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(dashboard, stable_status["sign"], (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # 顯示置信度
        confidence = stable_status["confidence"]
        cv2.putText(dashboard, f"Confidence: {confidence:.2%}", (20, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 繪製置信度條
        bar_width = int(200 * confidence)
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
            for i in range(1, len(confidence_history)):
                x1 = chart_x + (i-1) * chart_width // len(confidence_history)
                y1 = chart_y + chart_height - int(confidence_history[i-1] * chart_height)
                x2 = chart_x + i * chart_width // len(confidence_history)
                y2 = chart_y + chart_height - int(confidence_history[i] * chart_height)
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
    
    print(f"\n儀表板建立完成！共儲存 {frame_count // 10} 張截圖")

if __name__ == "__main__":
    video_path = "sign_language_app/raw_videos_backup/deaf/手語新手教室 第九課：購物、時態｜手語方向｜手語練習環節｜香港手語｜WeTV x 聾場蜜語 [rVdnKYFpQSg].mp4"
    create_dashboard(video_path, max_frames=50)
