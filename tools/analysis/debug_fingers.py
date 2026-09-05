"""調試手指偵測 - 顯示詳細的手指判斷資訊"""
import cv2
import sys
from hand_detector import HandSignRecognizer

def debug_video(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    recognizer = HandSignRecognizer()
    
    frame_count = 0
    while frame_count < max_frames:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = recognizer.hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                points = landmarks.landmark
                
                # 手動計算手指伸直度
                print(f"\n=== Frame {frame_count} ===")
                for finger in ["index", "middle", "ring", "pinky"]:
                    base_idx = recognizer.FINGER_MCPS[finger]
                    mcp = points[base_idx]
                    pip = points[base_idx + 1]
                    dip = points[base_idx + 2]
                    tip = points[base_idx + 3]
                    wrist = points[0]
                    
                    straight = recognizer._distance(mcp, tip)
                    segments = (
                        recognizer._distance(mcp, pip)
                        + recognizer._distance(pip, dip)
                        + recognizer._distance(dip, tip)
                    )
                    straight_ratio = straight / max(segments, 0.001)
                    
                    d_wrist_tip = recognizer._distance(wrist, tip)
                    d_wrist_pip = recognizer._distance(wrist, pip)
                    wrist_ratio = d_wrist_tip / max(d_wrist_pip, 0.001)
                    
                    is_straight = straight_ratio > recognizer.FINGER_STRAIGHT_THRESHOLD
                    is_extended = d_wrist_tip > d_wrist_pip * recognizer.FINGER_WRIST_EXTENSION_RATIO
                    
                    print(f"{finger:8s}: straight_ratio={straight_ratio:.3f} "
                          f"(threshold: {recognizer.FINGER_STRAIGHT_THRESHOLD}) "
                          f"wrist_ratio={wrist_ratio:.3f} "
                          f"(threshold: {recognizer.FINGER_WRIST_EXTENSION_RATIO}) "
                          f"straight={is_straight} extended={is_extended}")
        
        frame_count += 1
    
    cap.release()
    recognizer.close()

if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "sign_language_app/raw_videos_backup/deaf/手語短句 3 好 不好 聾人 聽障 聽人 聽不到 手語 說話 [FOcge8IAQJ4].mp4"
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    debug_video(video_path, max_frames)
