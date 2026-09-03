"""智能手指偵測 - 使用多種特徵綜合判斷"""
import math
import cv2
import sys
from hand_detector import HandSignRecognizer

def smart_finger_detection(points, recognizer):
    """使用多種特徵綜合判斷手指是否伸直"""
    fingers = {}
    wrist = points[0]
    
    for finger in ["index", "middle", "ring", "pinky"]:
        base_idx = recognizer.FINGER_MCPS[finger]
        mcp = points[base_idx]
        pip = points[base_idx + 1]
        dip = points[base_idx + 2]
        tip = points[base_idx + 3]
        
        # 特徵 1: 指尖到 MCP 的距離與總段長的比例
        straight = recognizer._distance(mcp, tip)
        segments = (
            recognizer._distance(mcp, pip)
            + recognizer._distance(pip, dip)
            + recognizer._distance(dip, tip)
        )
        straight_ratio = straight / max(segments, 0.001)
        
        # 特徵 2: 指尖到手腕的距離與 PIP 到手腕的距離比
        d_wrist_tip = recognizer._distance(wrist, tip)
        d_wrist_pip = recognizer._distance(wrist, pip)
        wrist_ratio = d_wrist_tip / max(d_wrist_pip, 0.001)
        
        # 特徵 3: 手指關節的角度（透過餘弦定理）
        a = recognizer._distance(mcp, pip)
        b = recognizer._distance(pip, dip)
        c = recognizer._distance(mcp, dip)
        if a > 0.001 and b > 0.001:
            cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
            cos_angle = max(-1, min(1, cos_angle))
            angle = math.degrees(math.acos(cos_angle))
        else:
            angle = 180
        
        # 綜合判斷
        is_extended = (
            angle > 120 
            and (straight_ratio > 0.35 or wrist_ratio > 0.70)
        )
        
        fingers[finger] = is_extended
        
        print(f"{finger:8s}: angle={angle:.1f}° straight_ratio={straight_ratio:.3f} "
              f"wrist_ratio={wrist_ratio:.3f} -> {'伸直' if is_extended else '彎曲'}")
    
    return fingers

if __name__ == "__main__":
    video_path = "sign_language_app/raw_videos_backup/deaf/手語短句 3 好 不好 聾人 聽障 聽人 聽不到 手語 說話 [FOcge8IAQJ4].mp4"
    cap = cv2.VideoCapture(video_path)
    recognizer = HandSignRecognizer()
    
    for frame_count in range(5):
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = recognizer.hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                print(f"\n=== Frame {frame_count} ===")
                fingers = smart_finger_detection(landmarks.landmark, recognizer)
                print(f"手指狀態: {fingers}")
                break
        else:
            print(f"\n=== Frame {frame_count}: 未偵測到手 ===")
    
    cap.release()
    recognizer.close()
