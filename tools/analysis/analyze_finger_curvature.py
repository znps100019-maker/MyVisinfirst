"""分析手指彎曲度並輸出視覺化結果"""
import cv2
import numpy as np
import math
from hand_detector import HandSignRecognizer

def analyze_and_visualize(video_path, max_frames=50):
    """分析手指彎曲度並建立視覺化"""
    cap = cv2.VideoCapture(video_path)
    recognizer = HandSignRecognizer()
    
    # 儲存分析結果
    results = {
        "straight_frames": 0,
        "curved_frames": 0,
        "no_hand_frames": 0,
        "angles": [],
    }
    
    frame_count = 0
    while frame_count < max_frames:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results_hands = recognizer.hands.process(rgb)
        
        if results_hands.multi_hand_landmarks:
            for landmarks in results_hands.multi_hand_landmarks:
                points = landmarks.landmark
                
                # 分析手指角度
                angles = []
                for finger in ["index", "middle", "ring", "pinky"]:
                    base_idx = recognizer.FINGER_MCPS[finger]
                    mcp = points[base_idx]
                    pip = points[base_idx + 1]
                    dip = points[base_idx + 2]
                    
                    # 計算 PIP 關節角度
                    a = recognizer._distance(mcp, pip)
                    b = recognizer._distance(pip, dip)
                    c = recognizer._distance(mcp, dip)
                    
                    if a > 0.001 and b > 0.001:
                        cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
                        cos_angle = max(-1, min(1, cos_angle))
                        angle = math.degrees(math.acos(cos_angle))
                        angles.append(angle)
                
                if angles:
                    avg_angle = sum(angles) / len(angles)
                    results["angles"].append(avg_angle)
                    
                    if avg_angle > 120:
                        results["straight_frames"] += 1
                    else:
                        results["curved_frames"] += 1
        else:
            results["no_hand_frames"] += 1
        
        frame_count += 1
    
    cap.release()
    recognizer.close()
    
    # 輸出分析結果
    print("=" * 50)
    print("手指彎曲度分析結果")
    print("=" * 50)
    print(f"總幀數: {frame_count}")
    print(f"偵測到手的幀數: {results['straight_frames'] + results['curved_frames']}")
    print(f"沒有偵測到手的幀數: {results['no_hand_frames']}")
    
    if results["angles"]:
        avg_angle = sum(results["angles"]) / len(results["angles"])
        print(f"\n平均手指關節角度: {avg_angle:.1f}°")
        print(f"  - 伸直 (>120°): {results['straight_frames']} 幀")
        print(f"  - 彎曲 (<120°): {results['curved_frames']} 幀")
        print(f"  - 彎曲比例: {results['curved_frames'] / max(results['straight_frames'] + results['curved_frames'], 1) * 100:.1f}%")
        
        if results["curved_frames"] > results["straight_frames"]:
            print("\n結論: 影片中的人物手指大部分時間是彎曲的")
            print("建議: 使用其他影片或調整測試策略")
        else:
            print("\n結論: 影片中的人物手指大部分時間是伸直的")
    
    return results

if __name__ == "__main__":
    video_path = "sign_language_app/raw_videos_backup/deaf/手語短句 3 好 不好 聾人 聽障 聽人 聽不到 手語 說話 [FOcge8IAQJ4].mp4"
    analyze_and_visualize(video_path, max_frames=50)
