"""分析 Unknown 手勢的特徵，找出為什麼無法辨識"""
import cv2
import math
from collections import Counter
from hand_detector import HandSignRecognizer

def analyze_unknown_gestures(video_path, max_frames=200):
    """分析無法辨識的手勢特徵"""
    cap = cv2.VideoCapture(video_path)
    recognizer = HandSignRecognizer()
    
    unknown_patterns = []
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
                
                # 計算手指狀態
                fingers = {}
                for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                    if finger == "thumb":
                        fingers[finger] = recognizer._is_thumb_extended(points, "Right")
                    else:
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
                        
                        fingers[finger] = (
                            straight_ratio > recognizer.FINGER_STRAIGHT_THRESHOLD
                            and wrist_ratio > recognizer.FINGER_WRIST_EXTENSION_RATIO
                        )
                
                # 分類手勢
                sign = recognizer._classify_sign(fingers, landmarks)
                
                if sign == "Unknown":
                    # 記錄 Unknown 的手指模式
                    pattern = tuple(fingers.values())
                    unknown_patterns.append(pattern)
                    
                    # 計算詳細特徵
                    thumb_spread = recognizer._thumb_spread_ratio(landmarks)
                    
                    print(f"\n=== Frame {frame_count}: Unknown 手勢 ===")
                    print(f"手指狀態: {fingers}")
                    print(f"拇指展開比例: {thumb_spread:.3f} (閾值: {recognizer.THUMB_SPREAD_THRESHOLD})")
                    
                    # 計算每個手指的詳細數值
                    for finger in ["index", "middle", "ring", "pinky"]:
                        base_idx = recognizer.FINGER_MCPS[finger]
                        mcp = points[base_idx]
                        pip = points[base_idx + 1]
                        dip = points[base_idx + 2]
                        tip = points[base_idx + 3]
                        
                        straight = recognizer._distance(mcp, tip)
                        segments = (
                            recognizer._distance(mcp, pip)
                            + recognizer._distance(pip, dip)
                            + recognizer._distance(dip, tip)
                        )
                        straight_ratio = straight / max(segments, 0.001)
                        
                        print(f"  {finger}: straight_ratio={straight_ratio:.3f}")
        
        frame_count += 1
    
    cap.release()
    recognizer.close()
    
    # 統計最常見的 Unknown 模式
    if unknown_patterns:
        print("\n" + "=" * 50)
        print("最常見的 Unknown 手勢模式:")
        pattern_counts = Counter(unknown_patterns)
        for pattern, count in pattern_counts.most_common(10):
            thumb, index, middle, ring, pinky = pattern
            print(f"  拇指:{'伸直' if thumb else '彎曲'} "
                  f"食指:{'伸直' if index else '彎曲'} "
                  f"中指:{'伸直' if middle else '彎曲'} "
                  f"無名指:{'伸直' if ring else '彎曲'} "
                  f"小指:{'伸直' if pinky else '彎曲'} "
                  f"-> {count} 次")
    
    return unknown_patterns

if __name__ == "__main__":
    video_path = "sign_language_app/raw_videos_backup/deaf/手語新手教室 第九課：購物、時態｜手語方向｜手語練習環節｜香港手語｜WeTV x 聾場蜜語 [rVdnKYFpQSg].mp4"
    analyze_unknown_gestures(video_path, max_frames=300)
