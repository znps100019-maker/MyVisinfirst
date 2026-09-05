"""快速掃描多個影片，找出有伸直手指的影片"""
import cv2
import math
import glob
import os
from hand_detector import HandSignRecognizer

def analyze_video(video_path, sample_frames=20):
    """快速分析影片中的手指狀態"""
    cap = cv2.VideoCapture(video_path)
    recognizer = HandSignRecognizer()
    
    straight_count = 0
    curved_count = 0
    total_hands = 0
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 取樣間隔
    if total_frames > sample_frames:
        interval = total_frames // sample_frames
    else:
        interval = 1
    
    while frame_count < total_frames:
        success, img = cap.read()
        if not success:
            break
        
        # 只處理特定幀
        if frame_count % interval == 0:
            img = cv2.flip(img, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = recognizer.hands.process(rgb)
            
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    points = landmarks.landmark
                    total_hands += 1
                    
                    # 分析手指角度
                    angles = []
                    for finger in ["index", "middle", "ring", "pinky"]:
                        base_idx = recognizer.FINGER_MCPS[finger]
                        mcp = points[base_idx]
                        pip = points[base_idx + 1]
                        dip = points[base_idx + 2]
                        
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
                        if avg_angle > 120:
                            straight_count += 1
                        else:
                            curved_count += 1
        
        frame_count += 1
    
    cap.release()
    recognizer.close()
    
    return {
        "video": os.path.basename(video_path),
        "total_hands": total_hands,
        "straight": straight_count,
        "curved": curved_count,
        "straight_ratio": straight_count / max(total_hands, 1)
    }

# 掃描所有影片
video_files = glob.glob("sign_language_app/raw_videos_backup/deaf/*.mp4")
print(f"找到 {len(video_files)} 個影片檔案\n")
print("=" * 80)

results = []
for video in video_files:
    try:
        result = analyze_video(video, sample_frames=15)
        results.append(result)
        print(f"影片: {result['video'][:50]}...")
        print(f"  偵測到手: {result['total_hands']} 次")
        print(f"  伸直: {result['straight']} 次 ({result['straight_ratio']*100:.0f}%)")
        print(f"  彎曲: {result['curved']} 次")
        print("-" * 40)
    except Exception as e:
        print(f"影片: {os.path.basename(video)[:50]}...")
        print(f"  錯誤: {e}")
        print("-" * 40)

# 排序找出最適合的影片
if results:
    results.sort(key=lambda x: x["straight_ratio"], reverse=True)
    print("\n" + "=" * 80)
    print("最適合測試的影片（依伸直手指比例排序）:")
    for i, result in enumerate(results[:3], 1):
        print(f"{i}. {result['video']} (伸直比例: {result['straight_ratio']*100:.0f}%)")
