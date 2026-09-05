"""綜合測試 - 測試多個影片的準確度"""
import cv2
import glob
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.path_fix import handle_non_ascii_path
handle_non_ascii_path()

from collections import Counter
from core.detectors.hand_detector import HandSignRecognizer

def test_video_accuracy(video_path, max_frames=100):
    """測試單個影片的準確度"""
    cap = cv2.VideoCapture(video_path)
    recognizer = HandSignRecognizer()
    
    detected_signs = []
    frame_count = 0
    
    while frame_count < max_frames:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)
        detections = recognizer.process(img)
        
        for detection in detections:
            detected_signs.append(detection["sign"])
        
        frame_count += 1
    
    cap.release()
    recognizer.close()
    
    if not detected_signs:
        return None
    
    # 計算準確度
    sign_counts = Counter(detected_signs)
    unknown_count = detected_signs.count("Unknown")
    known_count = len(detected_signs) - unknown_count
    accuracy = known_count / len(detected_signs) * 100
    
    return {
        "video": os.path.basename(video_path)[:50],
        "total_detections": len(detected_signs),
        "known_ratio": accuracy,
        "unknown_ratio": 100 - accuracy,
        "unique_signs": len(set(detected_signs)),
        "top_signs": sign_counts.most_common(3)
    }

# 測試所有影片
video_files = glob.glob("sign_language_app/raw_videos_backup/deaf/*.mp4")
print(f"測試 {len(video_files)} 個影片...\n")

results = []
for video in video_files:
    try:
        result = test_video_accuracy(video, max_frames=50)
        if result:
            results.append(result)
            print(f"影片: {result['video']}...")
            print(f"  準確度: {result['known_ratio']:.1f}%")
            print(f"  偵測數: {result['total_detections']}")
            print(f"  手勢種類: {result['unique_signs']}")
            print(f"  最常見: {result['top_signs'][0][0] if result['top_signs'] else 'N/A'}")
            print("-" * 40)
    except Exception as e:
        print(f"影片: {os.path.basename(video)[:50]}...")
        print(f"  錯誤: {e}")
        print("-" * 40)

# 計算平均準確度
if results:
    avg_accuracy = sum(r['known_ratio'] for r in results) / len(results)
    print(f"\n平均準確度: {avg_accuracy:.1f}%")
    print(f"最佳影片: {max(results, key=lambda x: x['known_ratio'])['video']}")
    print(f"最差影片: {min(results, key=lambda x: x['known_ratio'])['video']}")
