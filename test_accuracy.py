"""測試手勢辨識準確度"""
import cv2
import json
from collections import Counter
from hand_detector import HandSignRecognizer

def test_gesture_accuracy(video_path, max_frames=100):
    """測試手勢辨識準確度"""
    cap = cv2.VideoCapture(video_path)
    recognizer = HandSignRecognizer()
    
    detected_signs = []
    stable_signs = []
    
    frame_count = 0
    while frame_count < max_frames:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)
        detections = recognizer.process(img)
        
        for detection in detections:
            detected_signs.append(detection["sign"])
        
        stable_sign = recognizer.stable_status["sign"]
        if stable_sign != "No hand":
            stable_signs.append(stable_sign)
        
        frame_count += 1
    
    cap.release()
    recognizer.close()
    
    print("=" * 50)
    print("手勢辨識準確度測試")
    print("=" * 50)
    
    if detected_signs:
        print("\n偵測到的手勢分布:")
        sign_counts = Counter(detected_signs)
        for sign, count in sign_counts.most_common(10):
            percentage = count / len(detected_signs) * 100
            print(f"  {sign}: {count} 次 ({percentage:.1f}%)")
    
    if stable_signs:
        print("\n穩定手勢分布:")
        stable_counts = Counter(stable_signs)
        for sign, count in stable_counts.most_common(5):
            percentage = count / len(stable_signs) * 100
            print(f"  {sign}: {count} 次 ({percentage:.1f}%)")
    
    print(f"\n總共處理幀數: {frame_count}")
    print(f"偵測到手部的幀數: {len(detected_signs)}")
    print(f"穩定手勢的幀數: {len(stable_signs)}")
    
    # 計算準確度指標
    if detected_signs:
        unknown_count = detected_signs.count("Unknown")
        known_count = len(detected_signs) - unknown_count
        print(f"\n準確度指標:")
        print(f"  已知手勢比例: {known_count / len(detected_signs) * 100:.1f}%")
        print(f"  未知手勢比例: {unknown_count / len(detected_signs) * 100:.1f}%")

if __name__ == "__main__":
    video_path = "sign_language_app/raw_videos_backup/deaf/手語新手教室 第九課：購物、時態｜手語方向｜手語練習環節｜香港手語｜WeTV x 聾場蜜語 [rVdnKYFpQSg].mp4"
    test_gesture_accuracy(video_path, max_frames=200)
