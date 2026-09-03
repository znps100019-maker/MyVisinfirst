"""改善手指偵測閾值"""
import re

with open('hand_detector.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 調整閾值
improvements = {
    'FINGER_STRAIGHT_THRESHOLD = 0.82': 'FINGER_STRAIGHT_THRESHOLD = 0.65',  # 降低伸直判斷標準
    'FINGER_WRIST_EXTENSION_RATIO = 1.1': 'FINGER_WRIST_EXTENSION_RATIO = 1.0',  # 放寬手腕延伸要求
    'THUMB_STRAIGHT_THRESHOLD = 0.86': 'THUMB_STRAIGHT_THRESHOLD = 0.75',  # 降低拇指伸直標準
    'min_detection_confidence=0.7': 'min_detection_confidence=0.5',  # 提高偵測敏感度
    'min_tracking_confidence=0.6': 'min_tracking_confidence=0.4',  # 提高追蹤敏感度
    'stable_min_count=5': 'stable_min_count=3',  # 減少穩定所需幀數
}

for old, new in improvements.items():
    if old in content:
        content = content.replace(old, new)
        print(f"✓ 已修改: {old} -> {new}")
    else:
        print(f"✗ 未找到: {old}")

with open('hand_detector.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n改善完成！")
