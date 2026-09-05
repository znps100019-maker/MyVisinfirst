"""恢復到合理的閾值設定"""
with open('hand_detector.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 恢復到合理的中間值
improvements = {
    'FINGER_STRAIGHT_THRESHOLD = 0.40': 'FINGER_STRAIGHT_THRESHOLD = 0.55',
    'FINGER_WRIST_EXTENSION_RATIO = 0.70': 'FINGER_WRIST_EXTENSION_RATIO = 0.90',
    'THUMB_STRAIGHT_THRESHOLD = 0.50': 'THUMB_STRAIGHT_THRESHOLD = 0.65',
}

for old, new in improvements.items():
    if old in content:
        content = content.replace(old, new)
        print(f"✓ 已修改: {old} -> {new}")

with open('hand_detector.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n已恢復到合理閾值！")
