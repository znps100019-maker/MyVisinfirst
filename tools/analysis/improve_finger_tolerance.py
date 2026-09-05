"""添加手指判斷的容錯機制"""
with open('hand_detector.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 在 _extended_fingers 方法中添加容錯
old_method = '''            # 自適應判斷：使用多個特徵
            is_straight = (
                straight_ratio > self.FINGER_STRAIGHT_THRESHOLD
                or angle > 150
            )
            
            is_extended_from_wrist = (
                wrist_ratio > self.FINGER_WRIST_EXTENSION_RATIO
                or (angle > 140 and wrist_ratio > 0.85)
            )
            
            fingers[finger] = is_straight and is_extended_from_wrist'''

new_method = '''            # 自適應判斷：使用多個特徵，添加容錯機制
            is_straight = (
                straight_ratio > self.FINGER_STRAIGHT_THRESHOLD
                or angle > 150
                or (straight_ratio > self.FINGER_STRAIGHT_THRESHOLD * 0.8 
                    and angle > 130)
            )
            
            is_extended_from_wrist = (
                wrist_ratio > self.FINGER_WRIST_EXTENSION_RATIO
                or (angle > 140 and wrist_ratio > 0.85)
                or (angle > 120 and wrist_ratio > self.FINGER_WRIST_EXTENSION_RATIO * 0.9)
            )
            
            fingers[finger] = is_straight and is_extended_from_wrist'''

if old_method in content:
    content = content.replace(old_method, new_method)
    print("✓ 已添加手指判斷容錯機制")
else:
    print("✗ 未找到目標方法")

with open('hand_detector.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n手指判斷容錯機制添加完成！")
