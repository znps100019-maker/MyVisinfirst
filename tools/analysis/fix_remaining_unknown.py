"""修復剩餘的 Unknown 手勢"""
with open('hand_detector.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 在 _classify_sign 中添加更多手勢判斷
old_classify = '''        # 特殊手勢判斷
        if thumb and not any([index, middle, ring, pinky]):
            return "Good / Male (Thumbs up)"
        if pinky and not any([thumb, index, middle, ring]):
            return "Bad / Female (Pinky)"
        if index and pinky and not thumb and not middle and not ring:
            return "Cow / Horns"
        if thumb and index and pinky and not middle and not ring:
            return "I love you"'''

new_classify = '''        # 特殊手勢判斷
        if thumb and not any([index, middle, ring, pinky]):
            return "Good / Male (Thumbs up)"
        if pinky and not any([thumb, index, middle, ring]):
            return "Bad / Female (Pinky)"
        if index and pinky and not thumb and not middle and not ring:
            return "Cow / Horns"
        if thumb and index and pinky and not middle and not ring:
            return "I love you"
        # 新增：拇指+食指+中指+小指（無名指彎曲的 I love you 變體）
        if thumb and index and middle and pinky and not ring:
            return "I love you"
        # 新增：拇指+食指（手槍手勢）
        if thumb and index and not middle and not ring and not pinky:
            thumb_spread = self._thumb_spread_ratio(landmarks)
            if thumb_spread > 0.7:
                return "Number 7 (Gun)"
            else:
                return "Number 8"'''

if old_classify in content:
    content = content.replace(old_classify, new_classify)
    print("✓ 已添加新的手勢判斷")
else:
    print("✗ 未找到目標程式碼")

# 調整拇指展開判斷的閾值
old_threshold = 'THUMB_SPREAD_THRESHOLD = 0.58'
new_threshold = 'THUMB_SPREAD_THRESHOLD = 0.55'  # 降低一點以更好地判斷
content = content.replace(old_threshold, new_threshold)

with open('hand_detector.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n剩餘 Unknown 手勢修復完成！")
