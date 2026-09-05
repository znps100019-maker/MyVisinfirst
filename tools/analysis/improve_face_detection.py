"""改善臉部表情偵測"""
with open('face_detector.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 降低微笑判斷閾值
old_smile = 'is_smiling = smile_ratio > 0.56'
new_smile = 'is_smiling = smile_ratio > 0.50'
content = content.replace(old_smile, new_smile)

# 降低嘴巴張開閾值
old_mouth = 'is_mouth_open = mouth_ratio > 0.08'
new_mouth = 'is_mouth_open = mouth_ratio > 0.06'
content = content.replace(old_mouth, new_mouth)

# 添加更多表情判斷
old_expression = '''        # 4. Expressions Decision Tree
        expression = "Neutral"
        if is_left_closed and is_right_closed:
            expression = "Blink"
        elif is_left_closed:
            expression = "Wink Left"
        elif is_right_closed:
            expression = "Wink Right"
        elif is_mouth_open:
            expression = "Mouth Open"
        elif is_smiling:
            expression = "Smiling"'''

new_expression = '''        # 4. Expressions Decision Tree
        expression = "Neutral"
        if is_left_closed and is_right_closed:
            expression = "Blink"
        elif is_left_closed:
            expression = "Wink Left"
        elif is_right_closed:
            expression = "Wink Right"
        elif is_mouth_open and is_smiling:
            expression = "Laughing"
        elif is_mouth_open:
            expression = "Mouth Open"
        elif is_smiling:
            expression = "Smiling"'''

if old_expression in content:
    content = content.replace(old_expression, new_expression)
    print("✓ 已改善臉部表情判斷")
else:
    print("✗ 未找到表情判斷程式碼")

with open('face_detector.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n臉部表情改善完成！")
