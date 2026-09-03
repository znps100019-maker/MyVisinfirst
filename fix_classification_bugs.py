"""修復手勢分類邏輯的問題"""
with open('hand_detector.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修復 1: OK 手勢判斷條件太嚴格
old_ok = '''        if self._is_ok_sign(landmarks) and middle and ring and pinky:
            return "OK (Zero / Can)"'''
new_ok = '''        if self._is_ok_sign(landmarks):
            return "OK (Zero / Can)"'''
content = content.replace(old_ok, new_ok)

# 修復 2: 添加更準確的數字手勢判斷
old_classify = '''        is_thumb_spread = self._thumb_spread_ratio(landmarks) > self.THUMB_SPREAD_THRESHOLD

        if not is_thumb_spread:
            if index and middle and ring and pinky:
                return "Number 4 (Salute)"
            if index and middle and ring and not pinky:
                return "Number 3"
            if index and middle and not ring and not pinky:
                return "Number 2 (Victory)"
            if index and not middle and not ring and not pinky:
                return "Number 1 (Secret)"
        else:
            if index and middle and ring and pinky:
                return "Number 5 (Hello / Greet)"
            if index and middle and ring and not pinky:
                return "Number 9"
            if index and middle and not ring and not pinky:
                return "Number 8"
            if index and not middle and not ring and not pinky:
                return "Number 7 (Gun)"
            if pinky and not index and not middle and not ring:
                return "Number 6"
            if not any([index, middle, ring, pinky]):
                return "Good / Male (Thumbs up)"

        return "Unknown"'''

new_classify = '''        # 計算伸直的手指數量
        extended_count = sum([index, middle, ring, pinky])
        
        # 特殊手勢判斷
        if thumb and not any([index, middle, ring, pinky]):
            return "Good / Male (Thumbs up)"
        if pinky and not any([thumb, index, middle, ring]):
            return "Bad / Female (Pinky)"
        if index and pinky and not thumb and not middle and not ring:
            return "Cow / Horns"
        if thumb and index and pinky and not middle and not ring:
            return "I love you"
        
        # 數字手勢判斷（更直觀）
        if not thumb or self._thumb_spread_ratio(landmarks) <= self.THUMB_SPREAD_THRESHOLD:
            # 拇指未展開的情況
            if extended_count == 4:
                return "Number 4 (Salute)"
            elif extended_count == 3:
                return "Number 3"
            elif extended_count == 2:
                if index and middle:
                    return "Number 2 (Victory)"
            elif extended_count == 1:
                if index:
                    return "Number 1 (Secret)"
        else:
            # 拇指展開的情況
            if extended_count == 4:
                return "Number 5 (Hello / Greet)"
            elif extended_count == 3:
                return "Number 9"
            elif extended_count == 2:
                if index and middle:
                    return "Number 8"
                elif index and pinky:
                    return "Number 7 (Gun)"
            elif extended_count == 1:
                if pinky:
                    return "Number 6"
        
        return "Unknown"'''

if old_classify in content:
    content = content.replace(old_classify, new_classify)
    print("✓ 已修復手勢分類邏輯")
else:
    print("✗ 未找到目標程式碼，可能需要手動修改")

with open('hand_detector.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n修復完成！")
