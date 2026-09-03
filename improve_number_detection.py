"""改善數字手勢判斷"""
with open('hand_detector.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 在 _classify_sign 中添加更精確的判斷
old_classify = '''        # 計算伸直的手指數量
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
        
        # 檢查拇指是否真的展開（使用多個特徵）
        thumb_spread_ratio = self._thumb_spread_ratio(landmarks)
        is_thumb_spread = thumb_spread_ratio > self.THUMB_SPREAD_THRESHOLD
        
        # 數字手勢判斷
        if is_thumb_spread:
            # 拇指展開的情況
            if extended_count == 4 and index and middle and ring and pinky:
                return "Number 5 (Hello / Greet)"
            elif extended_count == 3 and index and middle and ring:
                return "Number 9"
            elif extended_count == 2:
                if index and middle:
                    return "Number 8"
                elif index and pinky:
                    return "Number 7 (Gun)"
            elif extended_count == 1 and pinky:
                return "Number 6"
        else:
            # 拇指未展開的情況
            if extended_count == 4 and index and middle and ring and pinky:
                return "Number 4 (Salute)"
            elif extended_count == 3 and index and middle and ring:
                return "Number 3"
            elif extended_count == 2 and index and middle:
                return "Number 2 (Victory)"
            elif extended_count == 1 and index:
                return "Number 1 (Secret)"
        
        return "Unknown"'''

if old_classify in content:
    content = content.replace(old_classify, new_classify)
    print("✓ 已改善數字手勢判斷")
else:
    print("✗ 未找到分類程式碼")

with open('hand_detector.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n數字手勢判斷改善完成！")
