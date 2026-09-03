"""改善拇指展開判斷邏輯"""
with open('hand_detector.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 查看目前的拇指展開判斷
old_thumb_spread = '''    def _thumb_spread_ratio(self, landmarks):
        points = landmarks.landmark
        thumb_tip = points[self.FINGER_TIPS["thumb"]]
        index_mcp = points[self.FINGER_MCPS["index"]]
        wrist = points[0]
        middle_mcp = points[self.FINGER_MCPS["middle"]]

        distance = self._distance(thumb_tip, index_mcp)
        palm_size = max(self._distance(wrist, middle_mcp), 0.001)
        return distance / palm_size'''

new_thumb_spread = '''    def _thumb_spread_ratio(self, landmarks):
        points = landmarks.landmark
        thumb_tip = points[self.FINGER_TIPS["thumb"]]
        index_mcp = points[self.FINGER_MCPS["index"]]
        wrist = points[0]
        middle_mcp = points[self.FINGER_MCPS["middle"]]
        index_tip = points[self.FINGER_TIPS["index"]]

        # 計算拇指與食指的夾角
        distance = self._distance(thumb_tip, index_mcp)
        palm_size = max(self._distance(wrist, middle_mcp), 0.001)
        spread_ratio = distance / palm_size
        
        # 計算拇指與食指尖端的距離（輔助判斷）
        thumb_index_distance = self._distance(thumb_tip, index_tip)
        thumb_index_ratio = thumb_index_distance / palm_size
        
        # 如果拇指和食指距離很遠，表示拇指展開
        return max(spread_ratio, thumb_index_ratio * 0.8)'''

if old_thumb_spread in content:
    content = content.replace(old_thumb_spread, new_thumb_spread)
    print("✓ 已改善拇指展開判斷")
else:
    print("✗ 未找到拇指展開判斷程式碼")

with open('hand_detector.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n拇指展開判斷改善完成！")
