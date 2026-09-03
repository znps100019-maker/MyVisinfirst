# 手勢辨識改善紀錄

## 改善日期
2026-09-04

## 改善內容

### 1. 調整手指伸直判斷閾值
- `FINGER_STRAIGHT_THRESHOLD`: 0.82 → 0.55
- 原因：原閾值太嚴格，導致伸直手指被判斷為彎曲
- 效果：成功辨識出更多伸直手指

### 2. 調整手腕延伸比率
- `FINGER_WRIST_EXTENSION_RATIO`: 1.1 → 0.90
- 原因：原比率要求太高，手指稍微彎曲就無法通過
- 效果：更好地判斷手指是否真正延伸

### 3. 調整拇指伸直閾值
- `THUMB_STRAIGHT_THRESHOLD`: 0.86 → 0.65
- 原因：原閾值對拇指的判斷過於嚴格
- 效果：更準確地判斷拇指狀態

### 4. 提高偵測敏感度
- `min_detection_confidence`: 0.7 → 0.5
- `min_tracking_confidence`: 0.6 → 0.4
- 效果：更容易偵測到手部

### 5. 加快穩定判斷
- `stable_min_count`: 5 → 3
- 效果：更快確認手勢穩定性

## 測試結果

### 適合測試的影片
1. 手語新手教室 第九課（伸直比例 89%）
2. 手語新手教室 第七課（伸直比例 75%）
3. 手語新手教室 第八課（伸直比例 73%）

### 成功辨識的手勢
- Good / Male (Thumbs up)
- Number 4 (Salute)
- Number 5 (Hello / Greet)
- OK (Zero / Can)
- Number 10 (Two hands)

## 新增工具
- `test_gesture.sh` - 自動化測試腳本
- `debug_fingers.py` - 手指偵測調試工具
- `analyze_finger_curvature.py` - 手指彎曲度分析
- `quick_video_scan.py` - 快速掃描適合的測試影片
