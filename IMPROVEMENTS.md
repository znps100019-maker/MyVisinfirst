# 手形辨識修正紀錄

## 實際呼叫流程

`main.py` 開啟相機或影片，逐幀呼叫 `core.detectors.hand_detector.HandSignRecognizer.process()`；
MediaPipe 產生 landmarks 後，先由 `_extended_fingers()` 判定五根手指，再由
`_classify_sign()` 產生當幀候選，最後由 `_record_candidate()` 與 `stable_status` 處理時間一致率、目標事件與手勢紀錄。

## 修正與驗證

| 問題案例 | 修改理由與預期結果 | 驗證 |
| --- | --- | --- |
| 拇食指靠近的握拳被判 OK | OK 現在必須同時滿足中指、無名指、小指伸直；握拳只會是握拳 | `test_fist_with_thumb_index_close_is_not_ok`、`test_ok_requires_three_other_extended_fingers` |
| 拇指+食指被判 8 | 移除距離分支；專案定義固定為 7，只有加上中指才是 8 | `test_number_seven_and_eight_follow_project_definition` |
| 拇指狀態被第二套距離規則覆寫 | 數字分類只使用 `_extended_fingers()` 的拇指結果 | `test_thumb_state_is_not_reinferred_from_distance` |
| 半彎手指被當伸直 | 加入 PIP/DIP 關節角度、指節直線比例與手腕延伸比例，對遮擋採保守 Unknown | `test_bent_finger_is_not_reported_as_extended` |
| 不同影像寬高比造成幾何結果不同 | 幾何運算前將 x/z 轉影像寬度、y 轉影像高度；JSON 與繪圖座標不變 | `test_geometry_is_invariant_to_frame_aspect_ratio` |
| 無手仍沿用舊目標、A/B 切換落後 | stable 狀態以當幀候選為主，無手立即停止目標；連續無手達門檻才重設歷史 | `test_switch_does_not_keep_old_stable_target` |
| 相同手勢重複紀錄異常 | 長時間無手會重新武裝，單幀掉追蹤不會重複加入手勢紀錄 | `test_long_no_hand_allows_same_sign_to_be_recorded_again`、`test_short_tracking_drop_does_not_duplicate_same_sign` |
| 雙手被默認合計 | 預設顯示 `Multiple hands` 與各手結果；只有 `--combine-two-hands` 才合計 | `test_two_hands_are_separate_by_default` |
| 一致率被誤叫信心或準確率 | 畫面標示「時間一致率」與「確認中/已確認」；人工標註才計算 accuracy | `test_one_frame_has_100_percent_consistency_but_is_not_confirmed`、evaluation tests |

## 評估規則

`core.evaluation.evaluate_predictions()` 在沒有人工標註時只回傳分布與覆蓋率；有標註時才回傳逐筆比較的準確率與錯誤清單。空影片、無法開啟的影片與零影格結果會使影片評估失敗。

## 範圍

獨立 KNN 訓練與 `model.json` 格式未修改；KNN 在主程式改為 `--use-knn` 明確啟用。這次修正仍是靜態手形分類，不宣稱完整手語翻譯。
