# CI 說明

目前 CI 只做兩件事：

1. 在 Python 3.10 與根目錄固定版本依賴下編譯 `main.py`、`core`、`sign_language_app` 與 `tests`。
2. 執行 `python -m unittest discover -s tests -p "test_*.py"`，驗證分類、幾何尺度、狀態切換、雙手模式與人工標註評估規則。

真實影片的準確率不會在沒有人工標註時被 CI 偽造；需要評估影片時使用：

```powershell
python tests/test_accuracy.py --video path/to/video.mp4 --label "Number 2 (Victory)"
python tests/comprehensive_test.py --video-dir path/to/labelled_videos
```
