#!/bin/bash
# 手勢辨識測試腳本

VIDEO_PATH="sign_language_app/raw_videos_backup/deaf/手語短句 3 好 不好 聾人 聽障 聽人 聽不到 手語 說話 [FOcge8IAQJ4].mp4"

echo "=== 測試 1: 基本手勢辨識（無臉部偵測）==="
python main.py --video "$VIDEO_PATH" --headless --print-joints --no-face --max-frames 30

echo ""
echo "=== 測試 2: 包含臉部表情偵測 ==="
python main.py --video "$VIDEO_PATH" --headless --print-joints --max-frames 30

echo ""
echo "=== 測試 3: 目標手勢偵測 ==="
python main.py --video "$VIDEO_PATH" --headless --target-sign "Number 2 (Victory)" --max-frames 50

echo ""
echo "所有測試完成！"
