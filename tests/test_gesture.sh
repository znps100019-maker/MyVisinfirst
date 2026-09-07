#!/usr/bin/env bash
# 手勢辨識冒煙測試。需傳入一個可讀取的影片路徑。
set -euo pipefail

VIDEO_PATH="${1:?用法：bash tests/test_gesture.sh path/to/video.mp4}"
MAX_FRAMES="${MAX_FRAMES:-30}"

echo "=== 測試 1: 基本手勢辨識（無臉部偵測）==="
python main.py --video "$VIDEO_PATH" --headless --print-joints --no-face --max-frames "$MAX_FRAMES"

echo ""
echo "=== 測試 2: 包含臉部表情偵測 ==="
python main.py --video "$VIDEO_PATH" --headless --print-joints --no-face --max-frames "$MAX_FRAMES"

echo ""
echo "=== 測試 3: 目標手勢偵測 ==="
python main.py --video "$VIDEO_PATH" --headless --no-face --target-sign "Number 2 (Victory)" --max-frames "$MAX_FRAMES"

echo ""
echo "所有測試完成！"
