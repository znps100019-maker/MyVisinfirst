# MyVisinfirst

這是一個基於 OpenCV、MediaPipe 與 NumPy 實作的**手部動作與手勢偵測專案**。
本專案已完成精簡化整理，專注於手勢偵測，並支援直接執行 Python 檔案啟動。

## 目錄結構說明

- `main.py` - 主程式：啟動相機迴圈，進行實時手勢辨識與關節點繪製。
- `hand_detector.py` - 手勢辨識模組：負責 MediaPipe 手部關節偵測與靜態手勢分類演算法。
- `requirements.txt` - 套件依賴清單。
- `tools/` - 專案工具目錄：
  - `mingit-2.54.0/` - 專案內置的 Windows 免安裝可攜式 Git 環境。

---

## 環境安裝與設定

本專案預設使用 **Python 3.10** 環境。

1. **載入專案內置工具路徑（PowerShell 視窗）**：
   在執行任何 Git 或 Python 指令前，若您的系統沒有全域安裝 Git，可載入專案內置環境：
   ```powershell
   # 載入內置環境路徑（這會將專案內的 Python 與 Git 加入目前工作階段的環境變數）
   $env:Path = "$PWD\.venv\Scripts;$PWD\tools\mingit-2.54.0\cmd;$env:Path"
   ```

2. **驗證環境是否準備就緒**：
   ```powershell
   python --version
   git --version
   ```

---

## 執行方式

您可以直接使用 Python 來執行主程式，不需要透過複雜的啟動腳本：

```powershell
# 直接啟動主程式（開啟影像視窗與相機）
python main.py
```

### 💡 實用啟動引數參數 (Arguments)

您可以在執行 `main.py` 時附加參數來控制不同的功能：

1. **選擇指定的相機鏡頭（預設為 0）**：
   ```powershell
   python main.py --camera 1
   ```

2. **設定相機解析度（適合效能較低或特定開發板）**：
   ```powershell
   python main.py --width 640 --height 480
   ```

3. **無畫面模式（Headless）與輸出關節 JSON 數據**：
   如果您想在背景執行，將偵測到的手部 21 個關節點與穩定手勢輸出給其他程式或終端機讀取：
   ```powershell
   python main.py --headless --print-joints --print-interval 0.5
   ```
   *輸出範例（JSON 格式）：*
   ```json
   {
     "timestamp": 1780641234.56,
     "stable_sign": "Thumbs up",
     "stable": {"sign": "Thumbs up", "is_stable": true, "confidence": 1.0},
     "face_expression": "Smiling",
     "target_sign": "",
     "target_detected": false,
     "hands": [
       {
         "handedness": "Right",
         "sign": "Thumbs up",
         "fingers": {"thumb": true, "index": false, "middle": false, "ring": false, "pinky": false},
         "joints": {
           "wrist": {"x": 320, "y": 240, "z": 0.0},
           "thumb_tip": {"x": 350, "y": 120, "z": -0.05}
           ...
         }
       }
     ]
   }
   ```

4. **指定目標手勢判定（Target Sign）**：
   當辨識到特定手勢且達到穩定影格數時，觸發目標事件輸出：
   ```powershell
   python main.py --target-sign "Fist"
   ```
   *目標命中輸出：*
   ```json
   {"event": "target_detected", "timestamp": 1780641250.12, "target_sign": "Fist", "stable": {"sign": "Fist", "is_stable": true, "confidence": 1.0}}
   ```

5. **關閉臉部表情偵測（--no-face）**：
   如果您只想單獨執行手勢辨識以節省系統 CPU 運算資源，可以加入此參數關閉臉部表情偵測：
   ```powershell
   python main.py --no-face
   ```

---

## 目前支援的手勢與數字辨識種類

本系統採用符合**台灣手語（算盤結構）**與 3D 空間幾何判定，目前支援以下手勢：

### 1. 單手辨識 (1~9 與特殊手勢)
*   `Number 1` (食指伸直)
*   `Number 2` (食、中指伸直)
*   `Number 3` (食、中、無名指伸直)
*   `Number 4` (食、中、無名、小指伸直)
*   `Number 5 (Open palm)` (五指全開 / 同「開掌」)
*   `Number 6` (大拇指 + 小指伸直)
*   `Number 7` (大拇指 + 食指伸直)
*   `Number 8` (大拇指 + 食指 + 中指伸直)
*   `Number 9` (大拇指 + 食指 + 中指 + 無名指伸直)
*   `Thumbs up` (僅大拇指伸直 / 同「比讚」)
*   `OK` (OK 手勢：大拇指與食指捏合，其餘三指伸直)
*   `I love you` (我愛你手勢：大拇指、食指、小指伸直)
*   `Pinky` (比小指)
*   `Fist` (握拳)
*   `Unknown` (未知手勢)

### 2. 雙手併計辨識 (6~10)
當相機畫面同時偵測到兩隻手時，系統將**累加雙手伸直的手指總數**，支援以下累加數字：
*   `Number 6 (Two hands)` (雙手伸直手指總數為 6，例如：5 指 + 1 指)
*   `Number 7 (Two hands)` (雙手伸直手指總數為 7，例如：5 指 + 2 指)
*   `Number 8 (Two hands)` (雙手伸直手指總數為 8，例如：5 指 + 3 指)
*   `Number 9 (Two hands)` (雙手伸直手指總數為 9，例如：5 指 + 4 指)
*   `Number 10 (Two hands)` (雙手伸直手指總數為 10，即雙手五指全開)
*   *備註：若雙手手指總數小於 6，將會顯示 `"Two hands (X fingers)"`。*

### 3. 臉部表情辨識 (Facial Expression)
系統支援偵測以下臉部特徵與表情狀態：
*   `Neutral` (無表情 / 正常狀態)
*   `Smiling` (微笑)
*   `Mouth Open` (張開嘴巴 / 驚訝)
*   `Blink` (雙眼閉合 / 眨眼)
*   `Wink Left` (眨左眼：左眼閉合，右眼張開)
*   `Wink Right` (眨右眼：右眼閉合，左眼張開)

---

## 鍵盤快捷鍵說明 (Keyboard Hotkeys)

在影像視窗顯示狀態下，選中該視窗可以使用以下鍵盤快捷鍵：

*   **`q` 鍵**：安全結束程式、釋放鏡頭資源並關閉視窗（亦可直接點擊視窗右上角「X」安全關閉）。
*   **`f` 鍵**：開啟/關閉臉上的科技感 3D 藍色網格線（隱藏藍色網格線以利看清臉部表情，此時背景依然會持續進行表情偵測，並在左上角 HUD 框中顯示偵測結果）。
