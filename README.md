# MyVisinfirst

這是一個基於 OpenCV、MediaPipe 與 NumPy 實作的**手部動作與手勢偵測專案**。
本專案已完成精簡化整理，專注於手勢偵測，並支援直接執行 Python 檔案啟動。

## 目錄結構說明

- `main.py` - 主程式：啟動相機迴圈，進行實時手勢辨識與關節點繪製。
- `run_web.py` - 網頁展示伺服器：一鍵啟動專案專屬的台灣手語互動教學與測驗網頁平台。
- `web/` - 網頁展示系統完整原始碼（支援純前端離線執行、MediaPipe 126D 骨架與 KNN 辨識）。
- `core/detectors/hand_detector.py` - 手部關節偵測、保守的靜態手形分類與穩定狀態機。
- `core/evaluation.py` - 有人工標註才計算準確率；沒有標註只報告覆蓋率與分布。
- `requirements.txt` - 套件依賴清單。
- `tools/` - 專案工具目錄：
  - `mingit-2.54.0/` - 專案內置的 Windows 免安裝可攜式 Git 環境。

---

## 🌐 網頁專屬展示平台 (Web Showcase & Interactive Platform)

專題支援純瀏覽器執行的專屬互動網頁，整合 **MediaPipe Hands** 與 **126 維度 KNN 台灣手語辨識模型**（支援 519 種常用手語詞彙），無須安裝任何伺服器或額外環境，直接在瀏覽器運作！

### 🚀 一鍵本機啟動網頁

```powershell
python run_web.py
```
> 執行後將自動開啟瀏覽器前往 `http://127.0.0.1:8080/web/`。允許攝影機存取後即可開始體驗。

### 🌟 五大核心功能模組

1. **實時手語辨識 (Real-time Recognition)**：
   - 21 個手部關節點霓虹發光骨架與指尖動態軌跡
   - 126 維度特徵向量提取與歸一化過程可視化
   - KNN Top-9 候選詞距離、機率分佈條與五指伸直狀態晶片
   - 即時信心度儀表板、時序穩定性進度條與句子產生器

2. **手語互動教室 (Sign Language Classroom)**：
   - 收錄日常問候、數字、情感、動物等多分類詞彙
   - AI「跟我做」即時手勢比對，辨識成功即獲正向音效反饋

3. **AI 闖關測驗 (Interactive Quiz Game)**：
   - 10 秒倒數動態計時條、動態隨機出題
   - 連擊 Combo 獎勵、Web Audio 擬真音效與結算證書

4. **519 詞庫大字典 (Sign Dictionary)**：
   - 完整 519 種台灣手語詞彙即時搜尋與分類過濾
   - 單雙手指標籤與樣本數據統計

5. **學習歷程記錄 (Learning Stats)**：
   - 本地儲存學習歷程、最高得分、測驗通關率與已掌握詞彙庫

### 📦 GitHub Pages 部署說明
專案根目錄已配置自動導向 `index.html`，只要在 GitHub 倉庫設定中啟用 **Settings → Pages → Source: Deploy from a branch (`main` / root)**，即可直接在線上存取專屬展示網頁！

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

## 執行方式 (Python 原生相機辨識)

所有即時辨識入口都集中在 `main.py`，不需要先執行其他辨識腳本：

```powershell
# 不帶參數：出現提示後直接按 Enter，開啟預設攝像頭做人員偵測
python main.py

# 直接辨識本機影片
python main.py path/to/video.mp4

# 直接辨識 YouTube 影片（預設先下載再辨識）
python main.py --video "https://www.youtube.com/watch?v=nE4kuhO0l3E"
```

執行 `python main.py` 後，若直接按 Enter，程式會使用攝像頭偵測鏡頭前的人；只有貼上影片路徑或 YouTube URL 才會進入影片辨識。請從 PowerShell 或命令提示字元執行，這樣可以看到攝像頭初始化與錯誤訊息；不要直接雙擊 `main.py`。如果預設攝像頭 0 無法開啟，可以嘗試：

```powershell
python main.py --camera 1
```

若仍無法開啟，請到 Windows「設定 → 隱私權與安全性 → 相機」允許桌面應用程式使用相機，並關閉 Teams、Zoom 或其他正在使用相機的程式。

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

3. **調整即時顯示視窗大小**：
   ```powershell
   python main.py --window-width 1280 --window-height 720

   # 或直接使用全螢幕
   python main.py --fullscreen
   ```

4. **指定 YouTube 處理方式**：
   ```powershell
   # 下載後播放，較穩定
   python main.py --video "https://www.youtube.com/watch?v=nE4kuhO0l3E" --stream-mode download

   # 直接使用線上串流
   python main.py --video "https://www.youtube.com/watch?v=nE4kuhO0l3E" --stream-mode stream
   ```

5. **無畫面模式（Headless）與輸出關節 JSON 數據**：
   如果您想在背景執行，將偵測到的手部 21 個關節點與穩定手勢輸出給其他程式或終端機讀取：
   ```powershell
   python main.py --headless --print-joints --print-interval 0.5
   ```
   *輸出範例（JSON 格式）：*
   ```json
   {
     "timestamp": 1780641234.56,
     "stable_sign": "Thumbs up",
     "stable": {"sign": "Good / Male (Thumbs up)", "is_stable": true, "consistency": 1.0},
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

6. **指定目標手勢判定（Target Sign）**：
   當辨識到特定手勢且達到穩定影格數時，觸發目標事件輸出：
   ```powershell
   python main.py --target-sign "Fist"
   ```
   *目標命中輸出：*
   ```json
    {"event": "target_detected", "timestamp": 1780641250.12, "target_sign": "Fist", "stable": {"sign": "Fist (Solidarity)", "is_stable": true, "consistency": 1.0}}
   ```

7. **關閉臉部表情偵測（--no-face）**：
   如果您只想單獨執行手勢辨識以節省系統 CPU 運算資源，可以加入此參數關閉臉部表情偵測：
   ```powershell
   python main.py --no-face
   ```

---

## 目前支援的手勢與數字辨識種類

本系統目前是靜態手形分類器，不擴充完整手語翻譯。畫面顯示數字或手指組合描述，並將連續結果稱為「手勢紀錄」。

### 1. 單手辨識 (手語與數字雙重語意)
*   `Number 1 (Secret)` (食指伸直 / 台灣手語：一、秘密（貼唇）)
*   `Number 2 (Victory)` (食指與中指伸直 / 通用手語：二、勝利、和平)
*   `Number 3` (食、中、無名指伸直 / 三)
*   `Number 4 (Salute)` (四指伸直，大拇指收起 / 台灣手語：四、敬禮/尊敬)
*   `Number 5 (Hello / Greet)` (五指全開)
*   `Number 6` (大拇指 + 小指伸直 / 六)
*   `Number 7 (Gun)` (大拇指 + 食指伸直 / 台灣手語：七、槍/手槍手勢)
*   `Number 8` (大拇指 + 食指 + 中指伸直 / 八)
*   `Number 9` (大拇指 + 食指 + 中指 + 無名指伸直 / 九)
*   `Good / Male (Thumbs up)` (僅大拇指伸直 / 台灣手語：男性、好、讚)
*   `Bad / Female (Pinky)` (僅小指伸直 / 台灣手語：女性、壞、差)
*   `I love you` (大拇指 + 食指 + 小指伸直，中指與無名指收起 / 通用手語：我愛你)
*   `Cow / Horns` (食指 + 小指伸直，其餘收起 / 台灣手語：牛、動物之角)
*   `OK (Zero / Can)` (大拇指與食指捏合，且中指、無名指、小指伸直)
*   `Fist (Solidarity)` (握拳 / 台灣手語：零、團結)
*   `Unknown` (未知手勢)

### 2. 雙手辨識
從 `main.py` 啟動時預設啟用雙手合併模式，會累加雙手伸直的手指總數，支援以下累加數字：
*   `Number 6 (Two hands)` (雙手伸直手指總數為 6，例如：5 指 + 1 指)
*   `Number 7 (Two hands)` (雙手伸直手指總數為 7，例如：5 指 + 2 指)
*   `Number 8 (Two hands)` (雙手伸直手指總數為 8，例如：5 指 + 3 指)
*   `Number 9 (Two hands)` (雙手伸直手指總數為 9，例如：5 指 + 4 指)
*   `Number 10 (Two hands)` (雙手伸直手指總數為 10，即雙手五指全開)
*   *備註：若雙手手指總數小於 6，將會顯示 `"Two hands (X fingers)"`。需要分開顯示時，請使用 `--separate-hands`。*

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
*   **`c` 鍵**：清除底部目前已確認的手勢紀錄。
