# 手語影片下載與 AI 辨識子系統 (Sign Language Downloader & AI Classifier)

這個資料夾是一個獨立的子系統，設計目的是**將「手語影片下載」、「特徵值擷取」、「模型訓練」與「實時辨識」完整分離**，方便您將此專案發佈至 GitHub，讓其他電腦也可以一鍵部署與使用。

## 📂 目錄檔案說明
1. `requirements.txt` - 本子專案所需的相依套件清單。
2. `downloader.py` - 影片下載工具。支援臺灣手語詞庫 (`idl`) 與 YouTube (`youtube`) 模式，並依名稱分類至 `raw_videos/<手語意思>/`。
3. `extract_dataset.py` - 讀取已下載的影片，使用 MediaPipe Hands 擷取手部的 21 個 3D 關節點，進行「平移與縮放歸一化」處理，並打包儲存至 `dataset.json`。
4. `train_classifier.py` - 讀取 `dataset.json` 特徵，編譯並生成 KNN 分類樣板資料庫 `model.json`。
5. `recognizer.py` - 相機、影片與 YouTube URL 的手形分類工具；影片模式會輸出時間軸字幕。它不是完整手語句子翻譯器。

---

## 🚀 快速開始使用指南（跨平台部署）

請在專案根目錄下，開啟終端機並依序執行以下步驟：

### 步驟 1：載入環境變數與安裝套件
如果您沒有全域安裝 Git/Python，請先在 PowerShell 載入專案內置路徑：
```powershell
# 載入內置環境路徑（PowerShell 視窗）
$env:Path = "$PWD\.venv\Scripts;$PWD\tools\mingit-2.54.0\cmd;$env:Path"

# 安裝子專案所需的相依套件 (包括影片下載用的 yt-dlp)
pip install -r sign_language_app/requirements.txt
```

### 步驟 2：下載手語影片 (二選一)

**選項 A：下載「臺灣手語詞庫」影片 (推薦)**
執行以下指令，透過下載器取得詞庫影片：
```powershell
python sign_language_app/downloader.py --mode idl
```

**選項 B：下載 YouTube 教學播放清單影片**
```powershell
python sign_language_app/downloader.py
```
*（影片將會儲存在 `sign_language_app/raw_videos/` 下）*

### 步驟 3：批次擷取手部關節特徵
讀取剛剛下載的影片，進行平移與比例歸一化，生成 63 維特徵資料庫：
```powershell
python sign_language_app/extract_dataset.py
```
*（特徵將會儲存在 `sign_language_app/dataset.json`）*

### 步驟 4：編譯模型樣板庫
將特徵資料編譯成極輕量且速度極快的 KNN 分類樣板：
```powershell
python sign_language_app/train_classifier.py
```
*（模型將會儲存在 `sign_language_app/model.json`）*

### 步驟 5：啟動相機進行實時手語辨識
```powershell
python sign_language_app/recognizer.py
```
* **鍵盤快捷鍵**：
  - 按鍵盤 **`q` 鍵** 或點選視窗右上角「X」：安全退出程式。
  - 按鍵盤 **`c` 鍵**：清空底部已串接的手勢紀錄。

### 步驟 6：辨識本機影片或網路上的手語影片
您可以傳入本機影片檔案路徑，或者**直接傳入 YouTube 網址**，程式會播放影片並顯示手形分類：
```powershell
# 1. 辨識本機下載的手語影片
python sign_language_app/recognizer.py --input sign_language_app/raw_videos/hello/xxxx.mp4

# 2. 辨識網路上的 YouTube 影片（即時串流）
python sign_language_app/recognizer.py --input "https://www.youtube.com/watch?v=nE4kuhO0l3E"
```

影片辨識的畫面顯示「手形分類」與「時間一致率」，不把靜態手形直接宣稱為詞義；本機影片另會輸出同名 `.srt` 與 `_timeline.txt`。沒有人工標註時只能報告分類分布與覆蓋率，不能稱為準確率。

---

## 🛠️ 技術優勢說明（適合分享於 GitHub）
* **無絕對路徑**：程式內所有檔案路徑均採用相對於程式腳本的動態路徑（Relative Path），複製到任何電腦直接執行即可運作。
* **低解析度下載**：`downloader.py` 限制只下載最低解析度的直接 mp4 串流，免除安裝 ffmpeg 的煩惱，且檔案極小、下載極快。
* **純演算法 KNN 實作**：使用 NumPy 實作鄰近點計算與投票機制，無須為了分類器而安裝龐大的 `scikit-learn` 套件。
* **路徑防呆機制**：繼承了 Windows 中文/非 ASCII 路徑的 Subdrive (Subst) 對應邏輯，防止 MediaPipe 因 Windows 路徑編碼 Bug 造成的異常崩潰。
