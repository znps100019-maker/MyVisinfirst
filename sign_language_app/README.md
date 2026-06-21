# 手語影片下載與 AI 辨識子系統 (Sign Language Downloader & AI Classifier)

這個資料夾是一個獨立的子系統，設計目的是**將「手語影片下載」、「特徵值擷取」、「模型訓練」與「實時辨識」完整分離**，方便您將此專案發佈至 GitHub，讓其他電腦也可以一鍵部署與使用。

## 📂 目錄檔案說明
1. `requirements.txt` - 本子專案所需的相依套件清單。
2. `downloader.py` - 下載手語影片。預設會下載您指定的 YouTube 手語單元一播放清單，並自動依影片名稱將影片歸類至 `raw_videos/<手語意思>/` 下。
3. `extract_dataset.py` - 讀取已下載的影片，使用 MediaPipe Hands 擷取手部的 21 個 3D 關節點，進行「平移與縮放歸一化」處理，並打包儲存至 `dataset.json`。
4. `train_classifier.py` - 讀取 `dataset.json` 特徵，編譯並生成 KNN 分類樣板資料庫 `model.json`。
5. `recognizer.py` - 載入 `model.json` 樣板庫，開啟相機，實時透過我們手寫的 KNN 演算法進行手語單字預測，並在畫面底部進行連貫手語翻譯。

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

### 步驟 2：下載手語影片
執行以下指令，系統會自動下載手語播放清單並依據影片名稱將影片分類到對應的手勢資料夾中：
```powershell
# 使用預設的「基礎手語單元一」播放清單進行下載與自動歸類
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
  - 按鍵盤 **`c` 鍵**：清空底部已串接的連貫手語語句。

---

## 🛠️ 技術優勢說明（適合分享於 GitHub）
* **無絕對路徑**：程式內所有檔案路徑均採用相對於程式腳本的動態路徑（Relative Path），複製到任何電腦直接執行即可運作。
* **低解析度下載**：`downloader.py` 限制只下載最低解析度的直接 mp4 串流，免除安裝 ffmpeg 的煩惱，且檔案極小、下載極快。
* **純演算法 KNN 實作**：使用 NumPy 實作鄰近點計算與投票機制，無須為了分類器而安裝龐大的 `scikit-learn` 套件。
* **路徑防呆機制**：繼承了 Windows 中文/非 ASCII 路徑的 Subdrive (Subst) 對應邏輯，防止 MediaPipe 因 Windows 路徑編碼 Bug 造成的異常崩潰。
