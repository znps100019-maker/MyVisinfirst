# 常用手語交流辨識流程

這一套流程用來建立「你好、謝謝、對不起、幫忙、不明白、廁所、醫院」等簡單交流手語的辨識模型。

## 1. 安裝套件

```powershell
pip install -r requirements.txt
pip install -r sign_language_app/requirements.txt
```

## 2. 下載範例影片

預設會使用專案內建的 YouTube 播放清單網址，並依照 `communication_signs.json` 的關鍵字自動分類到 `raw_videos/<label>/`。

```powershell
python sign_language_app/downloader.py
```

也可以指定其他影片或播放清單：

```powershell
python sign_language_app/downloader.py --url "https://www.youtube.com/watch?v=nE4kuhO0l3E&list=PLzI2EvXfsJoOJFf3f1aqjQj7LIIrWuKYu"
```

下載後請檢查 `sign_language_app/raw_videos/uncategorized/`。如果有影片沒有自動分類，手動移到正確資料夾即可。

## 3. 產生資料集並訓練

```powershell
python sign_language_app/extract_dataset.py
python sign_language_app/train_classifier.py
```

完成後會產生 `sign_language_app/model.json`。

## 4. 啟動攝影機辨識

```powershell
python sign_language_app/recognizer.py
```

快捷鍵：

- `q`: 離開
- `c`: 清除目前累積句子

## 5. 用影片測試

```powershell
python sign_language_app/video_recognizer.py --input "sign_language_app/raw_videos/hello/example.mp4"
```

或直接測 YouTube：

```powershell
python sign_language_app/video_recognizer.py --input "https://www.youtube.com/watch?v=nE4kuhO0l3E"
```

## 新增詞彙

到 `communication_signs.json` 加入新的項目：

```json
{
  "label": "new_label",
  "display": "中文顯示",
  "english": "English display",
  "keywords": ["影片標題可能出現的關鍵字"]
}
```

再把對應影片放到 `raw_videos/new_label/`，重新執行資料集產生與訓練即可。
