# 臉部表情辨識 (Facial Expression Recognition) 公開資料集指南與「喜怒哀樂」映射規範

本文件為 **MyVisinfirst 手勢與手語視覺辨識系統** 之臉部表情辨識模組（`core/detectors/face_detector.py`）所整理的**網路開源臉部表情資料庫（FER Datasets）指南**。
本系統支援傳統華人/台灣手語四大核心非手部訊號（NMS, Non-Manual Signals）：「**喜、怒、哀、樂**」與「**平靜**」之特徵映射與模型校準。

---

## 🌐 國際與台灣本土主流臉部表情開源資料庫

| 資料庫名稱 | 發布單位 / 來源 | 樣本規模 | 影像解析度 | 情緒標籤類別 | 特色與適用場景 | 下載連結 / 存取方式 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FER-2013** | ICML 2013 / Kaggle | 35,887 張 | 48 × 48 (灰階) | 7 種 (憤怒、厭惡、恐懼、開心、難過、驚訝、中性) | 國際最經典基準，輕量標準化，極易快速訓練與評測 | [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) 或 HuggingFace `Hemg/fer2013` |
| **AffectNet** | University of Denver | 420,000+ 張 (標註) | 真實多解析度 (彩色) | 8 種離散情緒 + 2 維連續維度 (Valence / Arousal) | 全球規模最大之自然環境 (In-the-wild) 人臉情緒庫 | [AffectNet 官網](http://mohammadmahoor.com/affectnet/) / [HuggingFace AffectNet](https://huggingface.co/datasets/AffectNet) |
| **TFEID (台灣本土)** | 陽明交大 / 台大腦與心智實驗室 | 6,604+ 張 | 高解析度彩色 / 灰階 | 6 大基本情緒 + 輕蔑 + 中性 (含人臉 FACS 評估) | **強烈推薦**：專為台灣/東亞族群設計，極契合台灣手語本土化面部特徵研究 | [TFEID 官方首頁](https://bml.ym.edu.tw/tfeid/) (學術研究免費申請) |
| **CK+ (Extended Cohn-Kanade)** | Carnegie Mellon University | 593 序列影片 (327 標註) | 640 × 490 (彩色/灰階) | 7 種情緒 + FACS Action Units (AU) 肌肉動作元 | 實驗室標準控制環境，具備每幀臉部肌肉動作 AU 精確標註 | [CK+ 官方頁面](https://www.jeffcohn.net/Resources/) / Kaggle 鏡像 |
| **RAF-DB** | 國際學術研究 | 29,672 張 | 真實網路照片 (彩色) | 7 種基本情緒 + 12 種複合情緒 (Compound Emotions) | 真實生活非受限多樣角度、年齡、性別與光影豐富 | [RAF-DB Dataset](http://www.whdeng.cn/RAF/model1.html) |
| **JAFFE** | 九州大學 (日本) | 213 張 | 256 × 256 (灰階) | 7 種基本情緒 (東亞女性受試者) | 經典東亞臉型微表情早期基準資料庫 | [JAFFE 官方網站](https://zenodo.org/record/3451524) |

---

## 🎯 標準 7 大情緒標籤與「喜、怒、哀、樂、平靜」映射規範

在傳統華人文化與台灣手語中，**「喜怒哀樂」** 是表達情感與文法語氣的最核心維度。國際標準資料庫（如 FER-2013、AffectNet）通常標註 7 類基本情緒，其映射轉換對照如下：

```mermaid
graph LR
    subgraph 國際標準 7 類情緒 (FER Standard)
        H[Happiness 開心]
        A[Anger 憤怒]
        S[Sadness 悲傷]
        Sur[Surprise 驚訝]
        N[Neutral 中性]
        D[Disgust 厭惡]
        F[Fear 恐懼]
    end

    subgraph 本專案五大情緒分類 (Emotion Classes)
        XI["😊 喜 (Happy / 微笑)"]
        NU["😠 怒 (Angry / 生氣)"]
        AI["😢 哀 (Sad / 難過)"]
        LE["😄 樂 (Joy / 大笑)"]
        NEU["😐 平靜 (Neutral / 放鬆)"]
    end

    H -->|嘴角微揚| XI
    H -->|張嘴大笑/高Arousal| LE
    Sur -->|歡樂驚喜| LE
    A --> NU
    D -->|皺眉排斥| NU
    S --> AI
    F -->|無助畏懼| AI
    N --> NEU
```

### 臉部肌肉動作元 (FACS Action Units) 幾何判定對照表

本系統 `core/detectors/face_detector.py` 透過 **MediaPipe FaceMesh (478 個 3D 關節點)** 進行無延遲即時運算：

| 情感分類 | 對應 FACS 動作元 (Action Units) | 關鍵點位 (Landmarks) | 幾何運算指標與門檻 |
| :--- | :--- | :--- | :--- |
| **喜 (Happy)** | **AU12** (提口角肌) + **AU6** (提頰肌) | 嘴角 (61, 291)、唇中心 (13, 14) | 嘴角上揚 `corner_elevation > +0.015` 或 `smile_ratio > 0.47`，嘴巴未大張 |
| **怒 (Angry)** | **AU4** (皺眉肌/降眉肌) + **AU7** (緊閉瞼板肌) | 眉心 (107, 336)、眼角 (133, 362) | 眉心距離縮短 `d_brow_inner < 0.315` 且眉眼壓低 `avg_brow_eye_dist < 0.20` |
| **哀 (Sad)** | **AU15** (降口角肌) + **AU1** (額肌內側部) | 嘴角 (61, 291)、內外眉 (107, 70, 336, 300) | 嘴角下垂 `corner_elevation < -0.012` 或八字眉傾斜 `avg_brow_slant < -0.010` |
| **樂 (Joy)** | **AU12** (提口角) + **AU25/26** (張嘴/下頷垂下) | 嘴角 (61, 291)、內唇 (13, 14) | 笑容展開且大張嘴 `mouth_ratio > 0.048` 且 `smile_ratio > 0.48` |
| **平靜 (Neutral)** | 各部肌肉放鬆基線 | 全臉 | 未達上述各情緒觸發門檻，回傳基線信心度 |

---

## 💻 如何下載與使用開源資料集？

### 方法 1：使用本專案內建下載與特徵提取工具 (推薦)

專案已內建 [`tools/download_expression_dataset.py`](file:///c:/Users/USER/OneDrive/桌面/MyVisinfirst/tools/download_expression_dataset.py)：

```powershell
# 1. 自動下載精選「喜、怒、哀、樂、平靜」測試樣本影像
python tools/download_expression_dataset.py --download-samples

# 2. 自動執行 MediaPipe FaceMesh 特徵提取並評測辨識率
python tools/download_expression_dataset.py --benchmark
```

### 方法 2：使用 Kaggle API 下載完整 FER-2013 資料集 (35,887 張)

若欲訓練深度卷積神經網路 (CNN) 或 SVM 分類器，可下載 Kaggle FER-2013：

```powershell
# 安裝 Kaggle CLI
pip install kaggle

# 下載 FER-2013 資料集 zip
kaggle datasets download -d msambare/fer2013 -p sign_language_app/datasets/

# 解壓縮
tar -xf sign_language_app/datasets/fer2013.zip -C sign_language_app/datasets/fer2013/
```

### 方法 3：申請台灣本土 TFEID 資料庫 (適合專案成果論文/報告)

台灣手語與華人面部微表情非常推薦引用 **TFEID (Taiwanese Facial Expression Image Database)**：
- 官方申請網址：[https://bml.ym.edu.tw/tfeid/](https://bml.ym.edu.tw/tfeid/)
- 聯絡信箱：`tfeidnycu@gmail.com`（國立陽明交通大學 腦與心智實驗室）
- 申請時附上大專生專題/學術研究說明即可免費獲取授權與完整高解析度照片。
