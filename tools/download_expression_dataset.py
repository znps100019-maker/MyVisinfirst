"""
臉部表情資料集下載與「喜怒哀樂」特徵評測工具
提供下載開源資料集 (FER-2013, AffectNet, TFEID 資訊) 與測試樣本影像，
並支援 MediaPipe 478 點特徵抽取與分類基準測試。
"""
import argparse
import json
import os
import sys
import urllib.request

# 確保專案根目錄加入 sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# 確保 Windows cp950 終端機正常輸出中文與 emoji
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


ONLINE_DATASETS = [
    {
        "name": "FER-2013 (Facial Expression Recognition 2013)",
        "source": "Kaggle / ICML 2013 Challenge",
        "scale": "35,887 張灰階人臉影像 (48x48)",
        "classes": "Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral",
        "url": "https://www.kaggle.com/datasets/msambare/fer2013",
        "command": "kaggle datasets download -d msambare/fer2013",
        "recommendation": "適合輕量快速訓練與基準測試，標籤對應明確。"
    },
    {
        "name": "AffectNet (In-the-Wild Facial Expressions)",
        "source": "University of Denver",
        "scale": "420,000+ 張自然場景人臉影像",
        "classes": "8 種基本情緒 + 效價 (Valence) / 喚醒度 (Arousal)",
        "url": "http://mohammadmahoor.com/affectnet/",
        "command": "huggingface-cli download AffectNet",
        "recommendation": "全球最大真實環境人臉資料集，光影與角度最豐富。"
    },
    {
        "name": "TFEID (Taiwanese Facial Expression Image Database)",
        "source": "國立陽明交通大學 / 國立台灣大學 腦與心智實驗室",
        "scale": "6,604+ 張台灣受試者高品質彩色照片",
        "classes": "喜 (Happy), 怒 (Angry), 哀 (Sad), 驚 (Surprise), 懼 (Fear), 厭 (Disgust), 輕蔑, 中性",
        "url": "https://bml.ym.edu.tw/tfeid/",
        "command": "前往官網免費學術授權申請 (tfeidnycu@gmail.com)",
        "recommendation": "台灣本土人臉表情資料庫，最契合台灣手語面部特徵研究！"
    },
    {
        "name": "CK+ (Extended Cohn-Kanade Dataset)",
        "source": "Carnegie Mellon University",
        "scale": "593 筆受控環境影片序列 (327 筆具備情緒標籤)",
        "classes": "7 種基本情緒 + FACS Action Units (AU) 細部標註",
        "url": "https://www.jeffcohn.net/Resources/",
        "command": "可直接在 Kaggle 下載 CK+ 整理版",
        "recommendation": "具備肌肉動作元 (AU) 標籤，適合驗證幾何特徵演算法。"
    }
]


SAMPLE_IMAGE_URLS = {
    "喜": [
        ("happy_1.jpg", "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=400&q=80"),
        ("happy_2.jpg", "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400&q=80")
    ],
    "怒": [
        ("angry_1.jpg", "https://images.unsplash.com/photo-1542909168-82c3e7fdca5c?w=400&q=80")
    ],
    "哀": [
        ("sad_1.jpg", "https://images.unsplash.com/photo-1499209974431-9dddcece7f88?w=400&q=80")
    ],
    "樂": [
        ("joy_1.jpg", "https://images.unsplash.com/photo-1492562080023-ab3db95bfbce?w=400&q=80")
    ],
    "平靜": [
        ("neutral_1.jpg", "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400&q=80")
    ]
}


def print_dataset_catalog():
    print("=" * 70)
    print("  網路開源臉部表情資料集目錄 (Facial Expression Recognition Datasets)")
    print("=" * 70)
    for idx, ds in enumerate(ONLINE_DATASETS, 1):
        print(f"\n[{idx}] {ds['name']}")
        print(f"    來源單位: {ds['source']}")
        print(f"    資料規模: {ds['scale']}")
        print(f"    標籤類別: {ds['classes']}")
        print(f"    官網網址: {ds['url']}")
        print(f"    下載指令: {ds['command']}")
        print(f"    特點推薦: {ds['recommendation']}")
    print("\n" + "=" * 70)
    print("  詳細完整介紹請參閱文件: docs/facial_expression_datasets.md")
    print("=" * 70, flush=True)


def download_sample_images(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"正在下載「喜、怒、哀、樂、平靜」測試樣本影像至: {output_dir} ...", flush=True)
    success_count = 0

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for emotion, items in SAMPLE_IMAGE_URLS.items():
        emotion_dir = os.path.join(output_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        for filename, url in items:
            dest_path = os.path.join(emotion_dir, filename)
            if os.path.exists(dest_path):
                print(f"  [已存在] {emotion}/{filename}")
                success_count += 1
                continue

            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp, open(dest_path, "wb") as f:
                    f.write(resp.read())
                print(f"  [下載成功] {emotion}/{filename}")
                success_count += 1
            except Exception as exc:
                print(f"  [跳過下載] {emotion}/{filename}: {exc}")

    print(f"\n下載程序完成，有效樣本數量：{success_count} 張。", flush=True)


def run_benchmark(sample_dir):
    try:
        from core.detectors.face_detector import FaceExpressionRecognizer
        import cv2
    except ImportError as e:
        print(f"無法載入辨識核心或 OpenCV: {e}")
        return

    if not os.path.exists(sample_dir):
        print(f"樣本目錄不存在：{sample_dir}，請先執行 --download-samples 下載測試樣本。")
        return

    print("=" * 60)
    print("  開始執行臉部表情辨識基準評測 (喜怒哀樂 FaceMesh 478 點特徵)")
    print("=" * 60)

    recognizer = FaceExpressionRecognizer(auto_init_mesh=False)

    total_images = 0
    correct_matches = 0

    for emotion in ("喜", "怒", "哀", "樂", "平靜"):
        folder = os.path.join(sample_dir, emotion)
        if not os.path.isdir(folder):
            continue

        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            total_images += 1
            res = recognizer.process(img)
            pred_emotion = res["emotion"] if res else "未偵測到人臉"
            is_correct = (pred_emotion == emotion)
            if is_correct:
                correct_matches += 1

            mark = "✅" if is_correct else "❌"
            print(f"  {mark} 標註: [{emotion}] ➔ 預測: [{pred_emotion}] ({fname})")

    recognizer.close()

    print("-" * 60)
    acc = (correct_matches / total_images * 100) if total_images > 0 else 0
    print(f"  評測結果: {correct_matches} / {total_images} 吻合 (準確率: {acc:.1f}%)")
    print("=" * 60, flush=True)


def main():
    parser = argparse.ArgumentParser(description="臉部表情資料集工具與喜怒哀樂評測")
    parser.add_argument("--info", action="store_true", help="列印國際與台灣本土表情資料集清單")
    parser.add_argument("--download-samples", action="store_true", help="下載精選五大表情測試影像")
    parser.add_argument("--benchmark", action="store_true", help="執行表情模型基準測試")
    parser.add_argument("--dir", default="sign_language_app/datasets/sample_expressions", help="樣本存放目錄")
    args = parser.parse_args()

    if not args.info and not args.download_samples and not args.benchmark:
        print_dataset_catalog()
        print("\n提示：可附加引數執行其他功能：")
        print("  python tools/download_expression_dataset.py --download-samples")
        print("  python tools/download_expression_dataset.py --benchmark")
        return

    if args.info:
        print_dataset_catalog()
    if args.download_samples:
        download_sample_images(args.dir)
    if args.benchmark:
        run_benchmark(args.dir)


if __name__ == "__main__":
    main()
