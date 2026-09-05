import os
import sys
import json
import numpy as np

def main():
    import argparse
    parser = argparse.ArgumentParser(description="編譯手語關節點特徵資料，生成 KNN 樣板模型。")
    parser.add_argument("--dataset", default="dataset.json", help="輸入的特徵資料庫 JSON 路徑。")
    parser.add_argument("--model", default="model.json", help="輸出的 KNN 模型 JSON 路徑。")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, args.dataset)
    model_path = os.path.join(script_dir, args.model)

    if not os.path.exists(dataset_path):
        print(f"錯誤：找不到特徵資料庫檔案 '{dataset_path}'。請先執行 extract_dataset.py。")
        sys.exit(1)

    print(f"正在讀取特徵資料庫: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 準備儲存成 KNN 模型格式
    model_data = {
        "samples": []
    }

    print("\n> 正在編譯手勢類別與計算特徵樣板...")
    total_samples = 0
    
    for category, vectors in dataset.items():
        if not vectors:
            print(f"  分類 '{category}': 沒有有效的特徵樣本，略過。")
            continue
            
        print(f"  分類 '{category}': 導入 {len(vectors)} 個影格特徵樣本。")
        for vec in vectors:
            # 確保向量維度是 126 (雙手 42點 * 3)
            if len(vec) == 126:
                model_data["samples"].append({
                    "label": category,
                    "vector": vec
                })
                total_samples += 1

    if total_samples == 0:
        print("錯誤：沒有任何有效的特徵樣本可以編譯！")
        sys.exit(1)

    # 寫入 model.json
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] KNN 模型樣板庫編譯完成！")
    print(f"  儲存路徑：{model_path}")
    print(f"  總特徵樣本數：{total_samples} 筆")
    print(f"  您現在可以執行 recognizer.py 開始實時辨識！")

if __name__ == "__main__":
    main()
