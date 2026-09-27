import json
import math
import os
from collections import Counter

MODEL_PATH = os.path.join("sign_language_app", "model.json")

with open(MODEL_PATH, "r", encoding="utf-8") as f:
    samples = json.load(f)["samples"]

print("總樣本:", len(samples))
print("向量長度:", len(samples[0]["vector"]))
print("標籤分佈:", dict(Counter(s["label"] for s in samples)))

# 同標籤內距離 vs 不同標籤間距離
v0 = samples[0]["vector"]
label0 = samples[0]["label"]

same_label_dists = []
diff_label_dists = []

for s in samples[1:]:
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(v0, s["vector"])))
    if s["label"] == label0:
        same_label_dists.append(dist)
    else:
        diff_label_dists.append(dist)

if same_label_dists:
    print(f"\n同標籤（{label0}）內距離:")
    print(f"  最小: {min(same_label_dists):.4f}")
    print(f"  最大: {max(same_label_dists):.4f}")
    print(f"  平均: {sum(same_label_dists)/len(same_label_dists):.4f}")

if diff_label_dists:
    print(f"\n不同標籤距離:")
    print(f"  最小: {min(diff_label_dists):.4f}")
    print(f"  最大: {max(diff_label_dists):.4f}")
    print(f"  平均: {sum(diff_label_dists)/len(diff_label_dists):.4f}")

# 看幾筆樣本前 5 個值
print("\n前 3 筆樣本的前 5 個值:")
for i in range(min(3, len(samples))):
    print(f"  樣本{i} ({samples[i]['label']}): {[round(x, 4) for x in samples[i]['vector'][:5]]}")