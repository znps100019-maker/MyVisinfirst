import json
import sys
from collections import Counter

def analyze_output(filename="output.json"):
    """分析手勢辨識輸出"""
    signs = []
    face_expressions = []
    hand_count = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'):
                continue
            try:
                data = json.loads(line)
                if 'stable_sign' in data:
                    signs.append(data['stable_sign'])
                if 'face_expression' in data and data['face_expression']:
                    face_expressions.append(data['face_expression'])
                if 'hands' in data:
                    hand_count.append(len(data['hands']))
            except json.JSONDecodeError:
                continue
    
    print("=" * 50)
    print("手勢辨識結果分析")
    print("=" * 50)
    
    if signs:
        print(f"\n最常見的手勢:")
        sign_counts = Counter(signs)
        for sign, count in sign_counts.most_common(5):
            print(f"  - {sign}: {count} 次")
    
    if face_expressions:
        print(f"\n臉部表情分布:")
        expr_counts = Counter(face_expressions)
        for expr, count in expr_counts.most_common():
            print(f"  - {expr}: {count} 次")
    
    if hand_count:
        print(f"\n手部偵測統計:")
        print(f"  - 平均偵測到手數: {sum(hand_count)/len(hand_count):.1f}")
        print(f"  - 最多偵測到手數: {max(hand_count)}")
        print(f"  - 沒有手的幀數: {hand_count.count(0)}")
    
    print(f"\n總共分析幀數: {len(signs)}")

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "output.json"
    analyze_output(filename)
