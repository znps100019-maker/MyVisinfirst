import json
from functools import lru_cache
from pathlib import Path


BUILT_IN_LABELS = {
    "Fist (Solidarity)": ("握拳", "Fist"),
    "OK (Zero / Can)": ("OK 手形", "OK"),
    "Number 1 (Secret)": ("數字 1 / 食指", "Number 1"),
    "Number 2 (Victory)": ("數字 2 / 食指+中指", "Number 2"),
    "Number 3": ("數字 3 / 三指", "Number 3"),
    "Number 4 (Salute)": ("數字 4 / 四指", "Number 4"),
    "Number 5 (Hello / Greet)": ("數字 5 / 五指", "Number 5"),
    "Number 6": ("數字 6 / 拇指+小指", "Number 6"),
    "Number 7 (Gun)": ("數字 7 / 拇指+食指", "Number 7"),
    "Number 8": ("數字 8 / 拇指+食指+中指", "Number 8"),
    "Number 9": ("數字 9 / 四指組合", "Number 9"),
    "Number 6 (Two hands)": ("雙手合計數字 6", "Number 6"),
    "Number 7 (Two hands)": ("雙手合計數字 7", "Number 7"),
    "Number 8 (Two hands)": ("雙手合計數字 8", "Number 8"),
    "Number 9 (Two hands)": ("雙手合計數字 9", "Number 9"),
    "Number 10 (Two hands)": ("雙手合計數字 10", "Number 10"),
    "Good / Male (Thumbs up)": ("拇指伸直", "Thumb extended"),
    "Bad / Female (Pinky)": ("小指伸直", "Pinky extended"),
    "I love you": ("拇指+食指+小指", "Thumb+index+pinky"),
    "Cow / Horns": ("食指+小指", "Index+pinky"),
    "Multiple hands": ("多手畫面", "Multiple hands"),
    "Unknown": ("無法判斷", "Unknown"),
    "No hand": ("沒有手", "No hand"),
}


@lru_cache(maxsize=1)
def load_signs():
    path = Path(__file__).with_name("communication_signs.json")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def label_map():
    labels = {label: {"display": display, "english": english} for label, (display, english) in BUILT_IN_LABELS.items()}
    for item in load_signs():
        labels[item["label"]] = {
            "display": item["display"],
            "english": item.get("english", item["label"]),
        }
    return labels


def display_label(label, include_english=True):
    info = label_map().get(label)
    if not info:
        return label

    if include_english and info["english"] and info["english"] != info["display"]:
        return f'{info["display"]} / {info["english"]}'
    return info["display"]


def classification_label(label, include_english=True):
    """Return a static shape label without rewriting custom database labels."""
    if label in BUILT_IN_LABELS:
        return display_label(label, include_english=include_english)
    if label in label_map():
        suffix = f" / {label_map()[label]['english']}" if include_english else ""
        return f"資料庫標籤: {label}{suffix}"
    return label


def label_keywords():
    keywords = {}
    for item in load_signs():
        values = set(item.get("keywords", []))
        values.add(item["label"].replace("_", " "))
        values.add(item["display"])
        if item.get("english"):
            values.add(item["english"])
        keywords[item["label"]] = sorted(values, key=len, reverse=True)
    return keywords
