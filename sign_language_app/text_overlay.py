from pathlib import Path

import cv2
import numpy as np


try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


FONT_CANDIDATES = [
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msjhbd.ttc",
    "C:/Windows/Fonts/mingliu.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def draw_text(frame, text, position, font_size=24, color=(255, 255, 255), thickness=2):
    """Draw Unicode text on an OpenCV BGR frame, with an OpenCV fallback."""
    if Image is None:
        cv2.putText(frame, _ascii_fallback(text), position, cv2.FONT_HERSHEY_SIMPLEX, font_size / 32, color, thickness)
        return frame

    font = _load_font(font_size)
    if font is None:
        cv2.putText(frame, _ascii_fallback(text), position, cv2.FONT_HERSHEY_SIMPLEX, font_size / 32, color, thickness)
        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    frame[:] = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return frame


def draw_panel(frame, rows, origin=(10, 10), width=420, row_height=32, padding=12):
    """Draw a compact readable HUD panel."""
    if not rows:
        return frame

    x, y = origin
    height = padding * 2 + row_height * len(rows)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (18, 18, 18), cv2.FILLED)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (70, 70, 70), 1)

    for index, row in enumerate(rows):
        text = row.get("text", "")
        color = row.get("color", (255, 255, 255))
        font_size = row.get("font_size", 22)
        draw_text(frame, text, (x + padding, y + padding + index * row_height), font_size, color)

    return frame


def _load_font(font_size):
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, font_size)
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _ascii_fallback(text):
    return text.encode("ascii", errors="ignore").decode("ascii") or "..."
