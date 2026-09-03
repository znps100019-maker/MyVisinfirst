from collections import Counter, deque
import math

import cv2
import mediapipe as mp

try:
    from sign_language_app.labels import display_label
    from sign_language_app.text_overlay import draw_panel, draw_text
except ImportError:
    display_label = lambda label, include_english=True: label
    draw_panel = None
    draw_text = None


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class HandSignRecognizer:
    """Detect hand landmarks and classify simple static hand signs."""

    THUMB_STRAIGHT_THRESHOLD = 0.65
    FINGER_STRAIGHT_THRESHOLD = 0.55
    FINGER_WRIST_EXTENSION_RATIO = 0.90
    THUMB_SPREAD_THRESHOLD = 0.58
    OK_PINCH_THRESHOLD = 0.38
    STABLE_CONFIDENCE_THRESHOLD = 0.6

    FINGER_TIPS = {
        "thumb": 4,
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20,
    }
    FINGER_PIPS = {
        "index": 6,
        "middle": 10,
        "ring": 14,
        "pinky": 18,
    }
    FINGER_MCPS = {
        "thumb": 2,
        "index": 5,
        "middle": 9,
        "ring": 13,
        "pinky": 17,
    }
    LANDMARK_NAMES = [
        "wrist",
        "thumb_cmc",
        "thumb_mcp",
        "thumb_ip",
        "thumb_tip",
        "index_mcp",
        "index_pip",
        "index_dip",
        "index_tip",
        "middle_mcp",
        "middle_pip",
        "middle_dip",
        "middle_tip",
        "ring_mcp",
        "ring_pip",
        "ring_dip",
        "ring_tip",
        "pinky_mcp",
        "pinky_pip",
        "pinky_dip",
        "pinky_tip",
    ]
    AVAILABLE_SIGNS = [
        "Fist (Solidarity)",
        "OK (Zero / Can)",
        "Number 1 (Secret)",
        "Number 2 (Victory)",
        "Number 3",
        "Number 4 (Salute)",
        "Number 5 (Hello / Greet)",
        "Number 6",
        "Number 7 (Gun)",
        "Number 8",
        "Number 9",
        "Good / Male (Thumbs up)",
        "Bad / Female (Pinky)",
        "I love you",
        "Cow / Horns",
        "Number 6 (Two hands)",
        "Number 7 (Two hands)",
        "Number 8 (Two hands)",
        "Number 9 (Two hands)",
        "Number 10 (Two hands)",
        "Unknown",
        "No hand",
    ]
    SIGN_ALIASES = {
        "1": "Number 1 (Secret)",
        "2": "Number 2 (Victory)",
        "3": "Number 3",
        "4": "Number 4 (Salute)",
        "5": "Number 5 (Hello / Greet)",
        "6": "Number 6",
        "7": "Number 7 (Gun)",
        "8": "Number 8",
        "9": "Number 9",
        "10": "Number 10 (Two hands)",
        "secret": "Number 1 (Secret)",
        "victory": "Number 2 (Victory)",
        "salute": "Number 4 (Salute)",
        "hello": "Number 5 (Hello / Greet)",
        "greet": "Number 5 (Hello / Greet)",
        "gun": "Number 7 (Gun)",
        "good": "Good / Male (Thumbs up)",
        "male": "Good / Male (Thumbs up)",
        "bad": "Bad / Female (Pinky)",
        "female": "Bad / Female (Pinky)",
        "thumbs up": "Good / Male (Thumbs up)",
        "pinky": "Bad / Female (Pinky)",
        "cow": "Cow / Horns",
        "horns": "Cow / Horns",
        "i love you": "I love you",
        "ok": "OK (Zero / Can)",
        "zero": "OK (Zero / Can)",
        "can": "OK (Zero / Can)",
        "fist": "Fist (Solidarity)",
    }

    def __init__(self, max_num_hands=2, history_size=8, stable_min_count=3):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        self.history = deque(maxlen=history_size)
        self.stable_min_count = stable_min_count
        self.sentence = []
        self.last_added_sign = None
        self.gesture_history = []  # 完整的手勢歷史
        self.gesture_start_time = None  # 目前手勢開始時間

    def process(self, frame):
        """Return detected hands with landmarks, handedness, and sign labels."""
        import time
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        current_time = time.time()

        detections = []
        if not results.multi_hand_landmarks:
            self.history.append("No hand")
            return detections

        handedness_list = results.multi_handedness or []

        for index, landmarks in enumerate(results.multi_hand_landmarks):
            handedness = "Unknown"
            if index < len(handedness_list):
                handedness = handedness_list[index].classification[0].label

            fingers = self._extended_fingers(landmarks, handedness)
            sign = self._classify_sign(fingers, landmarks)
            detections.append(
                {
                    "landmarks": landmarks,
                    "handedness": handedness,
                    "fingers": fingers,
                    "sign": sign,
                    "joint_points": self.joint_points(frame, landmarks),
                }
            )

        if len(detections) >= 2:
            total_extended = sum(detections[0]["fingers"].values()) + sum(
                detections[1]["fingers"].values()
            )
            if 6 <= total_extended <= 10:
                self.history.append(f"Number {total_extended} (Two hands)")
            else:
                self.history.append(f"Two hands ({total_extended} fingers)")
        elif len(detections) == 1:
            self.history.append(detections[0]["sign"])

        self._update_sentence()
        return detections

    def draw(self, frame, detections, target_sign=None, face_expression=None):
        """Draw hand skeletons, sign labels, and face expression info on a video frame."""
        self._draw_hands(frame, detections)
        self._draw_hud(frame, target_sign, face_expression)
        self._draw_sentence(frame)

    def _draw_hands(self, frame, detections):
        for detection in detections:
            mp_drawing.draw_landmarks(
                frame,
                detection["landmarks"],
                mp_hands.HAND_CONNECTIONS,
            )
            x, y = self._label_position(frame, detection["landmarks"])
            label = f'{detection["handedness"]}: {display_label(detection["sign"])}'
            self._draw_text(frame, label, (x, y), 22, (255, 255, 0))

    def _draw_hud(self, frame, target_sign, face_expression):
        stable = self.stable_status
        stable_sign = stable["sign"]
        show_box = stable_sign != "No hand" or face_expression is not None

        if not show_box:
            return

        rows = []
        if stable_sign != "No hand":
            rows.append(
                {
                    "text": f"手語: {display_label(stable_sign)}  {stable['confidence']:.0%}",
                    "color": (0, 255, 255),
                    "font_size": 24,
                }
            )
        if face_expression:
            rows.append(
                {
                    "text": f"表情: {face_expression}",
                    "color": (255, 200, 0),
                    "font_size": 22,
                }
            )
        if target_sign:
            target_sign = self.normalize_sign(target_sign)
            detected = self.is_target_detected(target_sign)
            color = (0, 255, 0) if detected else (180, 180, 180)
            text = "目標完成" if detected else "等待目標"
            rows.append(
                {
                    "text": f"{text}: {display_label(target_sign)}",
                    "color": color,
                    "font_size": 22,
                }
            )

        if draw_panel:
            draw_panel(frame, rows, width=460)
        else:
            box_height = 20 + len(rows) * 30
            cv2.rectangle(frame, (10, 10), (460, 10 + box_height), (0, 0, 0), cv2.FILLED)
            for i, row in enumerate(rows):
                cv2.putText(
                    frame,
                    row["text"].encode("ascii", errors="ignore").decode("ascii"),
                    (20, 40 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    row["color"],
                    2,
                )

    def _draw_sentence(self, frame):
        if not self.sentence:
            return

        height, width, _ = frame.shape
        cv2.rectangle(frame, (10, height - 55), (width - 10, height - 15), (0, 0, 0), cv2.FILLED)

        sentence_text = " -> ".join(display_label(sign, include_english=False) for sign in self.sentence)
        max_char_len = int(width / 12)
        if len(sentence_text) > max_char_len:
            sentence_text = "..." + sentence_text[-max_char_len:]

        self._draw_text(frame, f"句子: {sentence_text}", (20, height - 46), 24, (0, 255, 0))

    def close(self):
        self.hands.close()

    def _draw_text(self, frame, text, position, font_size, color):
        if draw_text:
            draw_text(frame, text, position, font_size, color)
        else:
            cv2.putText(
                frame,
                text.encode("ascii", errors="ignore").decode("ascii"),
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size / 32,
                color,
                2,
            )

    @property
    def stable_sign(self):
        return self.stable_status["sign"]

    @property
    def stable_status(self):
        if not self.history:
            return {
                "sign": "No hand",
                "count": 0,
                "total": 0,
                "confidence": 0.0,
                "is_stable": False,
            }

        sign, count = Counter(self.history).most_common(1)[0]
        total = len(self.history)
        confidence = count / total
        return {
            "sign": sign,
            "count": count,
            "total": total,
            "confidence": confidence,
            "is_stable": count >= self.stable_min_count
            and confidence >= self.STABLE_CONFIDENCE_THRESHOLD,
        }

    @classmethod
    def normalize_sign(cls, sign):
        if not sign:
            return ""

        cleaned = str(sign).strip()
        lowered = cleaned.lower()
        if lowered in cls.SIGN_ALIASES:
            return cls.SIGN_ALIASES[lowered]

        for available in cls.AVAILABLE_SIGNS:
            if lowered == available.lower():
                return available

        return cleaned

    @classmethod
    def target_matches(cls, stable_sign, target_sign, is_stable=True):
        if not target_sign or not is_stable:
            return False

        stable_sign = cls.normalize_sign(stable_sign)
        target_sign = cls.normalize_sign(target_sign)
        if stable_sign == target_sign:
            return True

        stable_number = cls._sign_number(stable_sign)
        target_number = cls._sign_number(target_sign)
        return (
            stable_number is not None
            and stable_number == target_number
            and "(Two hands)" in stable_sign
        )

    @staticmethod
    def _sign_number(sign):
        if not sign.startswith("Number "):
            return None
        value = sign.removeprefix("Number ").split(" ", 1)[0]
        return int(value) if value.isdigit() else None

    def is_target_detected(self, target_sign):
        stable = self.stable_status
        return self.target_matches(stable["sign"], target_sign, stable["is_stable"])

    def _update_sentence(self):
        stable = self.stable_status
        if stable["is_stable"]:
            sign = stable["sign"]
            if sign == "No hand":
                self.last_added_sign = None
            elif sign != "Unknown" and sign != self.last_added_sign:
                self.sentence.append(sign)
                self.last_added_sign = sign

    def clear_sentence(self):
        self.sentence = []
        self.last_added_sign = None

    def _extended_fingers(self, landmarks, handedness):
        points = landmarks.landmark
        fingers = {}
        wrist = points[0]

        for finger in self.FINGER_TIPS:
            if finger == "thumb":
                fingers[finger] = self._is_thumb_extended(points, handedness)
                continue

            base_idx = self.FINGER_MCPS[finger]
            mcp = points[base_idx]
            pip = points[base_idx + 1]
            dip = points[base_idx + 2]
            tip = points[base_idx + 3]

            straight = self._distance(mcp, tip)
            segments = (
                self._distance(mcp, pip)
                + self._distance(pip, dip)
                + self._distance(dip, tip)
            )
            is_straight = straight / max(segments, 0.001) > self.FINGER_STRAIGHT_THRESHOLD

            d_wrist_tip = self._distance(wrist, tip)
            d_wrist_pip = self._distance(wrist, pip)
            is_extended_from_wrist = (
                d_wrist_tip > d_wrist_pip * self.FINGER_WRIST_EXTENSION_RATIO
            )

            fingers[finger] = is_straight and is_extended_from_wrist

        return fingers

    def _is_thumb_extended(self, points, handedness):
        cmc = points[1]
        mcp = points[2]
        ip = points[3]
        tip = points[4]
        wrist = points[0]
        index_mcp = points[self.FINGER_MCPS["index"]]
        middle_mcp = points[self.FINGER_MCPS["middle"]]

        straight = self._distance(cmc, tip)
        segments = (
            self._distance(cmc, mcp)
            + self._distance(mcp, ip)
            + self._distance(ip, tip)
        )
        straight_ratio = straight / max(segments, 0.001)
        palm_size = max(self._distance(wrist, middle_mcp), 0.001)
        spread_from_palm = self._distance(tip, index_mcp) / palm_size
        lifted_from_wrist = self._distance(wrist, tip) > self._distance(wrist, ip) * 1.02

        return (
            straight_ratio > self.THUMB_STRAIGHT_THRESHOLD
            and (spread_from_palm > 0.35 or lifted_from_wrist)
        )

    def _thumb_spread_ratio(self, landmarks):
        points = landmarks.landmark
        thumb_tip = points[self.FINGER_TIPS["thumb"]]
        index_mcp = points[self.FINGER_MCPS["index"]]
        wrist = points[0]
        middle_mcp = points[self.FINGER_MCPS["middle"]]
        index_tip = points[self.FINGER_TIPS["index"]]

        # 計算拇指與食指的夾角
        distance = self._distance(thumb_tip, index_mcp)
        palm_size = max(self._distance(wrist, middle_mcp), 0.001)
        spread_ratio = distance / palm_size
        
        # 計算拇指與食指尖端的距離（輔助判斷）
        thumb_index_distance = self._distance(thumb_tip, index_tip)
        thumb_index_ratio = thumb_index_distance / palm_size
        
        # 如果拇指和食指距離很遠，表示拇指展開
        return max(spread_ratio, thumb_index_ratio * 0.8)

    def _classify_sign(self, fingers, landmarks):
        thumb = fingers["thumb"]
        index = fingers["index"]
        middle = fingers["middle"]
        ring = fingers["ring"]
        pinky = fingers["pinky"]

        if self._is_ok_sign(landmarks):
            return "OK (Zero / Can)"

        if thumb and index and pinky and not middle and not ring:
            return "I love you"
        if pinky and not any([thumb, index, middle, ring]):
            return "Bad / Female (Pinky)"
        if index and pinky and not thumb and not middle and not ring:
            return "Cow / Horns"

        if not any(fingers.values()):
            return "Fist (Solidarity)"

        # 計算伸直的手指數量
        extended_count = sum([index, middle, ring, pinky])
        
        # 特殊手勢判斷
        if thumb and not any([index, middle, ring, pinky]):
            return "Good / Male (Thumbs up)"
        if pinky and not any([thumb, index, middle, ring]):
            return "Bad / Female (Pinky)"
        if index and pinky and not thumb and not middle and not ring:
            return "Cow / Horns"
        if thumb and index and pinky and not middle and not ring:
            return "I love you"
        
        # 檢查拇指是否真的展開（使用多個特徵）
        thumb_spread_ratio = self._thumb_spread_ratio(landmarks)
        is_thumb_spread = thumb_spread_ratio > self.THUMB_SPREAD_THRESHOLD
        
        # 數字手勢判斷
        if is_thumb_spread:
            # 拇指展開的情況
            if extended_count == 4 and index and middle and ring and pinky:
                return "Number 5 (Hello / Greet)"
            elif extended_count == 3 and index and middle and ring:
                return "Number 9"
            elif extended_count == 2:
                if index and middle:
                    return "Number 8"
                elif index and pinky:
                    return "Number 7 (Gun)"
            elif extended_count == 1 and pinky:
                return "Number 6"
        else:
            # 拇指未展開的情況
            if extended_count == 4 and index and middle and ring and pinky:
                return "Number 4 (Salute)"
            elif extended_count == 3 and index and middle and ring:
                return "Number 3"
            elif extended_count == 2 and index and middle:
                return "Number 2 (Victory)"
            elif extended_count == 1 and index:
                return "Number 1 (Secret)"
        
        return "Unknown"

    def _is_ok_sign(self, landmarks):
        points = landmarks.landmark
        thumb_tip = points[self.FINGER_TIPS["thumb"]]
        index_tip = points[self.FINGER_TIPS["index"]]
        wrist = points[0]
        middle_mcp = points[self.FINGER_MCPS["middle"]]

        pinch_distance = self._distance(thumb_tip, index_tip)
        palm_size = max(self._distance(wrist, middle_mcp), 0.001)
        return pinch_distance / palm_size < self.OK_PINCH_THRESHOLD

    def _distance(self, point_a, point_b):
        return math.sqrt(
            (point_a.x - point_b.x) ** 2
            + (point_a.y - point_b.y) ** 2
            + (point_a.z - point_b.z) ** 2
        )

    def _label_position(self, frame, landmarks):
        height, width, _ = frame.shape
        x_values = [point.x for point in landmarks.landmark]
        y_values = [point.y for point in landmarks.landmark]

        x = max(10, int(min(x_values) * width))
        y = max(30, int(min(y_values) * height) - 10)
        return x, y

    def joint_points(self, frame, landmarks):
        height, width, _ = frame.shape
        points = {}

        for index, point in enumerate(landmarks.landmark):
            name = self.LANDMARK_NAMES[index]
            points[name] = {
                "x": int(point.x * width),
                "y": int(point.y * height),
                "z": round(point.z, 4),
            }

        return points
