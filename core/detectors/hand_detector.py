from collections import Counter, deque
import json
import math
import os

import cv2
import mediapipe as mp
import numpy as np

try:
    from sign_language_app.labels import classification_label
    from sign_language_app.text_overlay import draw_panel, draw_text
except ImportError:
    classification_label = lambda label, include_english=True: label
    draw_panel = None
    draw_text = None


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class HandSignRecognizer:
    """MediaPipe hand landmarks plus conservative static hand-shape rules."""

    # Conservative thresholds make bent or occluded fingers become Unknown.
    THUMB_STRAIGHT_THRESHOLD = 0.78
    FINGER_STRAIGHT_THRESHOLD = 0.82
    FINGER_MIN_JOINT_ANGLE = 150.0
    THUMB_MIN_JOINT_ANGLE = 145.0
    FINGER_WRIST_EXTENSION_RATIO = 1.08
    THUMB_WRIST_EXTENSION_RATIO = 1.05
    OK_PINCH_THRESHOLD = 0.38
    STABLE_CONFIDENCE_THRESHOLD = 0.6
    DEFAULT_NO_HAND_RESET_FRAMES = 3

    FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    FINGER_MCPS = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
    LANDMARK_NAMES = [
        "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
    ]
    AVAILABLE_SIGNS = [
        "Fist (Solidarity)", "OK (Zero / Can)", "Number 1 (Secret)",
        "Number 2 (Victory)", "Number 3", "Number 4 (Salute)",
        "Number 5 (Hello / Greet)", "Number 6", "Number 7 (Gun)",
        "Number 8", "Number 9", "Good / Male (Thumbs up)",
        "Bad / Female (Pinky)", "I love you", "Cow / Horns", "Multiple hands",
        "Number 6 (Two hands)", "Number 7 (Two hands)", "Number 8 (Two hands)",
        "Number 9 (Two hands)", "Number 10 (Two hands)", "Unknown", "No hand",
    ]
    SIGN_ALIASES = {
        "1": "Number 1 (Secret)", "2": "Number 2 (Victory)", "3": "Number 3",
        "4": "Number 4 (Salute)", "5": "Number 5 (Hello / Greet)", "6": "Number 6",
        "7": "Number 7 (Gun)", "8": "Number 8", "9": "Number 9",
        "10": "Number 10 (Two hands)", "secret": "Number 1 (Secret)",
        "victory": "Number 2 (Victory)", "salute": "Number 4 (Salute)",
        "hello": "Number 5 (Hello / Greet)", "greet": "Number 5 (Hello / Greet)",
        "gun": "Number 7 (Gun)", "good": "Good / Male (Thumbs up)",
        "male": "Good / Male (Thumbs up)", "bad": "Bad / Female (Pinky)",
        "female": "Bad / Female (Pinky)", "thumbs up": "Good / Male (Thumbs up)",
        "pinky": "Bad / Female (Pinky)", "cow": "Cow / Horns", "horns": "Cow / Horns",
        "i love you": "I love you", "ok": "OK (Zero / Can)", "zero": "OK (Zero / Can)",
        "can": "OK (Zero / Can)", "fist": "Fist (Solidarity)",
    }

    def __init__(
        self, max_num_hands=2, history_size=8, stable_min_count=3,
        combine_two_hands=False, use_knn=True,
        no_hand_reset_frames=DEFAULT_NO_HAND_RESET_FRAMES,
    ):
        self.hands = mp_hands.Hands(
            static_image_mode=False, max_num_hands=max_num_hands,
            min_detection_confidence=0.5, min_tracking_confidence=0.4,
        )
        self.history = deque(maxlen=history_size)
        self.stable_min_count = stable_min_count
        self.combine_two_hands = combine_two_hands
        self.use_knn = use_knn
        self.no_hand_reset_frames = max(1, no_hand_reset_frames)
        self.current_candidate = "No hand"
        self.current_hand_count = 0
        self.no_hand_streak = 0
        self.sentence = []
        self.last_added_sign = None
        self.gesture_history = []
        self.gesture_start_time = None

        self.knn_samples = []
        self.knn_matrix = None
        self.knn_labels = None
        self.current_knn_confidence = 0.0
        if self.use_knn:
            model_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "sign_language_app", "model.json")
            )
            try:
                if os.path.exists(model_path):
                    with open(model_path, "r", encoding="utf-8") as handle:
                        self.knn_samples = json.load(handle).get("samples", [])
                    if self.knn_samples:
                        self.knn_matrix = np.array(
                            [sample["vector"] for sample in self.knn_samples],
                            dtype=np.float32,
                        )
                        self.knn_labels = np.array(
                            [sample["label"] for sample in self.knn_samples]
                        )
            except (OSError, ValueError) as exc:
                print(f"Warning: Could not load KNN model.json: {exc}")

    def process(self, frame):
        """Detect hands, classify each hand, and update stable state."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        detections = []

        if results.multi_hand_landmarks:
            handedness_list = results.multi_handedness or []
            for index, landmarks in enumerate(results.multi_hand_landmarks):
                handedness = "Unknown"
                if index < len(handedness_list):
                    handedness = handedness_list[index].classification[0].label
                fingers = self._extended_fingers(landmarks, handedness, frame.shape)
                sign = self._classify_sign(fingers, landmarks, frame.shape)
                detections.append({
                    "landmarks": landmarks, "handedness": handedness,
                    "fingers": fingers, "sign": sign,
                    "joint_points": self.joint_points(frame, landmarks),
                })

            candidate = self._frame_candidate(detections)
            knn_candidate = "Unknown"
            knn_confidence = 0.0

            if self.use_knn and self.knn_samples:
                num_hands = len(results.multi_hand_landmarks)
                vector = self._normalize_landmarks(results.multi_hand_landmarks)
                knn_candidate, knn_confidence = self._classify_knn(
                    vector,
                    k=9,
                    num_hands=num_hands,
                )
                if knn_candidate != "Unknown":
                    candidate = knn_candidate
                    for detection in detections:
                        detection["sign"] = knn_candidate

            self.current_knn_confidence = knn_confidence
        else:
            candidate = "No hand"
            self.current_knn_confidence = 0.0

        self._record_candidate(candidate, len(detections))
        return detections

    def _frame_candidate(self, detections):
        if len(detections) == 0:
            return "No hand"
        if len(detections) == 1:
            return detections[0]["sign"]
        if not self.combine_two_hands:
            return "Multiple hands"
        total_extended = sum(sum(item["fingers"].values()) for item in detections)
        return f"Number {total_extended} (Two hands)" if 6 <= total_extended <= 10 else "Unknown"

    def _record_candidate(self, candidate, hand_count):
        """Update history without allowing old votes to survive a real gap."""
        self.current_candidate = candidate
        self.current_hand_count = hand_count
        if candidate == "No hand":
            self.no_hand_streak += 1
            self.history.append(candidate)
            if self.no_hand_streak >= self.no_hand_reset_frames:
                self.history.clear()
                self.history.append("No hand")
                self.last_added_sign = None
        else:
            if self.no_hand_streak >= self.no_hand_reset_frames:
                self.history.clear()
            self.no_hand_streak = 0
            self.history.append(candidate)
        self._update_sentence()

    def draw(self, frame, detections, target_sign=None, face_expression=None):
        self._draw_hands(frame, detections)
        self._draw_hud(frame, target_sign, face_expression)
        self._draw_sentence(frame)

    def _draw_hands(self, frame, detections):
        for detection in detections:
            mp_drawing.draw_landmarks(frame, detection["landmarks"], mp_hands.HAND_CONNECTIONS)
            x, y = self._label_position(frame, detection["landmarks"])
            label = f'{detection["handedness"]}: {classification_label(detection["sign"])}'
            self._draw_text(frame, label, (x, y), 22, (255, 255, 0))

    def _draw_hud(self, frame, target_sign, face_expression):
        stable = self.stable_status
        stable_sign = stable["sign"]
        if stable_sign == "No hand" and face_expression is None:
            return
        rows = []
        if stable_sign != "No hand":
            status_text = "已確認" if stable["is_stable"] else "確認中"
            shape_text = (
                "多手畫面（各手分開顯示；合計模式未啟用）"
                if stable_sign == "Multiple hands"
                else classification_label(stable_sign)
            )
            field_name = "手語辨識" if self.use_knn and self.knn_samples else "手形分類"
            if self.use_knn and getattr(self, "current_knn_confidence", 0.0) > 0:
                metric_text = f"模型信心度: {self.current_knn_confidence:.0%}  {status_text}"
            else:
                metric_text = f"時間一致率: {stable['consistency']:.0%}  {status_text}"
            rows.extend([
                {"text": f"{field_name}: {shape_text}", "color": (0, 255, 255), "font_size": 24},
                {
                    "text": metric_text,
                    "color": (0, 220, 255) if stable["is_stable"] else (180, 180, 180),
                    "font_size": 20,
                },
            ])
        if face_expression:
            # 依情緒自動套用專屬色彩（怒:紅色、哀:藍色、喜/樂:金黃/青綠、平靜:淡黃）
            color = (0, 240, 255)
            if "怒" in face_expression:
                color = (60, 60, 255)
            elif "哀" in face_expression:
                color = (255, 180, 70)
            elif "樂" in face_expression:
                color = (0, 255, 180)
            elif "平靜" in face_expression:
                color = (210, 210, 210)
            rows.append({"text": f"表情: {face_expression}", "color": color, "font_size": 22})
        if target_sign:
            target_sign = self.normalize_sign(target_sign)
            detected = self.is_target_detected(target_sign)
            rows.append({
                "text": f"{'目標完成' if detected else '等待目標'}: {classification_label(target_sign)}",
                "color": (0, 255, 0) if detected else (180, 180, 180),
                "font_size": 22,
            })
        if draw_panel:
            draw_panel(frame, rows, width=520)
        else:
            cv2.rectangle(frame, (10, 10), (520, 10 + 20 + len(rows) * 30), (0, 0, 0), cv2.FILLED)
            for index, row in enumerate(rows):
                cv2.putText(
                    frame, row["text"].encode("ascii", errors="ignore").decode("ascii"),
                    (20, 40 + index * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, row["color"], 2,
                )

    def _draw_sentence(self, frame):
        if not self.sentence:
            return
        height, width, _ = frame.shape
        text = " -> ".join(classification_label(sign, include_english=False) for sign in self.sentence)
        max_chars = max(12, int(width / 12))
        if len(text) > max_chars:
            text = "..." + text[-max_chars:]
        if draw_panel:
            draw_panel(frame, [{"text": f"手勢紀錄: {text}", "color": (0, 255, 0), "font_size": 24}],
                       origin=(10, height - 58), width=width - 20, row_height=34)
        else:
            self._draw_text(frame, f"手勢紀錄: {text}", (20, height - 30), 24, (0, 255, 0))

    def close(self):
        self.hands.close()

    def _draw_text(self, frame, text, position, font_size, color):
        if draw_text:
            draw_text(frame, text, position, font_size, color)
        else:
            cv2.putText(frame, text.encode("ascii", errors="ignore").decode("ascii"), position,
                        cv2.FONT_HERSHEY_SIMPLEX, font_size / 32, color, 2)

    @property
    def stable_sign(self):
        return self.stable_status["sign"]

    @staticmethod
    def _trailing_count(history, candidate):
        count = 0
        for label in reversed(history):
            if label != candidate:
                break
            count += 1
        return count

    @property
    def stable_status(self):
        candidate = self.current_candidate
        total = len(self.history)
        if candidate == "No hand" or total == 0:
            count = sum(label == "No hand" for label in self.history)
            consistency = count / total if total else 0.0
            return {
                "sign": "No hand", "candidate": candidate, "count": count, "total": total,
                "consistency": consistency, "confidence": consistency, "is_stable": False,
                "hand_count": self.current_hand_count,
            }
        count = sum(label == candidate for label in self.history)
        consistency = count / total
        is_stable = (
            count >= self.stable_min_count
            and self._trailing_count(self.history, candidate) >= self.stable_min_count
            and consistency >= self.STABLE_CONFIDENCE_THRESHOLD
        )
        return {
            "sign": candidate, "candidate": candidate, "count": count, "total": total,
            "consistency": consistency, "confidence": consistency, "is_stable": is_stable,
            "hand_count": self.current_hand_count,
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
        return stable_number is not None and stable_number == target_number and "(Two hands)" in stable_sign

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
        if stable["sign"] == "No hand":
            return
        if stable["is_stable"] and stable["sign"] not in ("Unknown", "Multiple hands"):
            if stable["sign"] != self.last_added_sign:
                self.sentence.append(stable["sign"])
                self.last_added_sign = stable["sign"]

    def clear_sentence(self):
        self.sentence = []
        self.last_added_sign = None

    def _extended_fingers(self, landmarks, handedness, frame_shape=None):
        points = landmarks.landmark
        fingers = {}
        wrist = points[0]
        for finger in self.FINGER_TIPS:
            if finger == "thumb":
                fingers[finger] = self._is_thumb_extended(points, handedness, frame_shape)
                continue
            mcp_index = self.FINGER_MCPS[finger]
            mcp, pip, dip, tip = (points[mcp_index], points[mcp_index + 1],
                                   points[mcp_index + 2], points[mcp_index + 3])
            path = (self._distance(mcp, pip, frame_shape)
                    + self._distance(pip, dip, frame_shape)
                    + self._distance(dip, tip, frame_shape))
            chord = self._distance(mcp, tip, frame_shape)
            straight = (
                chord / max(path, 1e-6) >= self.FINGER_STRAIGHT_THRESHOLD
                and min(self._angle(mcp, pip, dip, frame_shape),
                        self._angle(pip, dip, tip, frame_shape)) >= self.FINGER_MIN_JOINT_ANGLE
            )
            wrist_tip = self._distance(wrist, tip, frame_shape)
            wrist_pip = self._distance(wrist, pip, frame_shape)
            fingers[finger] = straight and wrist_tip >= wrist_pip * self.FINGER_WRIST_EXTENSION_RATIO
        return fingers

    def _is_thumb_extended(self, points, handedness, frame_shape=None):
        cmc, mcp, ip, tip = points[1], points[2], points[3], points[4]
        wrist = points[0]
        index_mcp = points[self.FINGER_MCPS["index"]]
        middle_mcp = points[self.FINGER_MCPS["middle"]]
        path = (self._distance(cmc, mcp, frame_shape)
                + self._distance(mcp, ip, frame_shape)
                + self._distance(ip, tip, frame_shape))
        chord = self._distance(cmc, tip, frame_shape)
        straight = (
            chord / max(path, 1e-6) >= self.THUMB_STRAIGHT_THRESHOLD
            and self._angle(cmc, mcp, ip, frame_shape) >= self.THUMB_MIN_JOINT_ANGLE
            and self._angle(mcp, ip, tip, frame_shape) >= self.THUMB_MIN_JOINT_ANGLE
        )
        palm_size = max(self._distance(wrist, middle_mcp, frame_shape), 1e-6)
        outside_palm = self._distance(tip, index_mcp, frame_shape) / palm_size >= 0.45
        wrist_extension = self._distance(wrist, tip, frame_shape) >= (
            self._distance(wrist, ip, frame_shape) * self.THUMB_WRIST_EXTENSION_RATIO
        )
        return straight and outside_palm and wrist_extension

    def _classify_sign(self, fingers, landmarks, frame_shape=None):
        thumb, index, middle = (bool(fingers["thumb"]), bool(fingers["index"]), bool(fingers["middle"]))
        ring, pinky = bool(fingers["ring"]), bool(fingers["pinky"])

        # OK requires the other three fingers to be extended; pinch alone is not OK.
        if self._is_ok_sign(fingers, landmarks, frame_shape):
            return "OK (Zero / Can)"
        if not any(fingers.values()):
            return "Fist (Solidarity)"
        if thumb and not any((index, middle, ring, pinky)):
            return "Good / Male (Thumbs up)"
        if pinky and not any((thumb, index, middle, ring)):
            return "Bad / Female (Pinky)"
        if index and pinky and not any((thumb, middle, ring)):
            return "Cow / Horns"
        if thumb and index and pinky and not middle and not ring:
            return "I love you"

        # Project definition: thumb+index is 7; adding middle makes 8.
        if thumb:
            if index and middle and ring and pinky:
                return "Number 5 (Hello / Greet)"
            if index and middle and ring and not pinky:
                return "Number 9"
            if index and middle and not ring and not pinky:
                return "Number 8"
            if index and not middle and not ring and not pinky:
                return "Number 7 (Gun)"
            if pinky and not index and not middle and not ring:
                return "Number 6"
        else:
            if index and middle and ring and pinky:
                return "Number 4 (Salute)"
            if index and middle and ring and not pinky:
                return "Number 3"
            if index and middle and not ring and not pinky:
                return "Number 2 (Victory)"
            if index and not middle and not ring and not pinky:
                return "Number 1 (Secret)"
        return "Unknown"

    def _is_ok_sign(self, fingers, landmarks, frame_shape=None):
        if not (fingers["middle"] and fingers["ring"] and fingers["pinky"]):
            return False
        if fingers["index"]:
            return False
        points = landmarks.landmark
        pinch_distance = self._distance(points[4], points[8], frame_shape)
        palm_size = max(self._distance(points[0], points[9], frame_shape), 1e-6)
        return pinch_distance / palm_size < self.OK_PINCH_THRESHOLD

    def _normalize_landmarks(self, multi_hand_landmarks):
        """Keep the existing KNN vector format for model compatibility."""
        if not multi_hand_landmarks:
            return [0.0] * 126
        hand1 = multi_hand_landmarks[0].landmark
        wrist, middle_mcp = hand1[0], hand1[9]
        scale = math.sqrt((middle_mcp.x - wrist.x) ** 2 + (middle_mcp.y - wrist.y) ** 2
                          + (middle_mcp.z - wrist.z) ** 2) or 1e-6
        normalized = []
        for landmark in hand1:
            normalized.extend([(landmark.x - wrist.x) / scale, (landmark.y - wrist.y) / scale,
                                (landmark.z - wrist.z) / scale])
        if len(multi_hand_landmarks) > 1:
            for landmark in multi_hand_landmarks[1].landmark:
                normalized.extend([(landmark.x - wrist.x) / scale, (landmark.y - wrist.y) / scale,
                                    (landmark.z - wrist.z) / scale])
        normalized.extend([0.0] * (126 - len(normalized)))
        return normalized[:126]

    def _classify_knn(self, vector, k=9, distance_threshold=None, num_hands=1):
        if not self.knn_samples:
            return "Unknown", 0.0

        if self.knn_matrix is not None and self.knn_labels is not None:
            query = np.asarray(vector, dtype=np.float32)
            diffs = self.knn_matrix - query
            sq_dists = np.sum(diffs * diffs, axis=1)

            k_val = min(k, len(self.knn_samples))
            top_k_indices = np.argpartition(sq_dists, k_val)[:k_val]
            top_k_sorted = top_k_indices[np.argsort(sq_dists[top_k_indices])]
            top_dists = np.sqrt(sq_dists[top_k_sorted])
            top_labels = self.knn_labels[top_k_sorted]

            min_dist = float(top_dists[0])
            max_dist = distance_threshold if distance_threshold is not None else (16.0 if num_hands == 1 else 45.0)
            if min_dist > max_dist:
                return "Unknown", 0.0

            weights = 1.0 / (top_dists + 1e-4)
            votes = Counter()
            for w, l in zip(weights, top_labels):
                votes[l] += float(w)
            total_weight = sum(votes.values())
            if total_weight <= 0:
                return "Unknown", 0.0

            best_label, best_weight = max(votes.items(), key=lambda item: item[1])
            confidence = best_weight / total_weight
            if confidence < 0.20:
                return "Unknown", 0.0
            return best_label, confidence

        # Fallback pure Python if numpy matrix not initialized
        neighbors = []
        for sample in self.knn_samples:
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vector, sample["vector"])))
            neighbors.append((distance, sample["label"]))
        neighbors.sort(key=lambda item: item[0])
        top_k = neighbors[:min(k, len(neighbors))]
        max_dist = distance_threshold if distance_threshold is not None else (16.0 if num_hands == 1 else 45.0)
        if not top_k or top_k[0][0] > max_dist:
            return "Unknown", 0.0
        votes = Counter()
        for distance, label in top_k:
            votes[label] += 1.0 / max(distance, 1e-6)
        total_weight = sum(votes.values())
        if total_weight <= 0:
            return "Unknown", 0.0
        best_label, best_weight = max(votes.items(), key=lambda item: item[1])
        confidence = best_weight / total_weight
        if confidence < 0.20:
            return "Unknown", 0.0
        return best_label, confidence

    @staticmethod
    def _scaled_point(point, frame_shape=None):
        if frame_shape is None:
            return point.x, point.y, point.z
        height, width = float(frame_shape[0]), float(frame_shape[1])
        # MediaPipe x/z use image-width scale; y uses image-height scale.
        return point.x * width, point.y * height, point.z * width

    def _distance(self, point_a, point_b, frame_shape=None):
        a, b = self._scaled_point(point_a, frame_shape), self._scaled_point(point_b, frame_shape)
        return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))

    def _angle(self, point_a, point_b, point_c, frame_shape=None):
        a, b, c = (self._scaled_point(point, frame_shape) for point in (point_a, point_b, point_c))
        ba = [left - right for left, right in zip(a, b)]
        bc = [left - right for left, right in zip(c, b)]
        norm_ba = math.sqrt(sum(value * value for value in ba))
        norm_bc = math.sqrt(sum(value * value for value in bc))
        if norm_ba == 0 or norm_bc == 0:
            return 0.0
        cosine = sum(left * right for left, right in zip(ba, bc)) / (norm_ba * norm_bc)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    def _label_position(self, frame, landmarks):
        height, width, _ = frame.shape
        x = max(10, int(min(point.x for point in landmarks.landmark) * width))
        y = max(30, int(min(point.y for point in landmarks.landmark) * height) - 10)
        return x, y

    def joint_points(self, frame, landmarks):
        height, width, _ = frame.shape
        return {
            name: {"x": int(point.x * width), "y": int(point.y * height), "z": round(point.z, 4)}
            for name, point in zip(self.LANDMARK_NAMES, landmarks.landmark)
        }
