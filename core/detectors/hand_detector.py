from collections import Counter, deque
import json
import math
import os
import statistics

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
    """完全基於 KNN 的動態手語辨識器（無規則式）"""

    STABLE_CONFIDENCE_THRESHOLD = 0.6
    KNN_K = 5
    KNN_MIN_CONFIDENCE = 0.5

    FACE_KEY_INDICES = [
        33, 133, 159, 145, 362, 263, 386, 374,
        1, 2, 98, 327, 61, 291, 13, 14, 78, 308, 82, 312,
    ]

    def __init__(self, max_num_hands=2, history_size=8, stable_min_count=5,
                 combine_two_hands=True, use_knn=True):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.history = deque(maxlen=history_size)
        self.stable_min_count = stable_min_count
        self.sentence = []
        self.last_added_sign = None
        self.combine_two_hands = combine_two_hands
        self.use_knn = use_knn
        self.current_knn_confidence = 0.0
        self.knn_window = deque(maxlen=30)

        self.knn_samples = []
        if self.use_knn:
            model_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..",
                             "sign_language_app", "model.json")
            )
            try:
                if os.path.exists(model_path):
                    with open(model_path, "r", encoding="utf-8") as handle:
                        self.knn_samples = json.load(handle).get("samples", [])
                    print(f"KNN 模型載入成功：{len(self.knn_samples)} 筆樣本")
            except (OSError, ValueError) as exc:
                print(f"KNN 模型載入失敗：{exc}")

    def process(self, frame, face_landmarks=None):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)

        detections = []
        if not results.multi_hand_landmarks:
            self.history.append("No hand")
            self.current_knn_confidence = 0.0
            self.knn_window.clear()
            return detections

        handedness_list = results.multi_handedness or []

        for index, landmarks in enumerate(results.multi_hand_landmarks):
            handedness = "Unknown"
            if index < len(handedness_list):
                handedness = handedness_list[index].classification[0].label
            
            detections.append({
                "landmarks": landmarks,
                "handedness": handedness,
                "fingers": {},
                "sign": "Unknown",
                "joint_points": self.joint_points(frame, landmarks),
            })

        knn_sign = "Unknown"
        knn_confidence = 0.0
        if self.use_knn and self.knn_samples:
            # 呼叫底下的完美特徵提取
            vector = self._normalize_landmarks(
                results.multi_hand_landmarks, face_landmarks
            )
            
            # 滑動視窗平均
            self.knn_window.append(vector)
            if len(self.knn_window) > 0:
                window_list = list(self.knn_window)
                avg_vector = [
                    sum(v[d] for v in window_list) / len(window_list)
                    for d in range(len(window_list[0]))
                ]
                knn_sign, knn_confidence = self._classify_knn(avg_vector, k=self.KNN_K)
            
            # 覆蓋結果
            for detection in detections:
                if (knn_sign != "Unknown" and knn_confidence >= self.KNN_MIN_CONFIDENCE):
                    detection["sign"] = knn_sign
                else:
                    detection["sign"] = "Unknown"
                    
        self.current_knn_confidence = knn_confidence

        if len(detections) >= 1:
            self.history.append(detections[0]["sign"])

        self._update_sentence()
        return detections

    def _normalize_landmarks(self, multi_hand_landmarks, face_landmarks=None):
        """
        【核心修正】完美分離左右手與臉部特徵，總計 186 維。
        """
        hands_data = []
        if multi_hand_landmarks:
            for h in multi_hand_landmarks:
                hands_data.append([{"x": p.x, "y": p.y, "z": p.z} for p in h.landmark])
        
        # 1. 將手按 X 座標排序（左邊永遠是左手，右邊永遠是右手）
        hands_data = sorted(hands_data, key=lambda h: h[0]["x"])
        
        # 2. 提取雙手特徵 (各 63 維)
        hand_vec = []
        for i in range(2):
            if i < len(hands_data):
                hand = hands_data[i]
                wrist = hand[0]
                middle_mcp = hand[9]
                scale = math.sqrt((middle_mcp["x"] - wrist["x"])**2 + (middle_mcp["y"] - wrist["y"])**2 + (middle_mcp["z"] - wrist["z"])**2) or 1e-6
                for p in hand:
                    hand_vec.extend([
                        (p["x"] - wrist["x"]) / scale,
                        (p["y"] - wrist["y"]) / scale,
                        (p["z"] - wrist["z"]) / scale,
                    ])
            else:
                hand_vec.extend([0.0] * 63) # 單手時另一隻手補 0

        # 3. 提取臉部特徵 (60 維)
        face_vec = []
        if face_landmarks:
            pts = face_landmarks.landmark
            nose = pts[1]
            left_eye = pts[33]
            right_eye = pts[263]
            fscale = math.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2) or 1e-6
            for idx in self.FACE_KEY_INDICES:
                p = pts[idx]
                face_vec.extend([
                    (p.x - nose.x) / fscale,
                    (p.y - nose.y) / fscale,
                    (p.z - nose.z) / fscale,
                ])
        else:
            face_vec.extend([0.0] * 60)

        return hand_vec + face_vec

    def _classify_knn(self, vector, k=5):
        if not self.knn_samples:
            return "Unknown", 0.0
        neighbors = []
        for sample in self.knn_samples:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vector, sample["vector"])))
            neighbors.append((dist, sample["label"]))
        neighbors.sort(key=lambda item: item[0])
        top_k = neighbors[:min(k, len(neighbors))]
        votes = Counter()
        for distance, label in top_k:
            votes[label] += 1.0 / max(distance, 1e-6)
        total = sum(votes.values())
        if total <= 0:
            return "Unknown", 0.0
        best_label, best_weight = max(votes.items(), key=lambda item: item[1])
        return best_label, best_weight / total

    def draw(self, frame, detections, target_sign=None, face_expression=None):
        self._draw_hands(frame, detections)
        self._draw_hud(frame, target_sign, face_expression)
        self._draw_sentence(frame)

    def _draw_hands(self, frame, detections):
        for detection in detections:
            mp_drawing.draw_landmarks(
                frame, detection["landmarks"], mp_hands.HAND_CONNECTIONS,
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
            metric = f"  KNN {self.current_knn_confidence:.0%}" if self.use_knn and self.current_knn_confidence > 0 else f"  {stable['confidence']:.0%}"
            rows.append({
                "text": f"手語: {display_label(stable_sign)}{metric}",
                "color": (0, 255, 255),
                "font_size": 24,
            })
        if face_expression:
            rows.append({
                "text": f"表情: {face_expression}",
                "color": (255, 200, 0),
                "font_size": 22,
            })

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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, row["color"], 2,
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
                position, cv2.FONT_HERSHEY_SIMPLEX, font_size / 32, color, 2,
            )

    @property
    def stable_sign(self):
        return self.stable_status["sign"]

    @property
    def stable_status(self):
        if not self.history:
            return {
                "sign": "No hand", "count": 0, "total": 0,
                "confidence": 0.0, "is_stable": False,
            }
        sign, count = Counter(self.history).most_common(1)[0]
        total = len(self.history)
        confidence = count / total
        return {
            "sign": sign, "count": count, "total": total,
            "confidence": confidence,
            "is_stable": count >= self.stable_min_count
            and confidence >= self.STABLE_CONFIDENCE_THRESHOLD,
        }

    @classmethod
    def normalize_sign(cls, sign):
        if not sign:
            return ""
        return str(sign).strip()

    @classmethod
    def target_matches(cls, stable_sign, target_sign, is_stable=True):
        if not target_sign or not is_stable:
            return False
        return cls.normalize_sign(stable_sign) == cls.normalize_sign(target_sign)

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
            name = self.LANDMARK_NAMES[index] if hasattr(self, 'LANDMARK_NAMES') else str(index)
            points[name] = {
                "x": int(point.x * width),
                "y": int(point.y * height),
                "z": round(point.z, 4),
            }
        return points