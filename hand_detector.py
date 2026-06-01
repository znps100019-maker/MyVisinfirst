from collections import Counter, deque
import math

import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class HandSignRecognizer:
    """Detect hand landmarks and classify simple static hand signs."""

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
        "Open palm",
        "Fist",
        "Thumbs up",
        "OK",
        "Number 1",
        "Number 2",
        "Number 3",
        "Number 4",
        "I love you",
        "Pinky",
        "Unknown",
        "No hand",
    ]
    SIGN_ALIASES = {
        "1": "Number 1",
        "2": "Number 2",
        "3": "Number 3",
        "4": "Number 4",
        "point": "Number 1",
        "point / 1": "Number 1",
        "v": "Number 2",
        "peace": "Number 2",
        "v / peace": "Number 2",
        "five": "Open palm",
        "open": "Open palm",
        "palm": "Open palm",
        "thumb": "Thumbs up",
        "thumbs": "Thumbs up",
    }

    def __init__(self, max_num_hands=2, history_size=8, stable_min_count=5):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.history = deque(maxlen=history_size)
        self.stable_min_count = stable_min_count

    def process(self, frame):
        """Return detected hands with landmarks, handedness, and sign labels."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)

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

        if detections:
            self.history.append(detections[0]["sign"])

        return detections

    def draw(self, frame, detections, target_sign=None):
        """Draw hand skeletons and sign labels on a video frame."""
        for detection in detections:
            mp_drawing.draw_landmarks(
                frame,
                detection["landmarks"],
                mp_hands.HAND_CONNECTIONS,
            )

            x, y = self._label_position(frame, detection["landmarks"])
            label = f'{detection["handedness"]}: {detection["sign"]}'
            cv2.putText(
                frame,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

        stable = self.stable_status
        stable_sign = stable["sign"]
        cv2.rectangle(frame, (0, 100), (520, 190), (0, 0, 0), cv2.FILLED)
        cv2.putText(
            frame,
            f"SIGN: {stable_sign} ({stable['confidence']:.0%})",
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )
        if target_sign:
            target_sign = self.normalize_sign(target_sign)
            detected = self.is_target_detected(target_sign)
            color = (0, 255, 0) if detected else (180, 180, 180)
            text = "TARGET HIT" if detected else "TARGET WAIT"
            cv2.putText(
                frame,
                f"{text}: {target_sign}",
                (10, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
            )

    def close(self):
        self.hands.close()

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
        return {
            "sign": sign,
            "count": count,
            "total": total,
            "confidence": count / total,
            "is_stable": count >= self.stable_min_count,
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

    def is_target_detected(self, target_sign):
        target_sign = self.normalize_sign(target_sign)
        stable = self.stable_status
        return (
            bool(target_sign)
            and stable["is_stable"]
            and self.normalize_sign(stable["sign"]) == target_sign
        )

    def _extended_fingers(self, landmarks, handedness):
        points = landmarks.landmark
        fingers = {}

        for finger, tip_id in self.FINGER_TIPS.items():
            if finger == "thumb":
                fingers[finger] = self._is_thumb_extended(points, handedness)
                continue

            pip_id = self.FINGER_PIPS[finger]
            mcp_id = self.FINGER_MCPS[finger]
            fingers[finger] = (
                points[tip_id].y < points[pip_id].y
                and points[pip_id].y < points[mcp_id].y
            )

        return fingers

    def _is_thumb_extended(self, points, handedness):
        tip = points[self.FINGER_TIPS["thumb"]]
        mcp = points[self.FINGER_MCPS["thumb"]]

        if handedness == "Right":
            return tip.x < mcp.x
        if handedness == "Left":
            return tip.x > mcp.x

        return abs(tip.x - mcp.x) > 0.08

    def _classify_sign(self, fingers, landmarks):
        thumb = fingers["thumb"]
        index = fingers["index"]
        middle = fingers["middle"]
        ring = fingers["ring"]
        pinky = fingers["pinky"]

        if self._is_ok_sign(landmarks) and middle and ring and pinky:
            return "OK"
        if all(fingers.values()):
            return "Open palm"
        if not any(fingers.values()):
            return "Fist"
        if thumb and not any([index, middle, ring, pinky]):
            return "Thumbs up"
        if index and middle and not any([thumb, ring, pinky]):
            return "Number 2"
        if index and middle and ring and not any([thumb, pinky]):
            return "Number 3"
        if index and middle and ring and pinky and not thumb:
            return "Number 4"
        if index and not any([thumb, middle, ring, pinky]):
            return "Number 1"
        if thumb and index and pinky and not middle and not ring:
            return "I love you"
        if pinky and not any([thumb, index, middle, ring]):
            return "Pinky"

        return "Unknown"

    def _is_ok_sign(self, landmarks):
        points = landmarks.landmark
        thumb_tip = points[self.FINGER_TIPS["thumb"]]
        index_tip = points[self.FINGER_TIPS["index"]]
        wrist = points[0]
        middle_mcp = points[self.FINGER_MCPS["middle"]]

        pinch_distance = self._distance(thumb_tip, index_tip)
        palm_size = max(self._distance(wrist, middle_mcp), 0.001)
        return pinch_distance / palm_size < 0.45

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
