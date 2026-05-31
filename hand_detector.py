from collections import Counter, deque

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

    def __init__(self, max_num_hands=2, history_size=8):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.history = deque(maxlen=history_size)

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
            sign = self._classify_sign(fingers)
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

    def draw(self, frame, detections):
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

        stable_sign = self.stable_sign
        cv2.rectangle(frame, (0, 100), (360, 160), (0, 0, 0), cv2.FILLED)
        cv2.putText(
            frame,
            f"SIGN: {stable_sign}",
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

    def close(self):
        self.hands.close()

    @property
    def stable_sign(self):
        if not self.history:
            return "No hand"
        return Counter(self.history).most_common(1)[0][0]

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

    def _classify_sign(self, fingers):
        thumb = fingers["thumb"]
        index = fingers["index"]
        middle = fingers["middle"]
        ring = fingers["ring"]
        pinky = fingers["pinky"]

        if all(fingers.values()):
            return "Open palm"
        if not any(fingers.values()):
            return "Fist"
        if thumb and not any([index, middle, ring, pinky]):
            return "Thumbs up"
        if index and middle and not any([thumb, ring, pinky]):
            return "V / Peace"
        if index and not any([thumb, middle, ring, pinky]):
            return "Point / 1"
        if thumb and index and pinky and not middle and not ring:
            return "I love you"
        if pinky and not any([thumb, index, middle, ring]):
            return "Pinky"

        return "Unknown"

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
