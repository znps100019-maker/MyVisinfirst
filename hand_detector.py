from collections import Counter, deque
import math

import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class HandSignRecognizer:
    """Detect hand landmarks and classify simple static hand signs."""

    # MediaPipe Hands 會回傳 21 個手部關節點，這些數字是各手指指尖的 landmark index。
    FINGER_TIPS = {
        "thumb": 4,
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20,
    }
    # PIP 是手指中間關節，用來判斷手指是否伸直。
    FINGER_PIPS = {
        "index": 6,
        "middle": 10,
        "ring": 14,
        "pinky": 18,
    }
    # MCP 是手指根部關節，可和 PIP、TIP 一起比較位置。
    FINGER_MCPS = {
        "thumb": 2,
        "index": 5,
        "middle": 9,
        "ring": 13,
        "pinky": 17,
    }
    # 把 MediaPipe 的 0~20 關節編號轉成容易理解的名稱。
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
        # min_detection_confidence 越高越不容易誤判，但太高可能比較難偵測到手。
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        # history 用來保存最近幾次辨識結果，避免手勢瞬間跳動造成誤判。
        self.history = deque(maxlen=history_size)
        self.stable_min_count = stable_min_count

    def process(self, frame):
        """Return detected hands with landmarks, handedness, and sign labels."""
        # OpenCV 讀到的是 BGR，MediaPipe 需要 RGB，所以要先轉色彩格式。
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)

        detections = []
        if not results.multi_hand_landmarks:
            # 沒有看到手時也記錄 No hand，讓穩定狀態能正確回到無手。
            self.history.append("No hand")
            return detections

        handedness_list = results.multi_handedness or []

        for index, landmarks in enumerate(results.multi_hand_landmarks):
            handedness = "Unknown"
            if index < len(handedness_list):
                handedness = handedness_list[index].classification[0].label

            # 先判斷每根手指是否伸直，再用規則分類成手勢名稱。
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
            # 畫出 MediaPipe 手部骨架線與 21 個關節點。
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
        
        # 只有在偵測到手勢（非 No hand）時，才顯示精簡的提示框，避免遮擋鏡頭
        if stable_sign != "No hand":
            # 依據是否有目標手勢，動態決定黑框高度
            box_bottom = 85 if target_sign else 55
            cv2.rectangle(frame, (10, 10), (325, box_bottom), (0, 0, 0), cv2.FILLED)
            
            cv2.putText(
                frame,
                f"SIGN: {stable_sign} ({stable['confidence']:.0%})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
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
                    (20, 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
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
        # 從最近幾次辨識結果中選出出現最多次的手勢，當作穩定手勢。
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
        # 允許使用者輸入 peace、1、thumb 這類簡短名稱，轉成正式手勢名稱。
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
        # 只有「穩定手勢」等於目標手勢時才算命中，避免單幀誤判。
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
        wrist = points[0]

        for finger, tip_id in self.FINGER_TIPS.items():
            if finger == "thumb":
                fingers[finger] = self._is_thumb_extended(points, handedness)
                continue

            # 其他四隻手指：使用雙重保險判定（直線比例 + 手腕距離比例）
            base_idx = self.FINGER_MCPS[finger]
            mcp = points[base_idx]
            pip = points[base_idx + 1]  # PIP
            dip = points[base_idx + 2]  # DIP
            tip = points[base_idx + 3]  # TIP

            straight = self._distance(mcp, tip)
            segments = (
                self._distance(mcp, pip)
                + self._distance(pip, dip)
                + self._distance(dip, tip)
            )
            # 1. 3D 直線距離與分段總和比例大於 0.82
            is_straight = (straight / max(segments, 0.001)) > 0.82
            
            # 2. 指尖到手腕的距離必須大於 PIP 關節到手腕的距離（加上 1.1 倍安全係數，防止收拳/遮擋誤判）
            d_wrist_tip = self._distance(wrist, tip)
            d_wrist_pip = self._distance(wrist, pip)
            is_extended_from_wrist = d_wrist_tip > (d_wrist_pip * 1.1)

            fingers[finger] = is_straight and is_extended_from_wrist

        return fingers

    def _is_thumb_extended(self, points, handedness):
        # 大拇指：利用 CMC(1) 到 TIP(4) 的直線長度與分段長度總和比例判定（排除左右手與旋轉限制）
        cmc = points[1]
        mcp = points[2]
        ip = points[3]
        tip = points[4]
        
        straight = self._distance(cmc, tip)
        segments = (
            self._distance(cmc, mcp)
            + self._distance(mcp, ip)
            + self._distance(ip, tip)
        )
        return (straight / max(segments, 0.001)) > 0.86

    def _thumb_spread_ratio(self, landmarks):
        # 計算大拇指指尖與食指根部的空間距離，並相對於手掌大小進行標準化
        points = landmarks.landmark
        thumb_tip = points[self.FINGER_TIPS["thumb"]]
        index_mcp = points[self.FINGER_MCPS["index"]]
        wrist = points[0]
        middle_mcp = points[self.FINGER_MCPS["middle"]]
        
        distance = self._distance(thumb_tip, index_mcp)
        palm_size = max(self._distance(wrist, middle_mcp), 0.001)
        return distance / palm_size

    def _classify_sign(self, fingers, landmarks):
        # 這裡是規則式手勢分類：依照哪幾根手指伸直來決定手勢。
        thumb = fingers["thumb"]
        index = fingers["index"]
        middle = fingers["middle"]
        ring = fingers["ring"]
        pinky = fingers["pinky"]

        # OK 手勢優先判定
        if self._is_ok_sign(landmarks) and middle and ring and pinky:
            return "OK"

        # 特殊手勢（我愛你、小指）
        if thumb and index and pinky and not middle and not ring:
            return "I love you"
        if pinky and not any([thumb, index, middle, ring]):
            return "Pinky"

        # 基礎狀態（握拳）
        if not any(fingers.values()):
            return "Fist"

        # 透過大拇指張開幅度（與食指根部的距離比例）來精準區分 1-4 與 5-9
        spread_ratio = self._thumb_spread_ratio(landmarks)
        is_thumb_spread = spread_ratio > 0.58

        if not is_thumb_spread:
            # 大拇指收起/貼近手掌：判定 1-4 與 拳頭
            if index and middle and ring and pinky:
                return "Number 4"
            if index and middle and ring and not pinky:
                return "Number 3"
            if index and middle and not ring and not pinky:
                return "Number 2"
            if index and not middle and not ring and not pinky:
                return "Number 1"
        else:
            # 大拇指張開：判定 5-9
            if index and middle and ring and pinky:
                return "Number 9 (Open palm)"
            if index and middle and ring and not pinky:
                return "Number 8"
            if index and middle and not ring and not pinky:
                return "Number 7"
            if index and not middle and not ring and not pinky:
                return "Number 6"
            if not any([index, middle, ring, pinky]):
                return "Number 5 (Thumbs up)"

        return "Unknown"

    def _is_ok_sign(self, landmarks):
        # OK 手勢的特徵是大拇指指尖和食指指尖靠近形成圈。
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
