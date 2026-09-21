"""
臉部表情辨識模組 (Facial Expression Recognition)
基於 MediaPipe FaceMesh 與臉部動作編碼系統 (FACS, Facial Action Coding System)，
支援台灣/華人傳統四大核心情感「喜、怒、哀、樂」與「平靜」即時分類，
並支援微表情（眨眼、眨單眼、張嘴驚訝）動態偵測與時序平滑濾波。
"""
import math
from collections import Counter, deque
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


class FaceExpressionRecognizer:
    """
    Detect facial landmarks and classify static & dynamic facial expressions.
    
    支援「喜、怒、哀、樂」四大核心情緒與平靜基線：
    - 喜 (Happy / Smiling / 微笑、歡喜)：AU12 (提口角肌) + 嘴角上揚
    - 怒 (Angry / Frowning / 憤怒、皺眉)：AU4 (皺眉肌/降眉肌) + 眉心緊縮 + 眉毛下壓
    - 哀 (Sad / Drooping / 悲傷、難過)：AU15 (降口角肌) + 嘴角下垂 + AU1 (八字眉)
    - 樂 (Joy / Laughing / 開懷大笑、狂喜)：AU12 + AU25/AU26 (張嘴大笑、笑逐顏開)
    - 平靜 (Neutral / 平常心)：各項肌肉處於放鬆基線區間
    
    微表情與眼睛動作：
    - 閉眼 (Blink), 眨左眼 (Wink Left), 眨右眼 (Wink Right), 驚訝 (Surprise / 僅張嘴)
    """

    EMOTIONS = ("喜", "怒", "哀", "樂", "平靜")

    EMOTION_LABELS = {
        "喜": "喜 (Happy / 微笑)",
        "怒": "怒 (Angry / 生氣)",
        "哀": "哀 (Sad / 難過)",
        "樂": "樂 (Joy / 大笑)",
        "平靜": "平靜 (Neutral)",
    }

    EMOTION_EN = {
        "喜": "Happy",
        "怒": "Angry",
        "哀": "Sad",
        "樂": "Joy",
        "平靜": "Neutral",
    }

    EMOTION_EMOJIS = {
        "喜": "😊",
        "怒": "😠",
        "哀": "😢",
        "樂": "😄",
        "平靜": "😐",
    }

    EMOTION_COLORS_BGR = {
        "喜": (0, 240, 255),    # 金黃/暖黃
        "怒": (60, 60, 255),    # 鮮紅/警戒
        "哀": (255, 180, 70),   # 柔和青藍
        "樂": (0, 255, 180),    # 歡樂亮綠
        "平靜": (210, 210, 210)  # 簡潔淡灰
    }

    def __init__(self, max_num_faces=1, smoothing_window=5, auto_init_mesh=True):
        self.max_num_faces = max_num_faces
        self.smoothing_window = smoothing_window
        self.history = deque(maxlen=smoothing_window)
        self.last_stable_emotion = "平靜"
        self.last_scores = {e: 0.2 for e in self.EMOTIONS}
        self._face_mesh = None

        if auto_init_mesh:
            self._init_face_mesh()

    def _init_face_mesh(self):
        """Lazy initializer for MediaPipe FaceMesh."""
        if self._face_mesh is None:
            self._face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=self.max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    @property
    def face_mesh(self):
        if self._face_mesh is None:
            self._init_face_mesh()
        return self._face_mesh

    def classify_landmarks(self, landmarks):
        """
        Classify 478 MediaPipe facial landmarks directly into FACS Action Units
        and emotions (喜、怒、哀、樂、平靜).
        
        landmarks can be a MediaPipe LandmarkList or a list of Point objects with .x, .y, .z.
        """
        pts = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks

        # 1. 眼睛長寬比 (Eye Aspect Ratio, EAR)
        # 左眼: 362 (內), 385 (右上), 386 (左上), 263 (外), 374 (左下), 380 (右下)
        left_ear = self._eye_aspect_ratio(pts, 362, 385, 386, 263, 374, 380)
        # 右眼: 33 (外), 159 (左上), 158 (右上), 133 (內), 145 (右下), 153 (左下)
        right_ear = self._eye_aspect_ratio(pts, 33, 159, 158, 133, 145, 153)

        is_left_closed = left_ear < 0.19
        is_right_closed = right_ear < 0.19
        is_blink = is_left_closed and is_right_closed

        # 2. 基準尺度 (Scale References)
        d_eyes = max(self._distance(pts[33], pts[263]), 0.001)
        d_face_height = max(self._distance(pts[10], pts[152]), 0.001)

        # 3. 嘴部幾何特徵 (AU12 提口角, AU25/26 張嘴, AU15 降口角)
        d_mouth = self._distance(pts[61], pts[291])
        smile_ratio = d_mouth / d_eyes

        d_inner_mouth = self._distance(pts[13], pts[14])
        mouth_ratio = d_inner_mouth / d_face_height

        # 嘴角垂直高低 (Corner Elevation)：相較於唇中心高度 (13 與 14 中點)
        # 螢幕坐標 y 朝下：嘴角 y 越小代表向上提 (嘴角上揚)
        mouth_center_y = (pts[13].y + pts[14].y) / 2.0
        corners_y = (pts[61].y + pts[291].y) / 2.0
        corner_elevation = (mouth_center_y - corners_y) / d_eyes

        # 4. 眉毛幾何特徵 (AU4 降眉/皺眉, AU1 挑眉心)
        # 眉心距離 (Glabella width)
        d_brow_inner = self._distance(pts[107], pts[336]) / d_eyes

        # 眉眼垂直距離
        brow_eye_r = self._distance(pts[107], pts[133]) / d_eyes
        brow_eye_l = self._distance(pts[336], pts[362]) / d_eyes
        avg_brow_eye_dist = (brow_eye_r + brow_eye_l) / 2.0

        # 眉毛傾斜度 (Brow Slant)
        slant_r = (pts[107].y - pts[70].y) / d_eyes
        slant_l = (pts[336].y - pts[300].y) / d_eyes
        avg_brow_slant = (slant_r + slant_l) / 2.0

        # 5. 計算喜怒哀樂與平靜之特徵得分
        score_le = 0.0
        if mouth_ratio > 0.048 and smile_ratio > 0.48:
            score_le = min(1.0, (smile_ratio - 0.46) * 3.5 + (mouth_ratio - 0.04) * 7.5)

        score_xi = 0.0
        if smile_ratio > 0.47 or corner_elevation > 0.015:
            score_xi = min(1.0, max(0.0, (smile_ratio - 0.45) * 3.0 + max(0.0, corner_elevation) * 12.0))
            if mouth_ratio > 0.065:
                score_xi *= 0.45

        score_nu = 0.0
        if d_brow_inner < 0.315 and avg_brow_eye_dist < 0.20:
            tight_score = max(0.0, (0.315 - d_brow_inner) * 9.0)
            lower_score = max(0.0, (0.20 - avg_brow_eye_dist) * 8.0)
            slant_score = max(0.0, avg_brow_slant * 6.0)
            score_nu = min(1.0, tight_score * 0.5 + lower_score * 0.35 + slant_score * 0.15)
            if smile_ratio > 0.49:
                score_nu *= 0.15

        score_ai = 0.0
        if corner_elevation < -0.012 or avg_brow_slant < -0.010:
            droop_score = max(0.0, (-0.010 - corner_elevation) * 16.0)
            sad_brow_score = max(0.0, (-avg_brow_slant) * 10.0)
            score_ai = min(1.0, droop_score * 0.65 + sad_brow_score * 0.35)
            if smile_ratio > 0.48 or mouth_ratio > 0.08:
                score_ai *= 0.15

        score_neutral = 0.32

        scores = {
            "喜": round(float(score_xi), 3),
            "怒": round(float(score_nu), 3),
            "哀": round(float(score_ai), 3),
            "樂": round(float(score_le), 3),
            "平靜": round(float(score_neutral), 3),
        }
        self.last_scores = scores

        # 6. 決定主候選情緒
        candidates = [("樂", score_le), ("喜", score_xi), ("怒", score_nu), ("哀", score_ai)]
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_emotion, best_score = candidates[0]

        if best_score < 0.28:
            candidate_emotion = "平靜"
            confidence = max(0.50, round(1.0 - best_score, 3))
        else:
            candidate_emotion = best_emotion
            confidence = round(best_score, 3)

        # 7. 時序平滑濾波 (Temporal Smoothing)
        self.history.append(candidate_emotion)
        counts = Counter(self.history)
        most_common_emotion, count = counts.most_common(1)[0]
        if count >= len(self.history) // 2 + 1:
            active_emotion = most_common_emotion
        else:
            active_emotion = self.last_stable_emotion
        self.last_stable_emotion = active_emotion

        # 8. 附帶微表情檢驗 (眨眼、眨單眼、驚訝張嘴)
        micro_action = ""
        if is_blink:
            micro_action = " [閉眼]"
        elif is_left_closed:
            micro_action = " [眨左眼]"
        elif is_right_closed:
            micro_action = " [眨右眼]"
        elif mouth_ratio > 0.085 and active_emotion not in ("樂", "喜"):
            micro_action = " [張嘴/驚訝]"

        full_expression = f"{self.EMOTION_EMOJIS[active_emotion]} {self.EMOTION_LABELS[active_emotion]}{micro_action}".strip()

        return {
            "landmarks": landmarks,
            "expression": full_expression,
            "emotion": active_emotion,
            "emotion_en": self.EMOTION_EN[active_emotion],
            "confidence": confidence,
            "scores": scores,
            "is_smiling": smile_ratio > 0.48 or corner_elevation > 0.015,
            "is_mouth_open": mouth_ratio > 0.055,
            "is_frowning": d_brow_inner < 0.295,
            "is_sad": corner_elevation < -0.015,
            "left_ear": round(left_ear, 3),
            "right_ear": round(right_ear, 3),
            "smile_ratio": round(smile_ratio, 3),
            "mouth_ratio": round(mouth_ratio, 3),
            "corner_elevation": round(corner_elevation, 3),
            "brow_dist_ratio": round(d_brow_inner, 3),
            "brow_eye_ratio": round(avg_brow_eye_dist, 3),
        }

    def process(self, frame):
        """Processes an image frame and returns detected expression or None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        if self._face_mesh is None:
            self._init_face_mesh()

        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        return self.classify_landmarks(results.multi_face_landmarks[0])

    def draw(self, frame, face_data):
        """Draw face mesh landmarks and emotion highlight badge on the frame."""
        if not face_data or "landmarks" not in face_data:
            return

        landmarks = face_data["landmarks"]

        # 1. 臉部細部網格 (Tesselation) - 科技感冷青色
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(220, 220, 90), thickness=1, circle_radius=1
            ),
        )

        # 2. 輪廓特徵線 (Contours) - 依據當前情緒切換動態光彩
        active_emotion = face_data.get("emotion", "平靜")
        contour_color = self.EMOTION_COLORS_BGR.get(active_emotion, (0, 120, 255))
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=contour_color, thickness=1, circle_radius=1
            ),
        )

    def close(self):
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None

    def _distance(self, p1, p2):
        return math.sqrt(
            (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2
        )

    def _eye_aspect_ratio(self, points, p1, p2, p3, p4, p5, p6):
        d_vert1 = self._distance(points[p2], points[p6])
        d_vert2 = self._distance(points[p3], points[p5])
        d_horiz = self._distance(points[p1], points[p4])
        return (d_vert1 + d_vert2) / max(2.0 * d_horiz, 0.001)
