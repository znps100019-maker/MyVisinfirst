import numpy as np

# 臉部關鍵點索引（20 個）
FACE_KEY_INDICES = [
    33, 133, 159, 145,   # 左眼
    362, 263, 386, 374,  # 右眼
    1, 2, 98, 327,       # 鼻子
    61, 291, 13, 14,     # 嘴角
    78, 308, 82, 312,    # 臉頰
]


def normalize_landmarks(landmarks_list):
    """單手 21 點 → 63 維（保留原本邏輯，向後相容）"""
    wrist = landmarks_list[0]
    middle_mcp = landmarks_list[9]
    dx = middle_mcp["x"] - wrist["x"]
    dy = middle_mcp["y"] - wrist["y"]
    dz = middle_mcp["z"] - wrist["z"]
    scale = np.sqrt(dx**2 + dy**2 + dz**2)
    if scale == 0:
        scale = 1e-6
    normalized = []
    for landmark in landmarks_list:
        normalized.extend([
            (landmark["x"] - wrist["x"]) / scale,
            (landmark["y"] - wrist["y"]) / scale,
            (landmark["z"] - wrist["z"]) / scale,
        ])
    return normalized


def normalize_two_hands(hands_list):
    """
    hands_list: list of landmarks_list（每手 21 點）
    回傳：126 維（第一手 63 + 第二手 63，不足補零）
    """
    result = []
    for hand in hands_list[:2]:
        # 以第一手的掌寬為縮放基準（兩手共用同一尺度，保留雙手相對位置）
        wrist = hand[0]
        middle_mcp = hand[9]
        dx = middle_mcp["x"] - wrist["x"]
        dy = middle_mcp["y"] - wrist["y"]
        dz = middle_mcp["z"] - wrist["z"]
        scale = np.sqrt(dx**2 + dy**2 + dz**2)
        if scale == 0:
            scale = 1e-6
        for landmark in hand:
            result.extend([
                (landmark["x"] - wrist["x"]) / scale,
                (landmark["y"] - wrist["y"]) / scale,
                (landmark["z"] - wrist["z"]) / scale,
            ])
    # 補齊到 126 維
    result.extend([0.0] * (126 - len(result)))
    return result[:126]


def normalize_face(face_landmarks_list):
    """
    face_landmarks_list: 468 點的 list of dict
    回傳：60 維（取 20 個關鍵點 × 3）
    """
    if not face_landmarks_list:
        return [0.0] * 60
    nose = face_landmarks_list[1]
    # 以兩眼距離為縮放基準
    left_eye = face_landmarks_list[33]
    right_eye = face_landmarks_list[263]
    dx = left_eye["x"] - right_eye["x"]
    dy = left_eye["y"] - right_eye["y"]
    scale = np.sqrt(dx**2 + dy**2)
    if scale == 0:
        scale = 1e-6
    result = []
    for idx in FACE_KEY_INDICES:
        p = face_landmarks_list[idx]
        result.extend([
            (p["x"] - nose["x"]) / scale,
            (p["y"] - nose["y"]) / scale,
            (p["z"] - nose["z"]) / scale,
        ])
    return result


def build_full_vector(hands_list, face_landmarks_list=None):
    """
    合併雙手 + 臉部 → 186 維
    hands_list: list of landmarks_list（0～2 手）
    face_landmarks_list: 468 點 list of dict 或 None
    """
    hand_vec = normalize_two_hands(hands_list) if hands_list else [0.0] * 126
    face_vec = normalize_face(face_landmarks_list) if face_landmarks_list else [0.0] * 60
    return hand_vec + face_vec  # 186 維


def mediapipe_landmarks_to_list(landmarks):
    """MediaPipe 手部物件 → list of dict"""
    return [{"x": point.x, "y": point.y, "z": point.z} for point in landmarks.landmark]


def mediapipe_face_to_list(face_landmarks):
    """MediaPipe 臉部物件 → list of dict（468 點）"""
    if face_landmarks is None:
        return None
    return [{"x": point.x, "y": point.y, "z": point.z} for point in face_landmarks.landmark]