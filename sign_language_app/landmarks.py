import numpy as np

FACE_KEY_INDICES = [33, 133, 159, 145, 362, 263, 386, 374, 1, 2, 98, 327, 61, 291, 13, 14, 78, 308, 82, 312]

def normalize_single_hand(hand):
    """將單手 21 點轉為 63 維，以該手手腕為原點"""
    if not hand: return [0.0] * 63
    wrist = hand[0]
    middle_mcp = hand[9]
    dx = middle_mcp["x"] - wrist["x"]
    dy = middle_mcp["y"] - wrist["y"]
    dz = middle_mcp["z"] - wrist["z"]
    scale = np.sqrt(dx**2 + dy**2 + dz**2) or 1e-6
    
    result = []
    for landmark in hand:
        result.extend([
            (landmark["x"] - wrist["x"]) / scale,
            (landmark["y"] - wrist["y"]) / scale,
            (landmark["z"] - wrist["z"]) / scale,
        ])
    return result

def normalize_face(face_landmarks_list):
    if not face_landmarks_list: return [0.0] * 60
    nose = face_landmarks_list[1]
    left_eye = face_landmarks_list[33]
    right_eye = face_landmarks_list[263]
    fscale = np.sqrt((left_eye["x"] - right_eye["x"]) ** 2 + (left_eye["y"] - right_eye["y"]) ** 2) or 1e-6
    result = []
    for idx in FACE_KEY_INDICES:
        p = face_landmarks_list[idx]
        result.extend([(p["x"] - nose["x"]) / fscale, (p["y"] - nose["y"]) / fscale, (p["z"] - nose["z"]) / fscale])
    return result

def build_full_vector(hands_list, face_landmarks_list=None):
    """
    特徵結構：[左手 63維] + [右手 63維] + [臉部 60維] = 186 維
    如果只有一隻手，第二隻手補 0。
    """
    # 將雙手按 x 座標排序，確保左手永遠在左邊，右手永遠在右邊 (避免左右手顛倒)
    hands_list = sorted(hands_list, key=lambda h: h[0]["x"])
    
    hand_vec = []
    for i in range(2):
        if i < len(hands_list):
            hand_vec.extend(normalize_single_hand(hands_list[i]))
        else:
            hand_vec.extend([0.0] * 63) # 沒手就補 0
            
    face_vec = normalize_face(face_landmarks_list) if face_landmarks_list else [0.0] * 60
    return hand_vec + face_vec # 總共 186 維

def mediapipe_landmarks_to_list(landmarks):
    return [{"x": point.x, "y": point.y, "z": point.z} for point in landmarks.landmark]

def mediapipe_face_to_list(face_landmarks):
    if face_landmarks is None: return None
    return [{"x": point.x, "y": point.y, "z": point.z} for point in face_landmarks.landmark]