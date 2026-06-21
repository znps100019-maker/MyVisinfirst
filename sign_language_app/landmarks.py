import numpy as np


def normalize_landmarks(landmarks_list):
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
        normalized.extend(
            [
                (landmark["x"] - wrist["x"]) / scale,
                (landmark["y"] - wrist["y"]) / scale,
                (landmark["z"] - wrist["z"]) / scale,
            ]
        )

    return normalized


def mediapipe_landmarks_to_list(landmarks):
    return [{"x": point.x, "y": point.y, "z": point.z} for point in landmarks.landmark]
