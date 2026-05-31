import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose


class ArmDetector:
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def find_arm_points(self, img):
        """Return right shoulder, elbow, and wrist points from a video frame."""
        height, width, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        shoulder = [
            int(landmarks[12].x * width),
            int(landmarks[12].y * height),
        ]
        elbow = [
            int(landmarks[14].x * width),
            int(landmarks[14].y * height),
        ]
        wrist = [
            int(landmarks[16].x * width),
            int(landmarks[16].y * height),
        ]

        return shoulder, elbow, wrist

    def find_arm_joints(self, img):
        points = self.find_arm_points(img)
        if points is None:
            return None

        shoulder, elbow, wrist = points
        return {
            "shoulder": {"x": shoulder[0], "y": shoulder[1]},
            "elbow": {"x": elbow[0], "y": elbow[1]},
            "wrist": {"x": wrist[0], "y": wrist[1]},
            "angle": self.calculate_angle(shoulder, elbow, wrist),
        }

    def close(self):
        self.pose.close()

    def calculate_angle(self, a, b, c):
        """Calculate the angle at point b from points a, b, and c."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
            a[1] - b[1],
            a[0] - b[0],
        )
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle
        return int(angle)
