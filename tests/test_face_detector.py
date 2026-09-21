"""
單元測試：臉部表情辨識模組 (喜怒哀樂與微表情檢測)
"""
import unittest
from core.detectors.face_detector import FaceExpressionRecognizer


class MockLandmarkPoint:
    def __init__(self, x=0.5, y=0.5, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


def create_baseline_landmarks():
    """Create 478 mock MediaPipe face landmarks with realistic baseline resting geometry."""
    pts = [MockLandmarkPoint(0.5, 0.5, 0.0) for _ in range(478)]

    # Outer eyes (d_eyes ~ 0.40)
    pts[33] = MockLandmarkPoint(0.30, 0.35, 0.0)   # Right eye outer
    pts[263] = MockLandmarkPoint(0.70, 0.35, 0.0)  # Left eye outer

    # Inner eyes
    pts[133] = MockLandmarkPoint(0.42, 0.35, 0.0)  # Right eye inner
    pts[362] = MockLandmarkPoint(0.58, 0.35, 0.0)  # Left eye inner

    # Eye top and bottom (open eyes, EAR ~ 0.28)
    pts[159] = MockLandmarkPoint(0.36, 0.33, 0.0)
    pts[158] = MockLandmarkPoint(0.38, 0.33, 0.0)
    pts[145] = MockLandmarkPoint(0.38, 0.37, 0.0)
    pts[153] = MockLandmarkPoint(0.36, 0.37, 0.0)

    pts[385] = MockLandmarkPoint(0.64, 0.33, 0.0)
    pts[386] = MockLandmarkPoint(0.62, 0.33, 0.0)
    pts[374] = MockLandmarkPoint(0.62, 0.37, 0.0)
    pts[380] = MockLandmarkPoint(0.64, 0.37, 0.0)

    # Face height (10 to 152 ~ 0.60)
    pts[10] = MockLandmarkPoint(0.50, 0.15, 0.0)   # Forehead top
    pts[152] = MockLandmarkPoint(0.50, 0.75, 0.0)  # Chin bottom

    # Eyebrows baseline (d_brow_inner ~ 0.35 * d_eyes = 0.14)
    pts[107] = MockLandmarkPoint(0.43, 0.27, 0.0)  # Right inner brow
    pts[70] = MockLandmarkPoint(0.28, 0.27, 0.0)   # Right outer brow
    pts[336] = MockLandmarkPoint(0.57, 0.27, 0.0)  # Left inner brow
    pts[300] = MockLandmarkPoint(0.72, 0.27, 0.0)  # Left outer brow

    # Mouth baseline (width ~ 0.18, height ~ 0.02, corners level with center)
    pts[61] = MockLandmarkPoint(0.41, 0.55, 0.0)   # Right mouth corner
    pts[291] = MockLandmarkPoint(0.59, 0.55, 0.0)  # Left mouth corner
    pts[0] = MockLandmarkPoint(0.50, 0.54, 0.0)    # Upper lip top
    pts[13] = MockLandmarkPoint(0.50, 0.545, 0.0)  # Upper lip inner
    pts[14] = MockLandmarkPoint(0.50, 0.555, 0.0)  # Lower lip inner
    pts[17] = MockLandmarkPoint(0.50, 0.56, 0.0)   # Lower lip bottom

    return pts


class FaceExpressionRecognizerTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = FaceExpressionRecognizer(smoothing_window=3, auto_init_mesh=False)

    def tearDown(self):
        self.recognizer.close()

    def test_distance_and_ear_math(self):
        p1 = MockLandmarkPoint(0.0, 0.0)
        p2 = MockLandmarkPoint(3.0, 4.0)
        self.assertAlmostEqual(self.recognizer._distance(p1, p2), 5.0)

        pts = create_baseline_landmarks()
        left_ear = self.recognizer._eye_aspect_ratio(pts, 362, 385, 386, 263, 374, 380)
        right_ear = self.recognizer._eye_aspect_ratio(pts, 33, 159, 158, 133, 145, 153)
        self.assertGreater(left_ear, 0.20)
        self.assertGreater(right_ear, 0.20)

    def test_neutral_baseline_expression(self):
        pts = create_baseline_landmarks()
        for _ in range(3):
            result = self.recognizer.classify_landmarks(pts)

        self.assertIsNotNone(result)
        self.assertEqual(result["emotion"], "平靜")
        self.assertIn("平靜", result["expression"])
        self.assertIn("平靜", result["scores"])
        self.assertGreaterEqual(result["confidence"], 0.50)

    def test_happy_expression_detection(self):
        """Test '喜' (Happy): mouth corners lifted up and wide smile."""
        pts = create_baseline_landmarks()
        # Widen mouth corners and lift up (AU12)
        pts[61] = MockLandmarkPoint(0.38, 0.52, 0.0)   # Lifted up and out
        pts[291] = MockLandmarkPoint(0.62, 0.52, 0.0)  # Lifted up and out
        # Keep mouth inner height small (not laughing)
        pts[13] = MockLandmarkPoint(0.50, 0.54, 0.0)
        pts[14] = MockLandmarkPoint(0.50, 0.55, 0.0)

        for _ in range(3):
            result = self.recognizer.classify_landmarks(pts)

        self.assertEqual(result["emotion"], "喜")
        self.assertTrue(result["is_smiling"])
        self.assertIn("喜 (Happy", result["expression"])

    def test_angry_expression_detection(self):
        """Test '怒' (Angry): eyebrows pinched together and lowered."""
        pts = create_baseline_landmarks()
        # Eyebrows pulled together (d_brow_inner reduced)
        pts[107] = MockLandmarkPoint(0.46, 0.30, 0.0)
        pts[336] = MockLandmarkPoint(0.54, 0.30, 0.0)
        # Lowered closer to eyes
        pts[107].y = 0.31
        pts[336].y = 0.31
        pts[70] = MockLandmarkPoint(0.28, 0.26, 0.0)   # Outer brows slant up
        pts[300] = MockLandmarkPoint(0.72, 0.26, 0.0)

        # Mouth flat (no smile)
        pts[61] = MockLandmarkPoint(0.42, 0.56, 0.0)
        pts[291] = MockLandmarkPoint(0.58, 0.56, 0.0)

        for _ in range(3):
            result = self.recognizer.classify_landmarks(pts)

        self.assertEqual(result["emotion"], "怒")
        self.assertTrue(result["is_frowning"])
        self.assertIn("怒 (Angry", result["expression"])

    def test_sad_expression_detection(self):
        """Test '哀' (Sad): mouth corners droop downwards, inner eyebrows raised."""
        pts = create_baseline_landmarks()
        # Corners pulled down (AU15)
        pts[61] = MockLandmarkPoint(0.41, 0.58, 0.0)   # Drooped down
        pts[291] = MockLandmarkPoint(0.59, 0.58, 0.0)  # Drooped down
        pts[13] = MockLandmarkPoint(0.50, 0.54, 0.0)
        pts[14] = MockLandmarkPoint(0.50, 0.55, 0.0)

        # Inner eyebrows pulled up (八字眉 AU1)
        pts[107] = MockLandmarkPoint(0.43, 0.24, 0.0)  # Inner higher
        pts[336] = MockLandmarkPoint(0.57, 0.24, 0.0)  # Inner higher
        pts[70] = MockLandmarkPoint(0.28, 0.28, 0.0)   # Outer lower
        pts[300] = MockLandmarkPoint(0.72, 0.28, 0.0)

        for _ in range(3):
            result = self.recognizer.classify_landmarks(pts)

        self.assertEqual(result["emotion"], "哀")
        self.assertTrue(result["is_sad"])
        self.assertIn("哀 (Sad", result["expression"])

    def test_joy_expression_detection(self):
        """Test '樂' (Joy / Laughing): wide open mouth + wide smile."""
        pts = create_baseline_landmarks()
        # Wide smile
        pts[61] = MockLandmarkPoint(0.36, 0.52, 0.0)
        pts[291] = MockLandmarkPoint(0.64, 0.52, 0.0)
        # Big open mouth (AU25/26)
        pts[13] = MockLandmarkPoint(0.50, 0.51, 0.0)
        pts[14] = MockLandmarkPoint(0.50, 0.58, 0.0)

        for _ in range(3):
            result = self.recognizer.classify_landmarks(pts)

        self.assertEqual(result["emotion"], "樂")
        self.assertTrue(result["is_mouth_open"])
        self.assertIn("樂 (Joy", result["expression"])

    def test_blink_detection(self):
        """Test eye closure detection (Blink)."""
        pts = create_baseline_landmarks()
        # Close eyes (EAR < 0.19)
        pts[159].y = 0.35
        pts[158].y = 0.35
        pts[145].y = 0.35
        pts[153].y = 0.35

        pts[385].y = 0.35
        pts[386].y = 0.35
        pts[374].y = 0.35
        pts[380].y = 0.35

        for _ in range(3):
            result = self.recognizer.classify_landmarks(pts)

        self.assertIn("閉眼", result["expression"])


if __name__ == "__main__":
    unittest.main()
