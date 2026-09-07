import math
import unittest
from collections import deque
from types import SimpleNamespace

from core.evaluation import evaluate_predictions
from core.detectors.hand_detector import HandSignRecognizer
from main import validate_processed_frames
from sign_language_app.labels import classification_label


def point(x, y, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def make_hand(extended=(), frame_shape=(480, 640), pinch=False):
    """Create deterministic, human-labelled landmark fixtures."""
    height, width = frame_shape
    pixels = [point(320, 400) for _ in range(21)]
    pixels[1], pixels[2], pixels[3] = point(300, 360), point(280, 340), point(250, 300)
    pixels[4] = point(220, 250) if "thumb" in extended else point(320, 350)

    finger_data = {
        "index": (5, 300, 320),
        "middle": (9, 325, 300),
        "ring": (13, 350, 310),
        "pinky": (17, 375, 325),
    }
    for name, (base, x, y) in finger_data.items():
        if name in extended:
            pixels[base] = point(x, y)
            pixels[base + 1] = point(x, y - 50)
            pixels[base + 2] = point(x, y - 100)
            pixels[base + 3] = point(x, y - 150)
        else:
            pixels[base] = point(x, y)
            pixels[base + 1] = point(x, y + 25)
            pixels[base + 2] = point(x + 20, y + 45)
            pixels[base + 3] = point(x + 5, y + 35)

    if pinch:
        pixels[4] = point(300, 290)
        pixels[8] = point(302, 292)

    normalized = [point(item.x / width, item.y / height, item.z / width) for item in pixels]
    return SimpleNamespace(landmark=normalized)


def bare_recognizer(history_size=8, stable_min_count=3, **kwargs):
    recognizer = HandSignRecognizer.__new__(HandSignRecognizer)
    recognizer.history = deque(maxlen=history_size)
    recognizer.stable_min_count = stable_min_count
    recognizer.combine_two_hands = kwargs.get("combine_two_hands", False)
    recognizer.no_hand_reset_frames = kwargs.get("no_hand_reset_frames", 3)
    recognizer.current_candidate = "No hand"
    recognizer.current_hand_count = 0
    recognizer.no_hand_streak = 0
    recognizer.sentence = []
    recognizer.last_added_sign = None
    return recognizer


class ClassificationRegressionTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = bare_recognizer()
        self.frame_shape = (480, 640)

    def classify(self, flags, pinch=False):
        hand = make_hand(flags, self.frame_shape, pinch=pinch)
        return self.recognizer._classify_sign(
            {name: name in flags for name in ("thumb", "index", "middle", "ring", "pinky")},
            hand,
            self.frame_shape,
        )

    def test_fist_with_thumb_index_close_is_not_ok(self):
        self.assertEqual(self.classify((), pinch=True), "Fist (Solidarity)")

    def test_ok_requires_three_other_extended_fingers(self):
        self.assertEqual(
            self.classify(("middle", "ring", "pinky"), pinch=True),
            "OK (Zero / Can)",
        )

    def test_number_seven_and_eight_follow_project_definition(self):
        self.assertEqual(self.classify(("thumb", "index")), "Number 7 (Gun)")
        self.assertEqual(self.classify(("thumb", "index", "middle")), "Number 8")

    def test_thumb_state_is_not_reinferred_from_distance(self):
        self.assertEqual(self.classify(("index", "middle")), "Number 2 (Victory)")
        self.assertEqual(self.classify(("index", "middle", "ring")), "Number 3")
        self.assertEqual(self.classify(("index", "middle", "ring", "pinky")), "Number 4 (Salute)")

    def test_bent_finger_is_not_reported_as_extended(self):
        straight = make_hand(("index",), self.frame_shape)
        bent = make_hand((), self.frame_shape)
        self.assertTrue(self.recognizer._extended_fingers(straight, "Right", self.frame_shape)["index"])
        self.assertFalse(self.recognizer._extended_fingers(bent, "Right", self.frame_shape)["index"])

    def test_geometry_is_invariant_to_frame_aspect_ratio(self):
        wide_shape = (480, 960)
        tall_shape = (960, 480)
        wide = make_hand(("index", "middle"), wide_shape)
        tall = make_hand(("index", "middle"), tall_shape)
        self.assertEqual(
            self.recognizer._extended_fingers(wide, "Right", wide_shape),
            self.recognizer._extended_fingers(tall, "Right", tall_shape),
        )


class StateRegressionTests(unittest.TestCase):
    def test_one_frame_has_100_percent_consistency_but_is_not_confirmed(self):
        recognizer = bare_recognizer()
        recognizer._record_candidate("Number 2 (Victory)", 1)
        status = recognizer.stable_status
        self.assertEqual(status["consistency"], 1.0)
        self.assertFalse(status["is_stable"])

    def test_switch_does_not_keep_old_stable_target(self):
        recognizer = bare_recognizer()
        for _ in range(3):
            recognizer._record_candidate("Number 2 (Victory)", 1)
        self.assertTrue(recognizer.is_target_detected("2"))
        recognizer._record_candidate("Number 3", 1)
        self.assertEqual(recognizer.stable_status["sign"], "Number 3")
        self.assertFalse(recognizer.stable_status["is_stable"])
        self.assertFalse(recognizer.is_target_detected("2"))

    def test_first_no_hand_frame_clears_target_immediately(self):
        recognizer = bare_recognizer()
        for _ in range(3):
            recognizer._record_candidate("Number 2 (Victory)", 1)
        self.assertTrue(recognizer.is_target_detected("2"))
        recognizer._record_candidate("No hand", 0)
        self.assertEqual(recognizer.stable_status["sign"], "No hand")
        self.assertFalse(recognizer.is_target_detected("2"))

    def test_long_no_hand_allows_same_sign_to_be_recorded_again(self):
        recognizer = bare_recognizer()
        for _ in range(3):
            recognizer._record_candidate("Number 2 (Victory)", 1)
        for _ in range(3):
            recognizer._record_candidate("No hand", 0)
        for _ in range(3):
            recognizer._record_candidate("Number 2 (Victory)", 1)
        self.assertEqual(recognizer.sentence, ["Number 2 (Victory)", "Number 2 (Victory)"])

    def test_short_tracking_drop_does_not_duplicate_same_sign(self):
        recognizer = bare_recognizer()
        for _ in range(3):
            recognizer._record_candidate("Number 2 (Victory)", 1)
        recognizer._record_candidate("No hand", 0)
        for _ in range(3):
            recognizer._record_candidate("Number 2 (Victory)", 1)
        self.assertEqual(recognizer.sentence, ["Number 2 (Victory)"])

    def test_two_hands_are_separate_by_default(self):
        recognizer = bare_recognizer()
        detections = [
            {"sign": "Number 5 (Hello / Greet)", "fingers": {"thumb": True}},
            {"sign": "Number 1 (Secret)", "fingers": {"index": True}},
        ]
        self.assertEqual(recognizer._frame_candidate(detections), "Multiple hands")
        recognizer.combine_two_hands = True
        detections[0]["fingers"] = {
            "thumb": True, "index": True, "middle": True, "ring": True, "pinky": True,
        }
        self.assertEqual(recognizer._frame_candidate(detections), "Number 6 (Two hands)")


class EvaluationRegressionTests(unittest.TestCase):
    def test_accuracy_requires_ground_truth_and_counts_unknown_as_error(self):
        result = evaluate_predictions(["OK (Zero / Can)", "OK (Zero / Can)"], ["Fist (Solidarity)", "Fist (Solidarity)"])
        self.assertTrue(result["has_ground_truth"])
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(len(result["errors"]), 2)

    def test_without_labels_reports_coverage_not_accuracy(self):
        result = evaluate_predictions(["Unknown", "Number 2 (Victory)"])
        self.assertFalse(result["has_ground_truth"])
        self.assertNotIn("accuracy", result)
        self.assertEqual(result["coverage"], 0.5)

    def test_empty_run_is_invalid(self):
        self.assertFalse(evaluate_predictions([])["valid"])

    def test_zero_frame_video_is_a_failed_run(self):
        with self.assertRaises(RuntimeError):
            validate_processed_frames(0, True)
        validate_processed_frames(0, False)

    def test_custom_database_label_is_not_presented_as_a_built_in_shape(self):
        self.assertTrue(classification_label("hello", include_english=False).startswith("資料庫標籤:"))


if __name__ == "__main__":
    unittest.main()
