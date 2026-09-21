import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import main


class MainCliTests(unittest.TestCase):
    def test_no_source_defaults_to_camera(self):
        with patch.object(sys, "argv", ["main.py"]):
            args = main.build_args()

        self.assertEqual(args.video, "")
        self.assertEqual(args.camera, 0)
        self.assertEqual(args.stream_mode, "download")
        self.assertEqual(args.window_width, 1280)
        self.assertEqual(args.window_height, 720)
        self.assertTrue(args.combine_two_hands)
        self.assertTrue(args.use_knn)

    def test_positional_source_is_supported(self):
        with patch.object(sys, "argv", ["main.py", "sample.mp4"]):
            args = main.build_args()

        self.assertEqual(args.video, "sample.mp4")

    def test_video_alias_and_stream_mode_are_supported(self):
        with patch.object(
            sys,
            "argv",
            ["main.py", "--input", "https://example.test/video", "--stream-mode", "stream"],
        ):
            args = main.build_args()

        self.assertEqual(args.video, "https://example.test/video")
        self.assertEqual(args.stream_mode, "stream")

    def test_separate_hands_can_disable_default_combine_mode(self):
        with patch.object(sys, "argv", ["main.py", "--separate-hands"]):
            args = main.build_args()

        self.assertFalse(args.combine_two_hands)

    def test_no_knn_can_disable_default_knn_mode(self):
        with patch.object(sys, "argv", ["main.py", "--no-knn"]):
            args = main.build_args()

        self.assertFalse(args.use_knn)

    def test_empty_interactive_input_keeps_camera_mode(self):
        with patch.object(sys, "argv", ["main.py"]):
            args = main.build_args()
        with patch.object(sys.stdin, "isatty", return_value=True):
            main.resolve_interactive_source(args, input_func=lambda _: "")

        self.assertEqual(args.video, "")

    def test_interactive_video_input_selects_video_mode(self):
        with patch.object(sys, "argv", ["main.py"]):
            args = main.build_args()
        with patch.object(sys.stdin, "isatty", return_value=True):
            main.resolve_interactive_source(args, input_func=lambda _: "sample.mp4")

        self.assertEqual(args.video, "sample.mp4")

    def test_empty_video_source_is_rejected_before_opening_capture(self):
        empty_capture = SimpleNamespace(isOpened=lambda: False)
        with patch.object(main.cv2, "VideoCapture", return_value=empty_capture) as capture:
            result = main.open_video_source(SimpleNamespace(video="   ", camera=0))

        self.assertFalse(result.isOpened())
        capture.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
