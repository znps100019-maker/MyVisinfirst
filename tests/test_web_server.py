import unittest

import run_web


class WebServerValidationTests(unittest.TestCase):
    def test_empty_video_url_is_rejected_without_downloader(self):
        result = run_web.resolve_or_download_video("  ", ".")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "未提供影片網址")

    def test_direct_video_url_is_returned_without_downloader(self):
        result = run_web.resolve_or_download_video(
            "https://example.test/demo.MP4?token=abc", "."
        )

        self.assertEqual(result, {
            "status": "ok",
            "url": "https://example.test/demo.MP4?token=abc",
            "type": "direct",
        })


if __name__ == "__main__":
    unittest.main()
