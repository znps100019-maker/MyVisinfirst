"""
AI 台灣手語即時辨識系統 - 本地網頁展示伺服器
執行本腳本將自動開啟瀏覽器體驗手語即時辨識網頁系統。
"""
import hashlib
import http.server
import json
import os
import re
import shutil
import socketserver
import sys
import webbrowser

# 確保 Windows cp950 終端機不因 emoji 崩潰
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def ensure_model_synced():
    """Ensure web/model.json is up-to-date with sign_language_app/model.json."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    source_model = os.path.join(project_root, "sign_language_app", "model.json")
    target_model = os.path.join(project_root, "web", "model.json")

    if not os.path.exists(source_model):
        print(f"[警告] 找不到手語特徵模型來源：{source_model}")
        return

    needs_copy = False
    if not os.path.exists(target_model):
        needs_copy = True
    else:
        if os.path.getmtime(source_model) > os.path.getmtime(target_model):
            needs_copy = True

    if needs_copy:
        os.makedirs(os.path.dirname(target_model), exist_ok=True)
        print(f"[同步] 正在同步模型檔案至網頁目錄 (12MB)...", flush=True)
        shutil.copyfile(source_model, target_model)
        print(f"[完成] 模型已就緒：{target_model}", flush=True)


def resolve_or_download_video(url, project_root):
    """
    Resolve direct video or download YouTube video using yt-dlp.
    Returns a web-accessible relative URL path.
    """
    clean_url = url.split("?")[0].lower()
    if clean_url.endswith((".mp4", ".webm", ".ogg", ".mov")):
        return {"status": "ok", "url": url, "type": "direct"}

    # If it's a relative path to temp_downloads or web
    if url.startswith("/sign_language_app/") or url.startswith("/web/"):
        return {"status": "ok", "url": url, "type": "relative"}

    # Use yt-dlp to download and cache in sign_language_app/temp_downloads
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    download_dir = os.path.join(project_root, "sign_language_app", "temp_downloads")
    os.makedirs(download_dir, exist_ok=True)
    out_filename = f"web_video_{url_hash}.mp4"
    out_path = os.path.join(download_dir, out_filename)

    if not os.path.exists(out_path):
        try:
            import yt_dlp
            print(f"[影片下載] 正在從網址解析/下載影片: {url} ...", flush=True)
            ydl_opts = {
                "format": "bestvideo[height<=720][ext=mp4]/bestvideo[ext=mp4]/best[ext=mp4]/best",
                "outtmpl": out_path,
                "quiet": True,
                "no_warnings": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"[影片下載完成] 已暫存至: {out_path}", flush=True)
        except Exception as exc:
            print(f"[影片下載失敗] {exc}", flush=True)
            return {"status": "error", "message": f"無法下載或解析此影片連結：{exc}"}

    web_url = f"/sign_language_app/temp_downloads/{out_filename}"
    return {"status": "ok", "url": web_url, "type": "downloaded"}


class RangeFileWrapper:
    """Wrap a file object to read only a specific range of bytes."""

    def __init__(self, f, length):
        self.f = f
        self.remaining = length

    def read(self, size=-1):
        if self.remaining <= 0:
            return b""
        if size < 0 or size > self.remaining:
            size = self.remaining
        chunk = self.f.read(size)
        self.remaining -= len(chunk)
        return chunk

    def close(self):
        self.f.close()


class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files with custom mime types, range requests, and API endpoints."""

    def end_headers(self):
        # 允許跨來源資源共享 (CORS) 與 Byte Ranges
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/resolve_video":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len)
            try:
                data = json.loads(post_body.decode("utf-8"))
                video_url = data.get("url", "").strip()
                if not video_url:
                    self.send_json_response(400, {"status": "error", "message": "未提供影片網址"})
                    return
                project_root = os.path.dirname(os.path.abspath(__file__))
                res = resolve_or_download_video(video_url, project_root)
                self.send_json_response(200, res)
            except Exception as e:
                self.send_json_response(500, {"status": "error", "message": str(e)})
            return

        super().do_POST()

    def send_json_response(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_head(self):
        """Handle HTTP 206 Partial Content (Range requests) for smooth video scrubbing."""
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()

        range_header = self.headers.get("Range")
        if not range_header:
            return super().send_head()

        match = re.match(r"^bytes=(\d*)-(\d*)$", range_header.strip())
        if not match:
            return super().send_head()

        file_size = os.path.getsize(path)
        start_str, end_str = match.groups()
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1

        if start >= file_size or end >= file_size or start > end:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        content_length = end - start + 1
        content_type = self.guess_type(path)

        self.send_response(206, "Partial Content")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(content_length))
        self.end_headers()

        f = open(path, "rb")
        f.seek(start)
        return RangeFileWrapper(f, content_length)

    def log_message(self, format, *args):
        # 只顯示重大錯誤，保持終端機清爽
        if "404" in str(args) or "500" in str(args):
            super().log_message(format, *args)


def find_free_port(start_port=8080, max_attempts=10):
    import socket
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def main():
    ensure_model_synced()

    port = find_free_port(8080)
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    url = f"http://127.0.0.1:{port}/web/"

    print("=" * 60)
    print("  [Sign Language AI] 台灣手語即時辨識系統 - 專題展示伺服器")
    print("=" * 60)
    print(f"  本地網址: {url}")
    print(f"  模型支援: 519 種台灣手語詞彙 (3,700 筆特徵)")
    print(f"  技術架構: MediaPipe Hands (JS) + 126D 相對空間座標 + KNN")
    print("-" * 60)
    print("  使用說明：")
    print("  1. 瀏覽器若未自動彈出，請複製上方網址至瀏覽器貼上。")
    print("  2. 在網頁中點擊「啟動攝影機」並允許相機權限即可開始即時辨識。")
    print("  3. 若欲部署至 GitHub Pages：將 web 目錄部署至 gh-pages 即可！")
    print("  4. 隨時可按 Ctrl + C 停止伺服器。")
    print("=" * 60, flush=True)

    try:
        # 自動開啟預設瀏覽器
        webbrowser.open(url)
    except Exception as exc:
        print(f"無法自動開啟瀏覽器，請手動開啟網址：{url} ({exc})", flush=True)

    try:
        # Allow reuse address to avoid WinError 10048
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("", port), QuietHTTPRequestHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[伺服器已停止] 感謝使用專題展示系統！")
        sys.exit(0)


if __name__ == "__main__":
    main()
