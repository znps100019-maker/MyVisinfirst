"""
AI 台灣手語即時辨識系統 - 本地網頁展示伺服器
執行本腳本將自動開啟瀏覽器體驗手語即時辨識網頁系統。
"""
import http.server
import os
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


class QuietHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files with custom mime types and reduced logging noise."""

    def end_headers(self):
        # 允許跨來源資源共享 (CORS)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

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
