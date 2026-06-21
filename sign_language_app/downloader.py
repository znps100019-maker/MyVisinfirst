import argparse
import os
import sys
import json
import re
import ssl
import shutil
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        encoded = text.encode(encoding, errors='replace')
        print(encoded.decode(encoding, errors='replace'))

# ==================== IDL Corpus Downloader Mode ====================

def download_single_idl_video(word_info, raw_dir):
    name = word_info.get("name")
    if not name:
        return None, "Invalid entry"

    word_dir = raw_dir / name
    word_dir.mkdir(parents=True, exist_ok=True)
    dest_path = word_dir / f"{name}.mp4"

    encoded_word = urllib.parse.quote(name)
    url = f"https://idl.nutc.edu.tw/words/{encoded_word}.mp4"
    context = ssl._create_unverified_context()

    if dest_path.exists() and dest_path.stat().st_size > 10240:
        return name, "Skipped (already exists)"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=context, timeout=15) as response:
            with open(dest_path, "wb") as out_file:
                out_file.write(response.read())
        return name, "Success"
    except Exception as e:
        return name, f"Failed: {e}"

def run_idl_downloader(args, raw_dir):
    api_url = "https://idl.nutc.edu.tw/getwords"
    context = ssl._create_unverified_context()

    safe_print(f"Fetching word list from API: {api_url}")
    try:
        with urllib.request.urlopen(api_url, context=context) as response:
            data = json.loads(response.read().decode('utf-8'))
    except Exception as e:
        safe_print(f"Error fetching word list: {e}")
        sys.exit(1)

    words = data.get("words", [])
    safe_print(f"Total words available: {len(words)}")

    if args.limit > 0:
        words = words[:args.limit]
        safe_print(f"Limiting to first {args.limit} words.")

    success_count = 0
    skip_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(download_single_idl_video, word, raw_dir): word for word in words}
        total = len(futures)
        completed = 0

        for future in as_completed(futures):
            completed += 1
            name, status = future.result()
            if status == "Success":
                success_count += 1
            elif status.startswith("Skipped"):
                skip_count += 1
            else:
                fail_count += 1
            
            if completed == 1 or completed == total or completed % 10 == 0 or (not status.startswith("Skipped") and "Success" not in status):
                safe_print(f"[{completed}/{total}] {name} -> {status}")

    safe_print("\nDownload summary:")
    safe_print(f"  Successfully downloaded: {success_count}")
    safe_print(f"  Skipped (existing): {skip_count}")
    safe_print(f"  Failed: {fail_count}")

# ==================== YouTube Downloader Mode ====================

DEFAULT_PLAYLIST = (
    "https://www.youtube.com/watch?v=nE4kuhO0l3E"
    "&list=PLzI2EvXfsJoOJFf3f1aqjQj7LIIrWuKYu"
)

try:
    from labels import label_keywords
except ImportError:
    try:
        from sign_language_app.labels import label_keywords
    except ImportError:
        def label_keywords():
            return {}

def normalize_text(value):
    return re.sub(r"\s+", " ", value or "").strip().lower()

def _keyword_matches(title, keyword):
    if keyword.isascii() and re.match(r"^[a-z0-9 _-]+$", keyword):
        return re.search(rf"\b{re.escape(keyword)}\b", title) is not None
    return keyword in title

def get_category_from_title(title):
    title_lower = normalize_text(title)
    candidates = []
    for label, keywords in label_keywords().items():
        for keyword in keywords:
            normalized = normalize_text(keyword)
            if normalized:
                candidates.append((label, normalized))

    for label, keyword in sorted(candidates, key=lambda item: len(item[1]), reverse=True):
        if _keyword_matches(title_lower, keyword):
            return label
    return "uncategorized"

def _find_downloaded_file(temp_dir, video_id, title):
    candidates = list(temp_dir.glob("*"))
    if video_id:
        for path in candidates:
            if video_id in path.name:
                return path

    safe_title = re.sub(r'[\\/*?:"<>|]', "", title)[:120]
    for path in candidates:
        if safe_title and safe_title in path.name:
            return path
    return None

def run_youtube_downloader(args, raw_dir):
    try:
        import yt_dlp
    except ImportError:
        safe_print("Missing dependency: yt-dlp. Install it with: pip install yt-dlp")
        sys.exit(1)

    url = args.url if args.url else DEFAULT_PLAYLIST
    temp_dir = raw_dir.parent / "temp_downloads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    options = {
        "format": "worst[ext=mp4]/worst",
        "outtmpl": str(temp_dir / "%(title).160s [%(id)s].%(ext)s"),
        "ignoreerrors": True,
        "no_warnings": True,
        "quiet": False,
    }

    safe_print(f"Downloading YouTube videos from: {url}")
    with yt_dlp.YoutubeDL(options) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
        except Exception as exc:
            safe_print(f"Download failed: {exc}")
            return

    entries = info.get("entries", [info]) if info else []
    moved = 0

    for entry in entries:
        if not entry:
            continue

        title = entry.get("title") or "untitled"
        video_id = entry.get("id") or ""
        category = get_category_from_title(title)
        category_dir = raw_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        downloaded = _find_downloaded_file(temp_dir, video_id, title)
        if downloaded is None:
            safe_print(f"[skip] Could not locate downloaded file for: {title}")
            continue

        destination = category_dir / downloaded.name
        if destination.exists():
            destination = category_dir / f"{destination.stem}-copy{destination.suffix}"
        shutil.move(str(downloaded), str(destination))
        moved += 1
        safe_print(f"[{category}] {title}")

    try:
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
    except OSError:
        pass
        
    safe_print(f"Done. Moved {moved} YouTube video(s) into: {raw_dir}")

# ==================== Main Entry Point ====================

def main():
    parser = argparse.ArgumentParser(description="Download sign language training videos.")
    parser.add_argument(
        "--mode",
        default="idl",
        choices=["idl", "youtube"],
        help="Download mode: 'idl' (Taiwan Sign Language Corpus) or 'youtube' (YouTube playlist).",
    )
    parser.add_argument("--url", help="YouTube video or playlist URL (forces youtube mode).")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of IDL videos to download.")
    parser.add_argument("--threads", type=int, default=20, help="Number of concurrent threads for IDL downloads.")
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parent,
        help="Base output directory. raw_videos/ will be created here.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.output_dir) / "raw_videos"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Force youtube mode if url is provided
    if args.url:
        args.mode = "youtube"

    if args.mode == "idl":
        run_idl_downloader(args, raw_dir)
    else:
        run_youtube_downloader(args, raw_dir)

if __name__ == "__main__":
    main()
