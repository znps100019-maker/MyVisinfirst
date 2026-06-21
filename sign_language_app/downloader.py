import argparse
import os
import re
import shutil
import sys
from pathlib import Path

try:
    from labels import label_keywords
except ImportError:
    from sign_language_app.labels import label_keywords


DEFAULT_PLAYLIST = (
    "https://www.youtube.com/watch?v=nE4kuhO0l3E"
    "&list=PLzI2EvXfsJoOJFf3f1aqjQj7LIIrWuKYu"
)


def normalize_text(value):
    return re.sub(r"\s+", " ", value or "").strip().lower()


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


def download_video(url, base_output_dir):
    try:
        import yt_dlp
    except ImportError:
        print("Missing dependency: yt-dlp. Install it with: pip install yt-dlp")
        sys.exit(1)

    base_output_dir = Path(base_output_dir)
    temp_dir = base_output_dir / "temp_downloads"
    raw_dir = base_output_dir / "raw_videos"
    temp_dir.mkdir(parents=True, exist_ok=True)

    options = {
        "format": "worst[ext=mp4]/worst",
        "outtmpl": str(temp_dir / "%(title).160s [%(id)s].%(ext)s"),
        "ignoreerrors": True,
        "no_warnings": True,
        "quiet": False,
    }

    print(f"Downloading sign-language videos from: {url}")
    with yt_dlp.YoutubeDL(options) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
        except Exception as exc:
            print(f"Download failed: {exc}")
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
            print(f"[skip] Could not locate downloaded file for: {title}")
            continue

        destination = category_dir / downloaded.name
        if destination.exists():
            destination = category_dir / f"{destination.stem}-copy{destination.suffix}"
        shutil.move(str(downloaded), str(destination))
        moved += 1
        print(f"[{category}] {title}")

    _cleanup_empty_dir(temp_dir)
    print(f"Done. Moved {moved} video(s) into: {raw_dir}")


def _keyword_matches(title, keyword):
    if keyword.isascii() and re.match(r"^[a-z0-9 _-]+$", keyword):
        return re.search(rf"\b{re.escape(keyword)}\b", title) is not None
    return keyword in title


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


def _cleanup_empty_dir(path):
    try:
        if path.exists() and not any(path.iterdir()):
            path.rmdir()
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Download and sort sign-language training videos.")
    parser.add_argument("--url", default=DEFAULT_PLAYLIST, help="YouTube video or playlist URL.")
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parent,
        help="Output directory. raw_videos/ will be created here.",
    )
    args = parser.parse_args()

    download_video(args.url, os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
