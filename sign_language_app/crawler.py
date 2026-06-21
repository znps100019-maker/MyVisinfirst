import argparse
import os
import sys
import json
import ssl
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

def download_single_video(word_info, raw_dir):
    name = word_info.get("name")
    if not name:
        return None, "Invalid entry"

    # Create word-specific folder under raw_videos
    word_dir = raw_dir / name
    word_dir.mkdir(parents=True, exist_ok=True)
    dest_path = word_dir / f"{name}.mp4"

    encoded_word = urllib.parse.quote(name)
    url = f"https://idl.nutc.edu.tw/words/{encoded_word}.mp4"
    context = ssl._create_unverified_context()

    # Check if file already exists and is not empty
    if dest_path.exists() and dest_path.stat().st_size > 10240: # >10KB
        return name, "Skipped (already exists)"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=context, timeout=15) as response:
            with open(dest_path, "wb") as out_file:
                out_file.write(response.read())
        return name, "Success"
    except Exception as e:
        return name, f"Failed: {e}"

def main():
    parser = argparse.ArgumentParser(description="Download all sign language videos from the IDL Corpus.")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of videos to download (0 means all).")
    parser.add_argument("--threads", type=int, default=20, help="Number of concurrent download threads.")
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parent,
        help="Base output directory for raw_videos/.",
    )
    args = parser.parse_args()

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

    raw_dir = Path(args.output_dir) / "raw_videos"
    raw_dir.mkdir(parents=True, exist_ok=True)

    safe_print(f"Downloading videos to: {raw_dir}")
    success_count = 0
    skip_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(download_single_video, word, raw_dir): word for word in words}
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
            
            # Print progress every 10 completions or on error/status changes
            if completed == 1 or completed == total or completed % 10 == 0 or not status.startswith("Skipped") and "Success" not in status:
                safe_print(f"[{completed}/{total}] {name} -> {status}")

    safe_print("\nDownload summary:")
    safe_print(f"  Successfully downloaded: {success_count}")
    safe_print(f"  Skipped (existing): {skip_count}")
    safe_print(f"  Failed: {fail_count}")

if __name__ == "__main__":
    main()
