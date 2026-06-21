import os
import sys
import re

# 嘗試載入 yt-dlp，若尚未安裝會提示使用者
try:
    import yt_dlp
except ImportError:
    print("錯誤：找不到 yt-dlp 模組。請確認是否已安裝套件（可透過 pip install yt-dlp 安裝）。")
    sys.exit(1)

# 手語詞彙分類關鍵字對照表 (用於自動將下載的影片分類到正確的資料夾)
CATEGORIES = {
    "hello": ["你好", "hello", "greet", "嗨"],
    "thank_you": ["謝謝", "thanks", "thank you", "感恩"],
    "goodbye": ["再見", "goodbye", "bye"],
    "sorry": ["對不起", "sorry", "抱歉"],
    "i": [" 我 ", "我", " me ", " i "],
    "you": [" 你 ", "你", "you"],
    "he_she": [" 他 ", "她", "他", "he", "she", "him", "her"]
}

# 預設手語教學 YouTube 播放清單（基礎手語單元1 嗨你好）
DEFAULT_PLAYLIST = "https://www.youtube.com/watch?v=nE4kuhO0l3E&list=PLzI2EvXfsJoOJFf3f1aqjQj7LIIrWuKYu"

def get_category_from_title(title):
    """根據影片標題，自動判斷手語語意分類標籤"""
    title_lower = title.lower()
    for label, keywords in CATEGORIES.items():
        for kw in keywords:
            # 如果關鍵字是前後有空格的英文，使用正則匹配以防誤判單字中的字母
            if kw.startswith(" ") or kw.endswith(" "):
                if re.search(r'\b' + re.escape(kw.strip()) + r'\b', title_lower):
                    return label
            elif kw in title_lower:
                return label
    return "uncategorized"

def download_video(url, base_output_dir):
    """下載單部影片或播放清單並進行分類"""
    print(f"正在分析連結: {url}")
    
    # 建立暫時存放原始下載檔案的目錄
    temp_dir = os.path.join(base_output_dir, "temp_downloads")
    os.makedirs(temp_dir, exist_ok=True)
    
    # yt-dlp 設定：下載最低畫質的直接 mp4 檔案 (極小、極快，且不需要額外安裝 ffmpeg 進行影音合併)
    ydl_opts = {
        'format': 'worst[ext=mp4]/worst', 
        'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': False
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # 先擷取資訊
        try:
            info = ydl.extract_info(url, download=True)
        except Exception as e:
            print(f"下載時發生錯誤: {e}")
            return
        
        # 整理並分類下載的檔案
        if info is None:
            print("無法解析連結資訊。")
            return
            
        entries = info.get('entries', [info]) # 如果是播放清單，會有多個 entries
        
        for entry in entries:
            if not entry:
                continue
            title = entry.get('title')
            ext = entry.get('ext', 'mp4')
            
            if not title:
                continue
                
            # 清理檔名中可能存在的不安全字元
            safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
            temp_file_path = os.path.join(temp_dir, f"{safe_title}.{ext}")
            
            if os.path.exists(temp_file_path):
                # 判定分類
                category = get_category_from_title(title)
                dest_dir = os.path.join(base_output_dir, "raw_videos", category)
                os.makedirs(dest_dir, exist_ok=True)
                
                # 移動到分類資料夾
                dest_file_path = os.path.join(dest_dir, f"{safe_title}.{ext}")
                try:
                    os.rename(temp_file_path, dest_file_path)
                    print(f"成功下載並歸類: [{category}] {safe_title}.{ext}")
                except Exception as e:
                    print(f"移動檔案時發生錯誤: {e}")
            else:
                print(f"找不到下載的暫存檔案: {temp_file_path}")
                
    # 清除暫存目錄
    try:
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except:
        pass

def main():
    import argparse
    parser = argparse.ArgumentParser(description="下載手語 YouTube 影片並分類。")
    parser.add_argument("--url", default=DEFAULT_PLAYLIST, help="YouTube 影片或播放清單網址。")
    parser.add_argument("--output-dir", default=".", help="專案儲存根目錄。")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.output_dir)
    download_video(args.url, base_dir)
    print("\n下載與分類程序執行完畢。")

if __name__ == "__main__":
    main()
