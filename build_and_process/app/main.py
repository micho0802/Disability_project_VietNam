from youtubesearchpython import VideosSearch
from pytube import YouTube
import json
import time

def search_vsl_videos(keyword="Ngôn ngữ kí hiệu Việt Nam", max_results=50):
    print(f"Searching for: {keyword}")
    videos = []
    search = VideosSearch(keyword, limit=20)
    total_fetched = 0

    while total_fetched < max_results:
        results = search.result()['result']
        for video in results:
            videos.append(video)
            total_fetched += 1
            if total_fetched >= max_results:
                break
        if total_fetched < max_results:
            search.next()
            time.sleep(1)

    return videos[:max_results]

def extract_video_metadata(video_info):
    try:
        yt = YouTube(video_info['link'])
        metadata = {
            "title": yt.title,
            "channel": yt.author,
            "video_id": yt.video_id,
            "url": yt.watch_url,
            "publish_date": yt.publish_date.isoformat() if yt.publish_date else None,
            "duration": yt.length,
            "view_count": yt.views,
            "like_count": yt.rating,  # may not be accurate
            "description": yt.description,
            "keywords": yt.keywords,
            "captions_available": bool(yt.captions),
        }
    except Exception as e:
        print(f"Error processing video: {video_info['title']}\n{e}")
        metadata = {
            "title": video_info.get("title"),
            "channel": video_info.get("channel", {}).get("name"),
            "video_id": video_info.get("id"),
            "url": video_info.get("link"),
            "publish_date": None,
            "duration": None,
            "view_count": None,
            "like_count": None,
            "description": None,
            "keywords": None,
            "captions_available": None,
        }
    return metadata

def main():
    vsl_videos = search_vsl_videos("Ngôn ngữ kí hiệu Việt Nam", max_results=50)
    all_metadata = []

    for video in vsl_videos:
        metadata = extract_video_metadata(video)
        all_metadata.append(metadata)

    # Save to JSON file
    with open("vietnamese_sign_language_videos.json", "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print("Metadata for 50 videos to vietnamese_sign_language_videos.json")

if __name__ == "__main__":
    main()
