import os
import json
from app.main import search_vsl_videos, extract_video_metadata, main

def test_video_search_limit():
    """Test that search returns exactly N videos"""
    results = search_vsl_videos("Ngôn ngữ kí hiệu Việt Nam", max_results=10)
    assert len(results) == 10
    assert isinstance(results[0], dict)

def test_metadata_fields():
    """Test metadata keys exist in one video"""
    videos = search_vsl_videos("Ngôn ngữ kí hiệu Việt Nam", max_results=1)
    metadata = extract_video_metadata(videos[0])
    required_keys = ["title", "channel", "video_id", "url", "publish_date", 
                     "duration", "view_count", "like_count", 
                     "description", "keywords", "captions_available"]
    for key in required_keys:
        assert key in metadata

def test_json_output():
    """Test if JSON file is created and readable"""
    filepath = "vietnamese_sign_language_videos.json"

    # Run scraper
    main()

    assert os.path.exists(filepath)

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert "title" in data[0]
