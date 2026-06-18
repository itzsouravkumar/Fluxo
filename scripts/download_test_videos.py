#!/usr/bin/env python3
"""Download test traffic videos for FLUXO vision pipeline testing.

Usage:
    python3 scripts/download_test_videos.py
"""

import subprocess
import sys
from pathlib import Path


TEST_VIDEOS = [
    {
        "name": "bengaluru_traffic_v1",
        "url": "https://www.youtube.com/watch?v=signal_jump_bengaluru",
        "description": "Bengaluru traffic junction (placeholder - replace with real URLs)",
    },
]


def check_yt_dlp():
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main():
    output_dir = Path("data/test_videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("FLUXO Test Video Downloader")
    print("=" * 40)

    if not check_yt_dlp():
        print("yt-dlp not found. Install it:")
        print("  pip install yt-dlp")
        print("")
        print("Or manually download Bengaluru traffic videos from YouTube:")
        print("  Search: 'Bengaluru Veerannapalya traffic CCTV'")
        print("  Search: 'Bengaluru silk board traffic'")
        print("  Search: 'Indian traffic junction drone'")
        print("")
        print(f"Save videos to: {output_dir}/")
        return

    print("yt-dlp found. Ready to download.")
    print("")
    print("Manual download recommended for best results:")
    print("  1. Search YouTube for 'Bengaluru traffic junction CCTV'")
    print("  2. Download 3-4 clips (30s-2min each)")
    print(f"  3. Save to: {output_dir}/")
    print("")
    print("Example command:")
    print(f'  yt-dlp -o "{output_dir}/%(title)s.%(ext)s" URL_HERE')


if __name__ == "__main__":
    main()
