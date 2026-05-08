import argparse
import csv
import multiprocessing as mp
import os
import random
import re
import subprocess
import time
import json
import uuid
from urllib.parse import urlparse, parse_qs

import requests
from tqdm import tqdm
import yt_dlp

"""
Download audio from YouTube, Bilibili, or direct MP3 links,
then save the result as FLAC.

This version processes up to the first 20 CSV rows and skips files
that have already been downloaded.

Example:
python download_20_only.py --csv_file test_B.csv --output_dir downloads --workers 4
"""

REGEX_PLAY_INFO = r'<script>window\.__playinfo__=(.*?)</script>'
REGEX_INITIAL_STATE = r'__INITIAL_STATE__=(.*?);\(function\(\)'
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/90.0.4430.212 Safari/537.36"
)
BILIBILI_URL = "https://www.bilibili.com"


def setup_argparse():
    parser = argparse.ArgumentParser(
    )
    parser.add_argument("--csv_file", required=True, help="Path to the CSV file containing download information.")
    parser.add_argument("--output_dir", required=True, help="Directory to save downloaded files.")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes (default: 4)")
    return parser.parse_args()


def read_csv(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename).strip()


def get_unique_filename(directory, extension):
    return os.path.join(directory, f"temp_{uuid.uuid4().hex}.{extension}")


def extract_youtube_video_id(url):
    parsed = urlparse(url)

    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        return parsed.path.lstrip("/")[:11]

    if parsed.netloc in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [""])[0][:11]
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/shorts/")[-1].split("/")[0][:11]
        if parsed.path.startswith("/embed/"):
            return parsed.path.split("/embed/")[-1].split("/")[0][:11]

    return ""


def get_play_url_web_page(url):
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": BILIBILI_URL,
        "accept-language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        "accept-encoding": "gzip, deflate, br",
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    html = response.text

    match_initial_state = re.search(REGEX_INITIAL_STATE, html)
    match_play_info = re.search(REGEX_PLAY_INFO, html)

    if not match_initial_state or not match_play_info:
        return None, None

    initial_state_json = json.loads(match_initial_state.group(1))
    play_info_json = json.loads(match_play_info.group(1))
    return initial_state_json, play_info_json


def parse_audio(play_info):
    audio_list = play_info["data"]["dash"]["audio"]
    return max(audio_list, key=lambda x: x["bandwidth"])


def download_file(url, headers, filename):
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(filename, "wb") as file, tqdm(
        desc=os.path.basename(filename),
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        leave=False,
    ) as bar:
        for data in response.iter_content(chunk_size=1024 * 64):
            if not data:
                continue
            size = file.write(data)
            bar.update(size)


def download_bilibili(bilibili_url, output_dir):
    initial_state, play_info = get_play_url_web_page(bilibili_url)
    if not initial_state or not play_info:
        raise RuntimeError("Failed to retrieve Bilibili video information.")

    video_data = initial_state["videoData"]
    bvid = video_data["bvid"]
    title = sanitize_filename(video_data["title"])
    unique_name = f"[{bvid}]{title}"

    best_audio = parse_audio(play_info)
    audio_url = best_audio["baseUrl"]

    headers = {
        "User-Agent": USER_AGENT,
        "Referer": BILIBILI_URL,
    }

    os.makedirs(output_dir, exist_ok=True)
    audio_filename = os.path.join(output_dir, f"{unique_name}_audio.m4a")
    download_file(audio_url, headers, audio_filename)
    return audio_filename


def download_youtube(url, output_path_without_ext):
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract YouTube video ID from URL: {url}")

    url_constructed = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "flac",
        }],
        "outtmpl": output_path_without_ext,
        "socket_timeout": 300,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url_constructed])

    return output_path_without_ext + ".flac"


def download_mp3(url, output_dir):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    mp3_path = get_unique_filename(output_dir, "mp3")
    with open(mp3_path, "wb") as f:
        f.write(response.content)
    return mp3_path


def convert_to_flac(input_path, output_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-c:a", "flac", output_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def process_row(row, output_dir):
    url = row["Url"]
    title = sanitize_filename(row["Title"])
    label = row["Bonafide Or Spoof"].strip().lower()
    split_set = row["Set"].strip().lower()

    output_filename = f"{split_set}_{label}_{title}.flac"
    output_path = os.path.join(output_dir, output_filename)
    output_base = os.path.splitext(output_path)[0]

    if os.path.exists(output_path):
        return "Skipped", None

    temp_file = None
    try:
        if "youtube.com" in url or "youtu.be" in url:
            final_path = download_youtube(url, output_base)
            if final_path != output_path and os.path.exists(final_path):
                os.replace(final_path, output_path)

        elif "bilibili.com" in url:
            temp_file = download_bilibili(url, output_dir)
            convert_to_flac(temp_file, output_path)

        elif urlparse(url).path.lower().endswith(".mp3"):
            temp_file = download_mp3(url, output_dir)
            convert_to_flac(temp_file, output_path)

        else:
            raise ValueError(f"Unsupported URL: {url}")

        return "Success", None

    except Exception as e:
        return "Failure", str(e)

    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


def worker(queue, output_dir, log_queue, progress_queue):
    while True:
        item = queue.get()
        if item is None:
            break

        index, row = item
        status, error = process_row(row, output_dir)
        log_queue.put((index, row.get("Url", ""), status, error))
        progress_queue.put(1)

        time.sleep(random.uniform(1, 2.5))


def logger(log_queue, log_file):
    with open(log_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "url", "status", "error"])

        while True:
            item = log_queue.get()
            if item is None:
                break

            index, url, status, error = item
            writer.writerow([index, url, status, error or ""])
            f.flush()


def progress_tracker(progress_queue, total_rows):
    pbar = tqdm(total=total_rows, desc=f"Downloading")
    completed = 0

    while completed < total_rows:
        increment = progress_queue.get()
        if increment is None:
            break
        completed += increment
        pbar.update(increment)

    pbar.close()


def main():
    args = setup_argparse()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = read_csv(args.csv_file)
    total_rows = len(rows)

    if total_rows == 0:
        print("No rows found in CSV.")
        return

    task_queue = mp.Queue()
    log_queue = mp.Queue()
    progress_queue = mp.Queue()

    for i, row in enumerate(rows):
        task_queue.put((i, row))

    log_file = os.path.join(args.output_dir, "download_log.csv")

    log_process = mp.Process(target=logger, args=(log_queue, log_file))
    log_process.start()

    progress_process = mp.Process(target=progress_tracker, args=(progress_queue, total_rows))
    progress_process.start()

    worker_count = min(args.workers, total_rows)
    workers = []

    for _ in range(worker_count):
        p = mp.Process(target=worker, args=(task_queue, args.output_dir, log_queue, progress_queue))
        workers.append(p)
        p.start()

    for _ in range(worker_count):
        task_queue.put(None)

    for w in workers:
        w.join()

    log_queue.put(None)
    log_process.join()

    progress_queue.put(None)
    progress_process.join()

    print(f"Done. Checked first {total_rows} row(s).")
    print(f"Saved files to: {args.output_dir}")
    print(f"Log file: {log_file}")


if __name__ == "__main__":
    mp.freeze_support()
    main()