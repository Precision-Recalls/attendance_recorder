import yt_dlp
from yt_dlp.utils import download_range_func


def download_video(video_link):
    start_time = 0  # accepts decimal value like 2.3
    end_time = 50

    yt_opts = {
        'verbose': True,
        'download_ranges': download_range_func(None, [(start_time, end_time)]),
        'force_keyframes_at_cuts': True,
    }

    with yt_dlp.YoutubeDL(yt_opts) as ydl:
        ydl.download(video_link)
