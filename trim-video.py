import os
from moviepy.video.io.VideoFileClip import VideoFileClip

# Manually specify the path to the ffmpeg executable
ffmpeg_path = '/opt/homebrew/bin/ffmpeg'
os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

# Load the video file
video = VideoFileClip("video2.mp4")

# Trim the video from 1 minute to 1 minute and 25 seconds
trimmed_video = video.subclip(64, 85)  # Start at 60s, end at 85s

# Save the trimmed video
trimmed_video.write_videofile("trimmed_video2.mp4")
