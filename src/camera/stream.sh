#!/bin/bash

VIDEO_PATH="../sample.mp4"
RTSP_URL="rtsp://localhost:8554/mystream"

ffmpeg -re -stream_loop -1 -i "$VIDEO_PATH" -vcodec copy -f rtsp "$RTSP_URL"
