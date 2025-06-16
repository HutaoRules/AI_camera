from deploy import main

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python deploy.py <video_path/rtsp_url> <cam_id>")
    else:
        video_path = sys.argv[1]
        cam_id = sys.argv[2]
        main(video_path, cam_id)