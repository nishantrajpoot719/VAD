# download_ffmpeg.py
import os
import zipfile
import requests
import io

FFMPEG_DIR = "ffmpeg_binary"
FFMPEG_EXE_PATH = os.path.join(FFMPEG_DIR, "ffmpeg.exe")

def download_and_extract_ffmpeg():
    if os.path.exists(FFMPEG_EXE_PATH):
        print("✅ FFmpeg already exists.")
        return

    print("📥 Downloading FFmpeg ZIP (Windows static build)...")
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))

    print("📦 Extracting only ffmpeg.exe and ffprobe.exe...")
    for f in z.namelist():
        if f.endswith("bin/ffmpeg.exe") or f.endswith("bin/ffprobe.exe"):
            target_path = os.path.join(FFMPEG_DIR, os.path.basename(f))
            os.makedirs(FFMPEG_DIR, exist_ok=True)
            with open(target_path, "wb") as out_file:
                out_file.write(z.read(f))

    print("✅ FFmpeg binaries extracted to ffmpeg_binary/")

if __name__ == "__main__":
    download_and_extract_ffmpeg()
