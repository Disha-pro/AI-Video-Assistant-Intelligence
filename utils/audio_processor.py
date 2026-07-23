import os
import yt_dlp
from pydub import AudioSegment

# ==========================================
# Configuration
# ==========================================

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ==========================================
# Download YouTube Audio
# ==========================================

def download_youtube_audio(url: str) -> str:
    """
    Download audio from YouTube and convert to WAV.
    Returns path of downloaded WAV file.
    """

    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,

        "quiet": False,
        "noplaylist": True,

        # Retry settings
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 10,

        # Optional (Uncomment if YouTube blocks downloads)
        # "cookiesfrombrowser": ("chrome",),
        # "cookiesfrombrowser": ("edge",),

        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            filename = ydl.prepare_filename(info)
            wav_path = os.path.splitext(filename)[0] + ".wav"

        print("✅ Download complete.")
        return wav_path

    except Exception as e:
        raise RuntimeError(f"YouTube download failed:\n{e}")


# ==========================================
# Convert Local Audio/Video to WAV
# ==========================================

def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio/video file into
    16kHz mono WAV.
    """

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    print("🔄 Converting to WAV...")

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    audio.export(output_path, format="wav")

    print("✅ Conversion complete.")

    return output_path


# ==========================================
# Split Audio into Chunks
# ==========================================

def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list[str]:
    """
    Split WAV into fixed-size chunks.
    """

    audio = AudioSegment.from_wav(wav_path)

    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]

        chunk_path = os.path.splitext(wav_path)[0] + f"_chunk_{i}.wav"

        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

    return chunks


# ==========================================
# Main Processing Function
# ==========================================

def process_input(source: str) -> list[str]:
    """
    Accepts:
        • YouTube URL
        • Local audio/video file

    Returns:
        List of WAV chunks.
    """

    if source.startswith(("http://", "https://")):

        print("=" * 60)
        print("🎥 YouTube URL detected")
        print("=" * 60)

        wav_path = download_youtube_audio(source)

    else:

        print("=" * 60)
        print("📂 Local File detected")
        print("=" * 60)

        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found:\n{source}")

        wav_path = convert_to_wav(source)

    print("=" * 60)
    print("✂ Chunking audio...")
    print("=" * 60)

    chunks = chunk_audio(wav_path)

    print(f"✅ Created {len(chunks)} chunk(s).")

    return chunks