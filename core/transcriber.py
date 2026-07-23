import os
import requests
import whisper
import torch
from pydub import AudioSegment

# ============================================================
# Configuration
# ============================================================

SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"

_model = None


# ============================================================
# Load Whisper Model
# ============================================================

def load_model():
    global _model

    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print("=" * 60)
        print(f"Loading Whisper '{WHISPER_MODEL}' on {device}...")
        print("=" * 60)

        _model = whisper.load_model(WHISPER_MODEL, device=device)

        print("✅ Whisper model loaded successfully.\n")

    return _model


# ============================================================
# Whisper Transcription
# ============================================================

def transcribe_chunk_whisper(chunk_path: str) -> str:

    model = load_model()

    result = model.transcribe(
        chunk_path,
        task="transcribe",
        fp16=torch.cuda.is_available(),
        verbose=False,
    )

    return result["text"].strip()


# ============================================================
# Sarvam API
# ============================================================

def _send_to_sarvam(piece_path: str) -> str:

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    with open(piece_path, "rb") as f:

        files = {
            "file": (
                os.path.basename(piece_path),
                f,
                "audio/wav"
            )
        }

        data = {
            "model": SARVAM_MODEL,
            "with_diarization": "false"
        }

        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(response.text)
        response.raise_for_status()

    return response.json().get("transcript", "")


# ============================================================
# Hinglish
# ============================================================

def transcribe_chunk_sarvam(chunk_path: str) -> str:

    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY not found.")

    audio = AudioSegment.from_wav(chunk_path)

    piece_ms = SARVAM_PIECE_SECONDS * 1000

    transcript = ""

    total = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):

        print(f"Sarvam piece {i+1}/{total}")

        piece = audio[start:start + piece_ms]

        piece_path = f"{chunk_path}_piece_{i}.wav"

        piece.export(piece_path, format="wav")

        try:
            transcript += _send_to_sarvam(piece_path) + " "

        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return transcript.strip()


# ============================================================
# Router
# ============================================================

def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
):

    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)

    return transcribe_chunk_whisper(chunk_path)


# ============================================================
# Transcribe Entire Audio
# ============================================================

def transcribe_all(
    chunks: list[str],
    language: str = "english"
) -> str:

    engine = "Sarvam AI" if language.lower() == "hinglish" else f"Whisper ({WHISPER_MODEL})"

    print(f"\nUsing {engine}\n")

    transcript = []

    for i, chunk in enumerate(chunks):

        print(f"Transcribing chunk {i+1}/{len(chunks)}")

        text = transcribe_chunk(chunk, language)

        transcript.append(text)

    print("\n✅ Transcription completed.\n")

    return "\n".join(transcript)