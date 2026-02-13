import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, Body
from fastapi.responses import Response, JSONResponse

# Keep import if present in repo, but we won't require it for request parsing
try:
    from models import OpenAISpeechRequest  # noqa: F401
except Exception:
    OpenAISpeechRequest = None  # type: ignore

from tts_service import NeuttsNanoGermanService

app = FastAPI(title="NeuTTS OpenAI-compatible API (Nano German)")

tts = NeuttsNanoGermanService()

@app.on_event("startup")
def _startup():
    tts.startup()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": os.getenv("NEUTTS_BACKBONE_REPO", "neuphonic/neutts-nano-german"),
        "voices": tts.list_voices(),
        "device": {
            "backbone": os.getenv("NEUTTS_BACKBONE_DEVICE", "cpu"),
            "codec": os.getenv("NEUTTS_CODEC_DEVICE", "cpu"),
        },
    }

def _parse_openai_tts_payload(payload: dict) -> tuple[str, str, str]:
    # OpenAI uses: input, voice, response_format
    # Some clients send: format instead of response_format
    text = payload.get("input") or payload.get("text") or ""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing required field 'input'.")

    voice = payload.get("voice") or "default"
    if not isinstance(voice, str) or not voice.strip():
        voice = "default"

    fmt = payload.get("response_format") or payload.get("format") or "mp3"
    if not isinstance(fmt, str) or not fmt.strip():
        fmt = "mp3"
    fmt = fmt.strip().lower()

    return text, voice.strip(), fmt

@app.post("/v1/audio/speech")
def openai_speech(payload: dict = Body(...)):
    try:
        text, voice, fmt = _parse_openai_tts_payload(payload)
    except ValueError as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})

    # Synthesize to wav @24k
    wav, sr, meta = tts.synthesize_wav_24k(text, voice)

    # Write to temp dir, read bytes, then return bytes (IMPORTANT: cannot return FileResponse
    # from a TemporaryDirectory because the directory is deleted before FileResponse reads it)
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / f"speech.{fmt}"
        try:
            tts.write_audio(wav, sr, fmt, out_path)
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": f"Audio export failed: {e}"})

        if not out_path.exists():
            return JSONResponse(status_code=500, content={"detail": f"Output file was not created: {out_path.name}"})

        data = out_path.read_bytes()

    media_type = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "flac": "audio/flac",
        "pcm": "audio/pcm",
    }.get(fmt, "application/octet-stream")

    return Response(
        content=data,
        media_type=media_type,
        headers={
            "x-tts-latency-s": str(meta.get("latency_s", "")),
            "x-tts-voice-id": str(meta.get("voice_id", "")),
        },
    )

@app.post("/synthesize")
def synthesize(payload: dict = Body(...)):
    # convenience alias
    return openai_speech(payload)
