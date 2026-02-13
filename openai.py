import os
import tempfile
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from models import OpenAISpeechRequest
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
    }

@app.post("/v1/audio/speech")
def openai_speech(req: OpenAISpeechRequest):
    wav, sr, meta = tts.synthesize_wav_24k(req.input, req.voice)

    suffix = f".{req.response_format}"
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / f"speech{suffix}"
        tts.write_audio(wav, sr, req.response_format, out_path)

        # Minimal OpenAI-compat: return raw bytes with correct content-type; clients usually infer by endpoint
        media_type = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "flac": "audio/flac",
            "pcm": "audio/pcm",
        }.get(req.response_format, "application/octet-stream")

        # NOTE: FileResponse streams from disk and is fine for large audio files.
        return FileResponse(path=str(out_path), media_type=media_type, filename=out_path.name, headers={
            "x-tts-latency-s": str(meta["latency_s"]),
            "x-tts-voice-id": meta["voice_id"],
        })

@app.post("/synthesize")
def synthesize(req: OpenAISpeechRequest):
    # convenience alias
    return openai_speech(req)
