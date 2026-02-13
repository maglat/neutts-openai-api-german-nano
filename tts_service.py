import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import soundfile as sf
from fastapi import HTTPException
from neutts import NeuTTS

@dataclass(frozen=True)
class VoiceAsset:
    wav_path: Path
    txt_path: Path
    codes_path: Optional[Path] = None

class VoiceRegistry:
    def __init__(self, voice_dir: Path, manifest_name: str = "voices.json") -> None:
        self.voice_dir = voice_dir
        self.manifest_path = voice_dir / manifest_name
        self.voices: Dict[str, VoiceAsset] = {}

    def load(self) -> None:
        if not self.manifest_path.exists():
            raise RuntimeError(f"voices manifest not found: {self.manifest_path}")

        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        voices: Dict[str, VoiceAsset] = {}

        for voice_id, entry in data.items():
            wav = self.voice_dir / entry["wav"]
            txt = self.voice_dir / entry["txt"]
            codes = self.voice_dir / entry["codes"] if "codes" in entry else None
            if not wav.exists():
                raise RuntimeError(f"missing wav for voice '{voice_id}': {wav}")
            if not txt.exists():
                raise RuntimeError(f"missing txt for voice '{voice_id}': {txt}")
            if codes is not None and not codes.exists():
                raise RuntimeError(f"missing codes for voice '{voice_id}': {codes}")

            voices[voice_id] = VoiceAsset(wav_path=wav, txt_path=txt, codes_path=codes)

        self.voices = voices

    def resolve(self, voice_id: str) -> VoiceAsset:
        if voice_id not in self.voices:
            raise HTTPException(status_code=400, detail=f"Unknown voice '{voice_id}'. Available: {sorted(self.voices.keys())}")
        return self.voices[voice_id]

class NeuttsNanoGermanService:
    def __init__(self) -> None:
        self.backbone_repo = os.getenv("NEUTTS_BACKBONE_REPO", "neuphonic/neutts-nano-german")
        self.codec_repo = os.getenv("NEUTTS_CODEC_REPO", "neuphonic/neucodec")
        self.backbone_device = os.getenv("NEUTTS_BACKBONE_DEVICE", "cpu")  # "cpu" or "cuda"
        self.codec_device = os.getenv("NEUTTS_CODEC_DEVICE", "cpu")        # "cpu" or "cuda"

        voice_dir = Path(os.getenv("VOICE_SAMPLES_DIR", "/voices"))
        self.registry = VoiceRegistry(voice_dir=voice_dir)
        self.tts: Optional[NeuTTS] = None

    def startup(self) -> None:
        # Load voice registry first (fail fast)
        self.registry.load()

        # Load model once
        self.tts = NeuTTS(
            backbone_repo=self.backbone_repo,
            backbone_device=self.backbone_device,
            codec_repo=self.codec_repo,
            codec_device=self.codec_device,
        )

    def _encode_reference(self, asset: VoiceAsset) -> np.ndarray:
        assert self.tts is not None
        # Prefer cached codes if provided:
        if asset.codes_path is not None:
            # torch.load would be typical, but keep it generic here by allowing numpy as well
            # If you use torch tensors, replace with torch.load(asset.codes_path)
            import torch
            return torch.load(asset.codes_path)

        # Otherwise compute on the fly
        return self.tts.encode_reference(str(asset.wav_path))

    def synthesize_wav_24k(self, text: str, voice_id: str) -> Tuple[np.ndarray, int, Dict]:
        assert self.tts is not None

        asset = self.registry.resolve(voice_id)
        ref_text = asset.txt_path.read_text(encoding="utf-8").strip()
        ref_codes = self._encode_reference(asset)

        t0 = time.time()
        wav = self.tts.infer(text, ref_codes, ref_text)  # returns waveform at 24kHz per docs
        latency_s = time.time() - t0

        meta = {
            "voice_id": voice_id,
            "latency_s": latency_s,
            "sample_rate": 24000,
            "backbone_repo": self.backbone_repo,
            "codec_repo": self.codec_repo,
            "backbone_device": self.backbone_device,
            "codec_device": self.codec_device,
        }
        return wav, 24000, meta

    def write_audio(self, wav: np.ndarray, sr: int, fmt: str, out_path: Path) -> None:
        # Write WAV first (PCM float) then optionally convert using ffmpeg for mp3/flac.
        tmp_wav = out_path.with_suffix(".wav")
        sf.write(str(tmp_wav), wav, sr)

        if fmt == "wav":
            tmp_wav.replace(out_path)
            return

        if fmt == "pcm":
            # raw little-endian float32 PCM
            pcm_path = out_path
            wav_f32 = wav.astype(np.float32, copy=False)
            pcm_path.write_bytes(wav_f32.tobytes())
            tmp_wav.unlink(missing_ok=True)
            return

        # mp3/flac/aac/opus require ffmpeg (recommended)
        import subprocess
        cmd = ["ffmpeg", "-y", "-i", str(tmp_wav), str(out_path)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tmp_wav.unlink(missing_ok=True)
