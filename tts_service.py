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
    """Loads voices from voices.json inside VOICE_SAMPLES_DIR.

    voices.json format:
      {
        "default": {"wav": "default.wav", "txt": "default.txt", "codes": "default.pt"},
        "greta":   {"wav": "greta.wav",   "txt": "greta.txt",   "codes": "greta.pt"}
      }
    """

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
            raise HTTPException(
                status_code=400,
                detail=f"Unknown voice '{voice_id}'. Available: {sorted(self.voices.keys())}",
            )
        return self.voices[voice_id]


class NeuttsNanoGermanService:
    def __init__(self) -> None:
        # ---------------------------------------------------------------------
        # Backbone selection (FP32 vs GGUF quantized variants)
        #
        # Priority:
        #   1) NEUTTS_BACKBONE_REPO (explicit)
        #   2) NEUTTS_MODEL_BASE + NEUTTS_BACKBONE_VARIANT
        #
        # Examples:
        #   FP32: NEUTTS_MODEL_BASE=neutts-nano-german, NEUTTS_BACKBONE_VARIANT=fp32
        #   Q8:   NEUTTS_MODEL_BASE=neutts-nano-german, NEUTTS_BACKBONE_VARIANT=q8
        #   Q4:   NEUTTS_MODEL_BASE=neutts-nano-german, NEUTTS_BACKBONE_VARIANT=q4
        #
        # Mapping (Neuphonic HF naming):
        #   fp32 -> neuphonic/<base>
        #   q8   -> neuphonic/<base>-q8-gguf
        #   q4   -> neuphonic/<base>-q4-gguf
        # ---------------------------------------------------------------------
        explicit_repo = os.getenv("NEUTTS_BACKBONE_REPO")
        model_base = os.getenv("NEUTTS_MODEL_BASE", "neutts-nano-german").strip()
        variant = os.getenv("NEUTTS_BACKBONE_VARIANT", "fp32").strip().lower()

        if explicit_repo and explicit_repo.strip():
            self.backbone_repo = explicit_repo.strip()
        else:
            if variant == "q4":
                self.backbone_repo = f"neuphonic/{model_base}-q4-gguf"
            elif variant == "q8":
                self.backbone_repo = f"neuphonic/{model_base}-q8-gguf"
            else:
                self.backbone_repo = f"neuphonic/{model_base}"

        self.codec_repo = os.getenv("NEUTTS_CODEC_REPO", "neuphonic/neucodec")

        # Devices: "cpu" or "cuda"
        self.backbone_device = os.getenv("NEUTTS_BACKBONE_DEVICE", "cpu")
        self.codec_device = os.getenv("NEUTTS_CODEC_DEVICE", "cpu")

        voice_dir = Path(os.getenv("VOICE_SAMPLES_DIR", "/voices"))
        self.registry = VoiceRegistry(voice_dir=voice_dir)
        self.tts: Optional[NeuTTS] = None

    def startup(self) -> None:
        # Load voices first (fail fast)
        self.registry.load()

        # If GGUF backbone is selected, we need llama-cpp-python installed
        if "gguf" in self.backbone_repo.lower():
            try:
                import llama_cpp  # noqa: F401
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "GGUF backbone selected but llama-cpp-python is not installed.\n"
                    "Fix: add 'llama-cpp-python' to requirements.txt and rebuild the image."
                ) from e

        # Load model once
        self.tts = NeuTTS(
            backbone_repo=self.backbone_repo,
            backbone_device=self.backbone_device,
            codec_repo=self.codec_repo,
            codec_device=self.codec_device,
        )

    def _encode_reference(self, asset: VoiceAsset):
        assert self.tts is not None

        # Prefer cached codes if provided (your *.pt files)
        if asset.codes_path is not None:
            import torch
            return torch.load(asset.codes_path, map_location="cpu")

        # Otherwise compute on the fly from wav
        return self.tts.encode_reference(str(asset.wav_path))

    def synthesize_wav_24k(self, text: str, voice_id: str) -> Tuple[np.ndarray, int, Dict]:
        assert self.tts is not None

        asset = self.registry.resolve(voice_id)
        ref_text = asset.txt_path.read_text(encoding="utf-8").strip()
        ref_codes = self._encode_reference(asset)

        t0 = time.time()
        wav = self.tts.infer(text, ref_codes, ref_text)  # 24kHz waveform
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
        # Write WAV first then optionally convert using ffmpeg for mp3/flac/opus/aac.
        tmp_wav = out_path.with_suffix(".wav")
        sf.write(str(tmp_wav), wav, sr)

        if fmt == "wav":
            tmp_wav.replace(out_path)
            return

        if fmt == "pcm":
            # raw little-endian float32 PCM
            wav_f32 = wav.astype(np.float32, copy=False)
            out_path.write_bytes(wav_f32.tobytes())
            tmp_wav.unlink(missing_ok=True)
            return

        import subprocess
        cmd = ["ffmpeg", "-y", "-i", str(tmp_wav), str(out_path)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tmp_wav.unlink(missing_ok=True)
