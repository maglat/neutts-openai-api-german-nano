import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

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
    """
    Supports BOTH:
      1) voices.json manifest (optional)
      2) automatic folder scan

    Merge rule:
      - If voices.json exists: load it first (authoritative)
      - Always scan folder: add any voices not present in manifest
      - This prevents voices.json from "hiding" other voices.
    """

    def __init__(self, voice_dir: Path, manifest_name: str = "voices.json") -> None:
        self.voice_dir = voice_dir
        self.manifest_path = voice_dir / manifest_name
        self.voices: Dict[str, VoiceAsset] = {}

    def _load_from_manifest(self) -> Dict[str, VoiceAsset]:
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

        return voices

    def _load_by_scanning(self) -> Dict[str, VoiceAsset]:
        voices: Dict[str, VoiceAsset] = {}
        if not self.voice_dir.exists():
            return voices

        for wav in sorted(self.voice_dir.glob("*.wav")):
            voice_id = wav.stem
            txt = self.voice_dir / f"{voice_id}.txt"
            if not txt.exists():
                continue

            codes = None
            for ext in (".pt", ".pth", ".bin", ".npy"):
                c = self.voice_dir / f"{voice_id}{ext}"
                if c.exists():
                    codes = c
                    break

            voices[voice_id] = VoiceAsset(wav_path=wav, txt_path=txt, codes_path=codes)

        return voices

    def load(self) -> None:
        voices: Dict[str, VoiceAsset] = {}

        # 1) Manifest (if present)
        if self.manifest_path.exists():
            voices.update(self._load_from_manifest())

        # 2) Always scan folder and merge missing voices
        scanned = self._load_by_scanning()
        for vid, asset in scanned.items():
            if vid not in voices:
                voices[vid] = asset

        self.voices = voices

        if not self.voices:
            raise RuntimeError(
                f"No voices found in {self.voice_dir}. "
                f"Provide voices.json OR at least one pair <voice>.wav + <voice>.txt."
            )

        # Ensure default exists
        if "default" not in self.voices:
            first = next(iter(self.voices.keys()))
            self.voices["default"] = self.voices[first]

    def resolve(self, voice_id: str) -> VoiceAsset:
        if voice_id not in self.voices:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown voice '{voice_id}'. Available: {sorted(self.voices.keys())}",
            )
        return self.voices[voice_id]

    def list_ids(self) -> list[str]:
        return sorted(self.voices.keys())


class NeuttsNanoGermanService:
    def __init__(self) -> None:
        self.backbone_repo = os.getenv("NEUTTS_BACKBONE_REPO", "neuphonic/neutts-nano-german")
        self.codec_repo = os.getenv("NEUTTS_CODEC_REPO", "neuphonic/neucodec")
        self.backbone_device = os.getenv("NEUTTS_BACKBONE_DEVICE", "cpu")
        self.codec_device = os.getenv("NEUTTS_CODEC_DEVICE", "cpu")

        voice_dir = Path(os.getenv("VOICE_SAMPLES_DIR", "/voices"))
        self.registry = VoiceRegistry(voice_dir=voice_dir)
        self.tts: Optional[NeuTTS] = None

    def list_voices(self) -> list[str]:
        try:
            return self.registry.list_ids()
        except Exception:
            return []

    def startup(self) -> None:
        self.registry.load()

        self.tts = NeuTTS(
            backbone_repo=self.backbone_repo,
            backbone_device=self.backbone_device,
            codec_repo=self.codec_repo,
            codec_device=self.codec_device,
        )

    def _encode_reference(self, asset: VoiceAsset):
        assert self.tts is not None

        if asset.codes_path is not None:
            suffix = asset.codes_path.suffix.lower()
            if suffix == ".npy":
                return np.load(asset.codes_path)

            import torch
            return torch.load(asset.codes_path, map_location="cpu")

        return self.tts.encode_reference(str(asset.wav_path))

    def synthesize_wav_24k(self, text: str, voice_id: str):
        assert self.tts is not None

        voice_id = (voice_id or "default").strip() or "default"
        asset = self.registry.resolve(voice_id)

        ref_text = asset.txt_path.read_text(encoding="utf-8").strip()
        ref_codes = self._encode_reference(asset)

        t0 = time.time()
        wav = self.tts.infer(text, ref_codes, ref_text)
        latency_s = time.time() - t0

        meta = {
            "voice_id": voice_id,
            "latency_s": latency_s,
            "sample_rate": 24000,
        }
        return wav, 24000, meta

    def write_audio(self, wav: np.ndarray, sr: int, fmt: str, out_path: Path) -> None:
        fmt = (fmt or "mp3").strip().lower()

        if fmt == "wav":
            sf.write(str(out_path), wav, sr)
            return

        tmp_wav = out_path.with_suffix(".wav")
        sf.write(str(tmp_wav), wav, sr)

        if fmt == "pcm":
            wav.astype(np.float32, copy=False).tofile(out_path)
            tmp_wav.unlink(missing_ok=True)
            return

        import subprocess

        if fmt == "mp3":
            cmd = ["ffmpeg", "-y", "-i", str(tmp_wav), "-codec:a", "libmp3lame", "-q:a", "4", str(out_path)]
        else:
            cmd = ["ffmpeg", "-y", "-i", str(tmp_wav), str(out_path)]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tmp_wav.unlink(missing_ok=True)

        if not out_path.exists():
            raise RuntimeError(f"ffmpeg did not create output: {out_path}")
