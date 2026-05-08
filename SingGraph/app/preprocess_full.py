import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection


TARGET_SR = 16000
CUT = 64600


class FullPreprocessor:
    def __init__(self, hf_token: str, demucs_model: str = "mdx_extra"):
        self.hf_token = hf_token
        self.demucs_model = demucs_model
        self.vad_pipeline = self._build_vad_pipeline()

    def _build_vad_pipeline(self):
        model = Model.from_pretrained(
            "pyannote/segmentation-3.0",
            use_auth_token=self.hf_token,
        )
        pipeline = VoiceActivityDetection(segmentation=model)
        pipeline.instantiate(
            {
                "min_duration_on": 0.0,
                "min_duration_off": 0.0,
            }
        )
        return pipeline

    def separate_audio(self, input_audio: Path, output_root: Path) -> Tuple[Path, Path]:
        cmd = [
            "demucs",
            "--device",
            "cpu",
            "--two-stems=vocals",
            "-n",
            self.demucs_model,
            "-o",
            str(output_root),
            str(input_audio),
        ]
        subprocess.run(cmd, check=True)

        stem_name = input_audio.stem
        vocals_wav = output_root / self.demucs_model / stem_name / "vocals.wav"
        no_vocals_wav = output_root / self.demucs_model / stem_name / "no_vocals.wav"

        if not vocals_wav.exists():
            raise FileNotFoundError(f"vocals.wav not found: {vocals_wav}")
        if not no_vocals_wav.exists():
            raise FileNotFoundError(f"no_vocals.wav not found: {no_vocals_wav}")

        return vocals_wav, no_vocals_wav

    def run_vad(self, vocals_wav: Path) -> List[Tuple[float, float]]:
        waveform, sample_rate = sf.read(str(vocals_wav), always_2d=True)
        waveform = torch.from_numpy(waveform.T).float()

        vad = self.vad_pipeline(
            {
                "waveform": waveform,
                "sample_rate": sample_rate,
            }
        )

        segments = []
        vad_str = str(vad)
        for line in vad_str.splitlines():
            line = line.strip()
            if not line:
                continue
            parsed = self._parse_vad_line(line)
            if parsed is not None:
                segments.append(parsed)

        return segments

    def _parse_vad_line(self, line: str):
        try:
            items = line.split("]")[0].split("[")[1].split("-->")
            start_times = items[0].strip().split(":")
            end_times = items[1].strip().split(":")

            start_time = (
                float(start_times[2])
                + float(start_times[1]) * 60.0
                + float(start_times[0]) * 3600.0
            )
            end_time = (
                float(end_times[2])
                + float(end_times[1]) * 60.0
                + float(end_times[0]) * 3600.0
            )
            return start_time, end_time
        except Exception:
            return None

    def _load_audio(self, path: Path):
        audio, sr = sf.read(str(path), always_2d=True)
        if sr != TARGET_SR:
            import librosa
            audio = audio.T
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR, axis=1)
            audio = audio.T
            sr = TARGET_SR
        return audio, sr

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        max_abs = np.max(np.abs(x))
        if max_abs > 0:
            x = x / max_abs
        return x.astype(np.float32)

    def _fix_length(self, x: np.ndarray, target_len: int = CUT) -> np.ndarray:
        if x.shape[0] >= target_len:
            start = (x.shape[0] - target_len) // 2
            return x[start:start + target_len]

        repeat = int(np.ceil(target_len / x.shape[0]))
        x = np.tile(x, repeat)
        return x[:target_len]

    def _to_mono_first_channel(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x
        return x[:, 0] if x.shape[1] > 0 else x.squeeze()

    def make_clip_pairs(
        self,
        vocals_wav: Path,
        no_vocals_wav: Path,
        segments: List[Tuple[float, float]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        vocal_audio, vocal_sr = self._load_audio(vocals_wav)
        non_vocal_audio, non_vocal_sr = self._load_audio(no_vocals_wav)

        if vocal_sr != TARGET_SR or non_vocal_sr != TARGET_SR:
            raise ValueError("Sample rate mismatch after resampling.")

        clip_pairs = []

        for start_time, end_time in segments:
            start_sample = int(start_time * TARGET_SR)
            end_sample = int(end_time * TARGET_SR)

            if end_sample <= start_sample:
                continue

            vocal_seg = vocal_audio[start_sample:end_sample]
            non_vocal_seg = non_vocal_audio[start_sample:end_sample]

            vocal_seg = self._to_mono_first_channel(vocal_seg)
            non_vocal_seg = self._to_mono_first_channel(non_vocal_seg)

            if vocal_seg.size == 0 or non_vocal_seg.size == 0:
                continue

            vocal_seg = self._fix_length(vocal_seg, CUT)
            non_vocal_seg = self._fix_length(non_vocal_seg, CUT)

            vocal_seg = self._normalize(vocal_seg)
            non_vocal_seg = self._normalize(non_vocal_seg)

            vocal_tensor = torch.tensor(vocal_seg, dtype=torch.float32).unsqueeze(0)
            non_vocal_tensor = torch.tensor(non_vocal_seg, dtype=torch.float32).unsqueeze(0)

            clip_pairs.append((vocal_tensor, non_vocal_tensor))

        return clip_pairs

    def preprocess_uploaded_file(self, input_audio_path: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        input_audio = Path(input_audio_path).resolve()
        if not input_audio.exists():
            raise FileNotFoundError(f"Input audio not found: {input_audio}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            vocals_wav, no_vocals_wav = self.separate_audio(input_audio, tmp_root)
            segments = self.run_vad(vocals_wav)
            clip_pairs = self.make_clip_pairs(vocals_wav, no_vocals_wav, segments)

        if len(clip_pairs) == 0:
            raise RuntimeError("No valid vocal segments found after VAD.")

        return clip_pairs