import io
import numpy as np
import soundfile as sf
import librosa
import torch

TARGET_SR = 16000
CUT = 64600


def pad_random_fixed(x: np.ndarray, max_len: int = CUT) -> np.ndarray:
    x_len = x.shape[0]
    if x_len == 0:
        raise ValueError("Empty waveform")
    if x_len >= max_len:
        start = (x_len - max_len) // 2
        return x[start:start + max_len]
    repeat = int(max_len / x_len) + 1
    return np.tile(x, repeat)[:max_len]


def load_audio_bytes(file_bytes: bytes, target_sr: int = TARGET_SR) -> np.ndarray:
    data, sr = sf.read(io.BytesIO(file_bytes), always_2d=True)
    data = data.T

    if data.shape[0] > 1:
        data = data[0]
    else:
        data = data.squeeze(0)

    if sr != target_sr:
        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=target_sr)

    return data.astype(np.float32)


def normalize_audio(x: np.ndarray) -> np.ndarray:
    max_abs = np.max(np.abs(x))
    if max_abs > 0:
        x = x / max_abs
    return x


def preprocess_audio(file_bytes: bytes) -> torch.Tensor:
    x = load_audio_bytes(file_bytes, target_sr=TARGET_SR)
    x = pad_random_fixed(x, CUT)
    x = normalize_audio(x)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)