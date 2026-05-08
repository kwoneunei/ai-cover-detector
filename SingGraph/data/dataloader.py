import os

import librosa
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from utils.utils import str_to_bool

from data.RawBoost import process_Rawboost_feature

___author__ = "Xuanjun Chen"
__email__ = "d12942018@ntu.edu.tw"


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, start: int, end: int):
    x_len = x.shape[0]

    if x_len == 0:
        raise ValueError("Empty waveform received.")

    if end <= x_len:
        return x[start:end]

    padded_x = np.tile(x, (end // x_len + 1))
    return padded_x[start:end]


def parse_label_from_key(key: str) -> int:
    label = key.split("_")[0].lower()

    if label == "bonafide":
        return 1
    elif label == "deepfake":
        return 0
    else:
        raise ValueError(f"Unknown label prefix: {label}")


class Dataset_SingFake(Dataset):
    def __init__(self, args, base_dir, algo, state, is_mixture=False, target_sr=16000):
        """
        Custom dataset for flat directory structure:
        base_dir/
            bonafide_xxx.flac
            deepfake_xxx.flac
        """
        self.base_dir = base_dir
        self.target_sr = target_sr
        self.cut = 64600
        self.args = args
        self.algo = algo
        self.state = state

        self.file_list = []

        if not os.path.exists(self.base_dir):
            raise FileNotFoundError(f"{self.base_dir} does not exist!")

        for file in os.listdir(self.base_dir):
            if file.endswith(".flac"):
                self.file_list.append(file[:-5])

        self.file_list.sort()

        if len(self.file_list) == 0:
            raise ValueError(f"No .flac files found in {self.base_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        key = self.file_list[index]
        file_path = os.path.join(self.base_dir, key + ".flac")

        try:
            X, fs = librosa.load(file_path, sr=self.target_sr, mono=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {file_path}") from e

        if X is None or np.size(X) == 0:
            raise RuntimeError(f"Loaded empty audio file: {file_path}")

        if X.ndim > 1 and X.shape[0] > 1:
            channel_id = np.random.randint(X.shape[0])
            X = X[channel_id]

        if X.ndim > 1:
            X = np.squeeze(X)

        if np.size(X) == 0:
            raise RuntimeError(f"Empty waveform after channel selection: {file_path}")

        if self.state == "train":
            X = process_Rawboost_feature(X, fs, self.args, self.algo)

        waveform_shift = X.shape[0] - self.cut
        if waveform_shift > 0:
            x_start = np.random.randint(0, waveform_shift)
        else:
            x_start = 0

        x_end = x_start + self.cut
        X_pad = pad_random(X, x_start, x_end)

        max_abs = np.max(np.abs(X_pad))
        if max_abs > 0:
            X_pad = X_pad / max_abs

        x_inp = Tensor(X_pad)
        y = parse_label_from_key(key)

        return x_inp, y


class Dataset_SingFake_mert_w2v(Dataset):
    def __init__(self, args, config, base_dir, algo, state,
                 target_sr=16000, target_sr2=24000):
        """
        Custom dataset for flat directory structure:
        base_dir/
            bonafide_xxx.flac
            deepfake_xxx.flac

        Existing training/evaluation pipeline expects:
            return x_inp, x2_inp, y

        Since your dataset only has one audio file per sample,
        this dataset returns the same waveform twice:
            return x_inp, x2_inp, y
        """
        self.base_dir = base_dir
        self.is_rawboost = str_to_bool(config["is_rawboost"])
        self.target_sr = target_sr
        self.target_sr2 = target_sr2
        self.cut16 = 64600
        self.duration = 4.0375
        self.cut24 = 96900
        self.args = args
        self.algo = algo
        self.state = state

        self.file_list = []

        if not os.path.exists(self.base_dir):
            raise FileNotFoundError(f"{self.base_dir} does not exist!")

        for file in os.listdir(self.base_dir):
            if file.endswith(".flac"):
                self.file_list.append(file[:-5])

        self.file_list.sort()

        if len(self.file_list) == 0:
            raise ValueError(f"No .flac files found in {self.base_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        key = self.file_list[index]
        y = parse_label_from_key(key)

        file_path = os.path.join(self.base_dir, key + ".flac")

        try:
            X, fs = librosa.load(file_path, sr=self.target_sr, mono=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {file_path}") from e

        if X is None or np.size(X) == 0:
            raise RuntimeError(f"Loaded empty audio file: {file_path}")

        if X.ndim > 1 and X.shape[0] > 1:
            channel_id = np.random.randint(X.shape[0])
            X = X[channel_id]

        if X.ndim > 1:
            X = np.squeeze(X)

        if np.size(X) == 0:
            raise RuntimeError(f"Empty waveform after channel selection: {file_path}")

        if self.state == "train" and self.is_rawboost:
            X = process_Rawboost_feature(X, fs, self.args, self.algo)

        waveform_shift = X.shape[0] - self.cut16
        if waveform_shift > 0:
            x_start = np.random.randint(0, waveform_shift)
        else:
            x_start = 0

        x_end = x_start + self.cut16
        X_pad = pad_random(X, x_start, x_end)

        max_abs_x = np.max(np.abs(X_pad))
        if max_abs_x > 0:
            X_pad = X_pad / max_abs_x

        x_inp = Tensor(X_pad)

        # 모델이 두 입력(batch_x, batch_x2)을 기대하므로 동일 입력을 두 번 반환
        x2_inp = Tensor(X_pad.copy())

        return x_inp, x2_inp, y