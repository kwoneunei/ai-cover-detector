import os
import math
import json

import numpy as np
import librosa


class BpmProcessor:
    def __init__(
        self,
        train_acc_path,
        json2bpm_path,
        bpm2json_path,
        sample_rate,
        threshold,
    ):
        with open(json2bpm_path, "r", encoding="utf-8") as f:
            j2b_dict = json.load(f)

        with open(bpm2json_path, "r", encoding="utf-8") as f:
            b2j_dict = json.load(f)

        self.train_acc_path = train_acc_path
        self.j2b_dict = j2b_dict
        self.b2j_dict = b2j_dict

        self.sr = sample_rate
        self.thr = threshold / self.sr
        self.waveform_thr = threshold

    def _find_audio_path(self, base_without_ext: str):
        for ext in [".wav", ".flac"]:
            path = base_without_ext + ext
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"Audio file not found: {base_without_ext}.wav/.flac")

    def load_audio_by_json(self, sel_json_name):
        base = os.path.join(self.train_acc_path, sel_json_name[:-5])
        audio_path = self._find_audio_path(base)
        wav, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        return wav

    def sel_accom_from_bpm_group(self, bpm_num, y):
        count, downbeats_duration = 0, 0
        sel_json = None

        if str(bpm_num) not in self.b2j_dict:
            return sel_json, downbeats_duration

        candidate_jsons = self.b2j_dict[str(bpm_num)]
        filtered_gen = filter(lambda x: x.startswith(str(y)), candidate_jsons)
        candidate_list = list(filtered_gen)

        if len(candidate_list) != 0:
            while True:
                sel_json = np.random.choice(candidate_list)
                bpm_beat = self.j2b_dict[sel_json]
                count += 1

                pos1 = np.where(np.array(bpm_beat["beat_positions"]) == 1)[0]
                pos1_len = pos1.shape[0]

                # At least one bar period
                if pos1_len > 2:
                    downbeats_duration = (
                        bpm_beat["downbeats"][-1] - bpm_beat["downbeats"][0]
                    )
                    break

                # Stop because not found
                if count >= 5:
                    break

        return sel_json, downbeats_duration

    def accom_beat_padding(self, waveform, s_name, dbs_duration):
        content = self.j2b_dict[s_name]
        start, end = content["downbeats"][0], content["downbeats"][-1]
        sel_waveform = waveform[int(start * self.sr): int(end * self.sr)]

        if dbs_duration < self.thr and dbs_duration > 0:
            cp_num = math.ceil(self.thr / dbs_duration)
            sel_waveform = np.concatenate([sel_waveform] * cp_num)

        waveform_thr = int(self.thr * self.sr) + 1
        return sel_waveform[:waveform_thr]

    def sv_beat_align(self, wav_sv, sel_json):
        downbeats = self.j2b_dict[sel_json + ".json"]["downbeats"]
        wav_seg = wav_sv[int(downbeats[0] * self.sr): int(downbeats[-1] * self.sr)]

        rand_start = np.random.choice(downbeats[0:-1])
        rand_start_seg = wav_sv[int(rand_start * self.sr): int(downbeats[-1] * self.sr)]

        if rand_start_seg.shape[0] >= self.waveform_thr:
            output = rand_start_seg
        else:
            remain_len = self.waveform_thr - rand_start_seg.shape[0] - wav_seg.shape[0]
            if remain_len >= 0:
                padded_len = math.ceil(remain_len / wav_seg.shape[0]) + 1
            else:
                padded_len = 1
            padded_wav_seg = np.concatenate([wav_seg] * padded_len)
            output = np.concatenate((rand_start_seg, padded_wav_seg))

        output = output[:self.waveform_thr]
        return output


if __name__ == "__main__":
    train_acc_path = "/mnt/ironwolf/kwoneunei/ai-cover-detector/SingFake/split_dump/train/non_vocals/"
    train_vocal_path = "/mnt/ironwolf/kwoneunei/ai-cover-detector/SingFake/split_dump/train/vocals/"
    b2j_path = "/mnt/ironwolf/kwoneunei/ai-cover-detector/SingFake/split_dump/train/bpm2json.json"
    j2b_path = "/mnt/ironwolf/kwoneunei/ai-cover-detector/SingFake/split_dump/train/json2bpm.json"

    with open(j2b_path, "r", encoding="utf-8") as d:
        json2bpm_dict = json.load(d)

    BpmProCls = BpmProcessor(
        train_acc_path=train_acc_path,
        json2bpm_path=j2b_path,
        bpm2json_path=b2j_path,
        sample_rate=16000,
        threshold=64600,
    )

    # non_vocals에서 flac 또는 wav 아무거나 하나 테스트
    test_non_vocal = None
    for fname in os.listdir(train_acc_path):
        if fname.endswith(".flac") or fname.endswith(".wav"):
            test_non_vocal = fname
            break

    if test_non_vocal is None:
        raise FileNotFoundError("No test file found in train/non_vocals")

    json_name = os.path.splitext(test_non_vocal)[0] + ".json"
    wav_path = os.path.join(train_acc_path, test_non_vocal)
    wav, sample_rate = librosa.load(wav_path, sr=16000)

    bpm_n = json2bpm_dict[json_name]["bpm"]

    sel_json, dbs_duration = BpmProCls.sel_accom_from_bpm_group(bpm_n, "1")
    if sel_json is not None:
        sel_wav = BpmProCls.load_audio_by_json(sel_json)
        padded_waveform = BpmProCls.accom_beat_padding(sel_wav, sel_json, dbs_duration)
        print(f"padded_waveform: {padded_waveform.shape}")
    else:
        print("No matching accompaniment found.")

    # vocals에서도 flac 또는 wav 아무거나 하나 테스트
    test_vocal = None
    for fname in os.listdir(train_vocal_path):
        if fname.endswith(".flac") or fname.endswith(".wav"):
            test_vocal = fname
            break

    if test_vocal is None:
        raise FileNotFoundError("No test file found in train/vocals")

    vocal_key = os.path.splitext(test_vocal)[0]
    wav_path = os.path.join(train_vocal_path, test_vocal)
    wav11, sample_rate = librosa.load(wav_path, sr=16000)

    wav_sv = BpmProCls.sv_beat_align(wav11, vocal_key)
    print(f"wav_sv: {wav_sv.shape}")