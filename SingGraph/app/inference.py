import json
from pathlib import Path

import torch

from app.preprocess import preprocess_audio
from SingGraph.model.SingGraph import Wav2Vec2Model


class SingGraphInference:
    def __init__(self, config_path: str, weight_path: str, device: str | None = None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model = Wav2Vec2Model(self.config["model_config"], self.device).to(self.device)
        state = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict_bytes(self, file_bytes: bytes) -> dict:
        x = preprocess_audio(file_bytes).to(self.device)
        x2 = x.clone()

        out = self.model(x, x2)
        prob = torch.softmax(out, dim=1)[0].detach().cpu().numpy()

        return {
            "pred_label": int(prob.argmax()),
            "bonafide_score": float(prob[1]),
            "deepfake_score": float(prob[0]),
        }