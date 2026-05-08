import os
from pathlib import Path

from huggingface_hub import hf_hub_download


HF_REPO_ID = os.environ.get("HF_REPO_ID", "eunei/ai-cover-detector-singgraph")
HF_FILENAME = os.environ.get("HF_FILENAME", "best.pth")
HF_REPO_TYPE = "model"

LOCAL_WEIGHT_DIR = Path(os.environ.get("LOCAL_WEIGHT_DIR", "SingGraph/weights"))
LOCAL_WEIGHT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_weight_file() -> str:
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        repo_type=HF_REPO_TYPE,
        local_dir=str(LOCAL_WEIGHT_DIR),
        local_dir_use_symlinks=False,
    )
    return local_path