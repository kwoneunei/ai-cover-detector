from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

train_dir = Path("./SingGraph/dataset/train")
dev_dir = Path("./SingGraph/dataset/dev")
dev_dir.mkdir(parents=True, exist_ok=True)

all_files = [p for p in train_dir.iterdir() if p.is_file()]

bonafide_files = [p for p in all_files if p.name.startswith("bonafide_")]
deepfake_files = [p for p in all_files if p.name.startswith("deepfake_")]

if len(bonafide_files) == 0 and len(deepfake_files) == 0:
    raise ValueError("파일명이 bonafide_ 또는 deepfake_ 로 시작하지 않습니다.")

# 각 클래스별 10%를 dev로 분리
_, bonafide_dev = train_test_split(
    bonafide_files, test_size=0.1, random_state=1234
)
_, deepfake_dev = train_test_split(
    deepfake_files, test_size=0.1, random_state=1234
)

dev_files = bonafide_dev + deepfake_dev

for src in dev_files:
    dst = dev_dir / src.name
    shutil.move(str(src), str(dst))   # 확인 전에는 copy 권장

print(f"bonafide total: {len(bonafide_files)}, dev: {len(bonafide_dev)}")
print(f"deepfake total: {len(deepfake_files)}, dev: {len(deepfake_dev)}")
print(f"total dev files: {len(dev_files)}")