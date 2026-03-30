# 🎵 AI Cover Song Detection — 오디오 딥페이크 탐지 기술을 활용한 AI 커버곡 식별 시스템

---

## 📌 프로젝트 소개

생성형 AI 기술의 발전으로 특정 가수의 목소리를 모방한 AI 커버곡이 YouTube, SNS 등 플랫폼에서 빠르게 확산되고 있습니다. 이러한 AI 커버곡은 가수의 **음성권 및 저작권 침해** 문제를 초래할 수 있습니다.

기존 오디오 딥페이크 탐지 모델은 주로 연구용 데이터셋에서만 성능이 검증되었을 뿐, **실제 AI 커버곡 환경에서의 동작 여부는 충분히 검증되지 않았습니다.**

본 프로젝트는 SingFake를 베이스라인으로, 실제 플랫폼에서 수집한 AI 커버곡 데이터를 활용해 모델을 파인튜닝하고, 음원 파일을 업로드하면 AI 커버곡 여부를 자동으로 판별해주는 **웹 기반 서비스**까지 구현하는 것을 목표로 합니다.

---

## 🎯 프로젝트 목표

### 1. 기존 딥페이크 탐지 모델의 실환경 적용 가능성 검증
- AASIST 등 기존 오디오 딥페이크 탐지 모델 적용
- SingFake, CtrSVDD 공개 데이터셋에서 평가된 기존 모델들을 YouTube AI 커버곡 환경에서 추가 테스트
- 연구용 데이터셋과 실제 플랫폼 환경 간의 성능 차이 분석

### 2. 실환경 반영 학습을 통한 탐지 성능 향상
- AI 커버곡 데이터셋 수집 및 정제
- Augmentation 기반 재학습 및 모델 구조 개선
- 실제 업로드 환경(압축, 리버브, 노이즈)에 대한 적응력 향상

### 3. 웹 기반 AI 커버곡 판별 시스템 구현
- 사용자가 음원 파일을 업로드하면 AI 커버곡 여부 자동 판별
- 판별 결과와 신뢰도를 직관적으로 시각화

---

## 🗂️ 데이터셋

| 데이터셋 | 설명 | 비고 |
|---|---|---|
| [SingFake](https://singfake.org/) | 베이스라인. bonafide 28.93h + deepfake 29.40h | ICASSP 2024 |
| 자체 수집 데이터 | YouTube 등에서 수집한 실제 AI 커버곡 | 본 프로젝트 구축 |

### 자체 데이터 수집 파이프라인

```
YouTube/Bilibili 수집 (yt-dlp)
  → 오디오 추출 (ffmpeg, 16kHz)
  → 보컬 분리 (Demucs htdemucs)
  → 보컬 구간 세그먼트 추출 (PyAnnote VAD)
  → 클립 필터링 및 메타데이터 어노테이션
```

---

## 🧠 모델

베이스라인으로 평가하는 모델들은 다음과 같습니다.

| 모델 | 설명 |
|---|---|
| LFCC + AASIST | 스펙트로그램 기반 경량 모델 |
| Wav2Vec2 + AASIST | Raw waveform 기반, SingFake 논문에서 평가된 베이스라인 |
| Whisper encoder | 노이즈 강건성 강점 |
| SingGraph | MERT + wav2vec2 + Graph, 현재 SOTA |

파인튜닝은 각 모델의 **사전학습된 체크포인트(ASVspoof 등으로 학습된 원본 가중치)**를 불러와, SingFake 및 수집된 실환경 AI 커버곡 데이터로 진행합니다.

---

## 🛠️ 개발 환경 및 기술 스택

| 분류 | 스택 |
|---|---|
| 언어 | Python 3.10+ |
| 딥러닝 | PyTorch |
| 오디오 처리 | Librosa, ffmpeg, Demucs, PyAnnote |
| 웹 백엔드 | FastAPI (또는 Flask) |
| 웹 프론트엔드 | HTML / CSS / JavaScript |
| 실험 관리 | WandB |

---

## 📊 평가 지표

- **EER** (Equal Error Rate) — SingFake 공식 지표
- **F1-score**
- **ROC-AUC**
- Seen/Unseen singer 분리 평가
- Vocal / Mixture 입력 형태별 비교

---

## 📁 프로젝트 구조

```
📦 ai-cover-detection
 ┣ 📂 data/
 ┃ ┣ 📂 raw/                  # 수집된 원본 오디오
 ┃ ┗ 📂 processed/            # 전처리 완료 데이터
 ┣ 📂 src/
 ┃ ┣ 📂 data_collection/      # yt-dlp 크롤링, 전처리 파이프라인
 ┃ ┣ 📂 models/               # AASIST, Wav2Vec2, SingGraph 구현체
 ┃ ┣ 📂 train/                # 학습 스크립트
 ┃ ┣ 📂 evaluate/             # EER, F1 평가 스크립트
 ┃ ┗ 📂 web/                  # FastAPI 백엔드 + 프론트엔드
 ┣ 📂 configs/                # 실험 설정 파일 (yaml)
 ┣ 📂 notebooks/              # 분석 및 시각화
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```

---

## 🗓️ 개발 일정

| 기간 | 내용 |
|---|---|
| ~3/18 | 프로젝트 주제 및 방향 설정, 제안서 작성 완료 ✅ |
| ~4/13 | 공개 데이터셋 학습, 베이스라인 모델 실험, 중간 보고서 작성 |
| ~4/30 | Augmentation / Consistency loss 기반 성능 개선 실험 |
| ~5/13 | 모델 최종 점검, 웹 서비스 연동 및 데모 구현 |
| ~5/25 | 결과 보고서 작성 |

---

## 📈 예상 결과 및 기대효과

- 기존 오디오 딥페이크 탐지 모델의 실환경 성능을 **정량적으로 분석**한 결과 제공
- SingFake 베이스라인 대비 **실환경 AI 커버곡 탐지 성능 향상** 모델 제안
- 음원 파일 업로드만으로 AI 커버곡 여부를 확인할 수 있는 **웹 서비스 구현**
- 음성 저작권 침해 방지에 기여할 수 있는 기술적·사회적 활용 가능성 검증


---

## 📚 참고 문헌

1. Y. Zang et al., "SingFake: Singing Voice Deepfake Detection," *ICASSP 2024*
2. J. Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks," *ICASSP 2022*
3. A. Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," *NeurIPS 2020*
4. Y. Zang et al., "CtrSVDD: A Benchmark Dataset and Baseline Analysis for Controlled Singing Voice Deepfake Detection," *Interspeech 2024*






----
| 구분          | SingFake                                               | 우리 프로젝트                                                   |
| ----------- | ------------------------------------------------------ | --------------------------------------------------------- |
| 프로젝트 성격     | singing deepfake detection을 위한 **데이터셋 및 benchmark 연구** | AI 커버곡 식별을 실제로 수행하는 **서비스형 시스템 개발**                       |
| 핵심 목표       | 인간 노래와 AI 노래를 구분할 수 있는 **탐지 문제 제안 및 성능 평가**            | 사용자가 업로드한 음원이 **AI 커버곡인지 아닌지 판별**하고 결과를 제공                |
| 입력 데이터      | 연구용으로 정리된 **노래 오디오 클립 데이터셋**                           | 사용자가 실제로 업로드한 **완성 음원(mp3, wav 등)**                       |
| 출력 형태       | bonafide / deepfake 분류 결과, benchmark 성능 수치             | AI 여부 판별 + **의심 구간 시각화 + 설명 가능한 분석 결과**                   |
| 활용 목적       | 연구자들이 모델 성능을 비교하는 **기준 데이터셋 제공**                       | 일반 사용자·창작자·플랫폼이 활용할 수 있는 **실용적 판별 도구**                    |
| 데이터 관점      | in-the-wild singing deepfake를 모아 **탐지 과제 자체를 정립**      | 선행연구를 바탕으로 **실제 사용 환경에 맞는 입력 처리 파이프라인** 구축                |
| 시스템 흐름      | 데이터셋 기반 오프라인 실험 중심                                     | 업로드 → 전처리 → 구간 분할 → 판별 → 결과 리포트까지 이어지는 **end-to-end 시스템** |
| 설명 가능성      | 주로 모델 성능 비교 중심                                         | 단순 판별을 넘어서 **어느 구간이 왜 의심되는지** 보여주는 방향                     |
| 저작권 보호와의 연결 | 직접적 활용보다는 탐지 연구 자체에 초점                                 | **AI 커버곡 식별을 통한 음성 저작권 보호 지원**에 초점                        |
| 최종 산출물      | 데이터셋, baseline, benchmark 결과                           | 탐지 모델 + 웹/앱 형태의 **분석 시스템** + 결과 리포트                       |
