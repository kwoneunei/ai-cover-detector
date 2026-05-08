# Streamlit 웹 구현 가이드라인

## 1. 목표

웹사이트는 다음 흐름으로 동작하도록 구현한다.

1. 사용자가 음원 파일 업로드
2. 서버가 업로드 음원 전처리
3. 서버가 SingGraph 모델로 추론
4. clip 단위 결과를 곡 단위 결과로 집계
5. 최종 판별 결과와 점수를 화면에 표시

중요한 점은 가중치 파일은 사용자가 받는 것이 아니라 Streamlit 서버가 1회 다운로드해서 사용한다는 것이다.

---

## 2. 참고할 저장소

### 코드
- GitHub: https://github.com/kwoneunei/ai-cover-detector.git

### 가중치
- Hugging Face: eunei/ai-cover-detector-singgraph

팀원은 GitHub의 SingGraph 코드를 기준으로 웹을 만들고,
Hugging Face의 best.pth를 불러와 모델에 연결하면 된다.

---

## 3. 구현해야 할 구조

권장 위치는 SingGraph/app/ 아래다.

SingGraph/
  app/
    streamlit_app.py
    download_weights.py
    preprocess_full.py
    inference.py
    aggregate.py

### 각 파일 역할

download_weights.py
- Hugging Face에서 best.pth 다운로드
- 서버에 이미 있으면 재다운로드하지 않음

preprocess_full.py
- 업로드 음원 전처리
- 실제 사용한 전처리 흐름 반영
  - 보컬/반주 분리
  - VAD
  - clip 생성
  - 모델 입력 형태로 변환

inference.py
- SingGraph 모델 로드
- best.pth 불러오기
- clip 단위 추론 수행

aggregate.py
- clip별 점수를 곡 단위 점수로 합침
- 평균값 또는 최대값 방식 사용 가능

streamlit_app.py
- 웹 화면 구성
- 파일 업로드
- 분석 버튼
- 결과 표시

---

## 4. 전체 동작 흐름

### 앱 시작 시
1. 환경변수에서 Hugging Face repo 정보 읽기
2. download_weights.py로 best.pth 다운로드
3. 모델 로드
4. st.cache_resource로 캐싱

### 사용자가 파일 업로드 시
1. 업로드 파일 임시 저장
2. preprocess_full.py 실행
3. clip 단위 (vocals, non_vocals) 생성
4. inference.py로 clip별 추론
5. aggregate.py로 최종 점수 계산
6. 결과 표시

---

## 5. Streamlit에서 꼭 써야 할 방식

모델은 한 번만 로드해야 한다.

예:
@st.cache_resource
def load_engine():
    return SingGraphInference()

이렇게 해야 사용자마다 2.5GB 가중치를 다시 읽지 않는다.

---

## 6. 표시할 결과

웹에는 최소한 아래를 보여준다.

- 예측 결과: bonafide / deepfake
- 곡 단위 deepfake score
- 분석된 clip 수
- clip별 점수 목록 또는 그래프

추가 가능:
- 처리 시간
- 위험 구간 시각화
- 업로드 음원 재생

---

## 7. 환경변수

서버에서 아래 값을 설정해야 한다.

export HF_TOKEN="허깅페이스_토큰"
export HF_REPO_ID="eunei/ai-cover-detector-singgraph"
export HF_FILENAME="best.pth"

---

## 8. 팀원이 구현할 순서

1. GitHub 최신 코드 pull
2. SingGraph/app/ 폴더 생성
3. download_weights.py 작성
4. preprocess_full.py 작성
5. inference.py 작성
6. aggregate.py 작성
7. streamlit_app.py 작성
8. 로컬 테스트
9. 서버 배포

---

## 9. 중요한 주의사항

### 1) 사용자가 가중치를 받는 구조로 만들지 말 것
- 가중치는 서버가 받는다
- 사용자는 음원만 업로드한다

### 2) 앱에서 단순 리샘플링만 하지 말 것
- 실제 학습에 사용한 전처리와 최대한 비슷하게 맞춰야 한다
- 가능하면 보컬/반주 분리 + VAD + clip 생성 반영

### 3) 요청마다 모델 다시 로드하지 말 것
- 반드시 캐싱

### 4) 긴 음원 처리 시간 고려
- 전처리가 무거우므로 로딩 메시지 필요
- st.spinner() 사용 권장

---

# Streamlit에 가중치를 다운받을 수 있게 하면 서버가 커?

결론부터 말하면 네, 서버는 어느 정도 커야 합니다.

이유는 세 가지입니다.

### 1. 가중치 파일이 큼
best.pth가 약 2.5GB급이면,
서버 디스크에 그만한 공간이 필요합니다.

최소:
- 가중치 파일 2.5GB
- 캐시
- 임시 업로드 파일
- 분리된 오디오 파일

까지 감안해야 하므로 여유 디스크가 꽤 있어야 합니다.

### 2. 메모리도 필요함
모델 로드 시
- 가중치 메모리
- PyTorch 오버헤드
- 오디오 전처리 메모리

가 같이 들어갑니다.

그래서 RAM도 작으면 불안정합니다.

### 3. 전처리가 무거움
현재 파이프라인은 단순 추론이 아니라
- demucs
- pyannote VAD
- clip 생성

까지 포함하므로 CPU/GPU 자원을 꽤 씁니다.

---

## 현실적인 서버 권장

### 최소 실험용
- RAM: 16GB 이상
- 디스크: 20GB 이상
- CPU: 4코어 이상

### 권장
- RAM: 32GB 이상
- 디스크: 50GB 이상
- CPU: 8코어 이상
- 가능하면 GPU

### 왜 GPU가 좋냐
- 모델 추론
- demucs 분리

가 빨라집니다.

---

## Streamlit Cloud로 되냐
무겁습니다.
현재 구조 그대로면 Streamlit Cloud 같은 가벼운 호스팅은 불리합니다.

이유:
- 2.5GB 가중치
- demucs
- pyannote
- 큰 임시 파일

그래서 이런 서비스는 보통
- GPU 서버
- 고메모리 리눅스 서버

에서 돌리는 게 맞습니다.

---

## 더 현실적인 운영 방법

### 방법 1
웹 서버와 추론 서버를 분리
- Streamlit: UI만
- 별도 백엔드 서버: 전처리/추론

### 방법 2
초기 버전은 전처리를 단순화
- 보컬/반주 분리 생략
- 빠른 1차 판별만 제공

### 방법 3
가중치는 서버 시작 때 한 번만 받기
- 사용자 요청마다 다운로드 금지

---

## 최종 정리

팀원은 다음 원칙으로 만들면 된다.

- GitHub SingGraph 코드 기반으로 구현
- Hugging Face eunei/ai-cover-detector-singgraph에서 best.pth를 서버 시작 시 1회 다운로드
- 업로드 음원에 대해 전처리 → 추론 → 곡 단위 점수 집계
- 결과를 Streamlit 화면에 표시
- 모델은 st.cache_resource로 캐싱
- 사용자는 가중치를 다운로드하지 않음
- 서버는 2.5GB 가중치와 무거운 전처리를 감당할 수 있을 정도로 커야 함
