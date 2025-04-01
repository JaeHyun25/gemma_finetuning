# Gemma 모델 파인튜닝 프로젝트

이 프로젝트는 Gemma 모델을 LoRA를 사용하여 파인튜닝하는 코드를 제공합니다.

## 프로젝트 구조

```
gemma_finetuning/
├── config/
│   └── config.yaml      # 설정 파일
├── src/
│   ├── data_processor.py  # 데이터 처리 모듈
│   ├── model_utils.py     # 모델 관련 유틸리티
│   └── train.py          # 학습 스크립트
├── requirements.txt      # 의존성 패키지
└── README.md            # 프로젝트 설명
```

## 설치 방법

1. 저장소 클론
```bash
git clone [repository-url]
cd gemma_finetuning
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
.\venv\Scripts\activate  # Windows
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

1. 설정 파일 수정
- `config/config.yaml` 파일에서 필요한 설정을 수정합니다.
- 모델 경로, 학습 파라미터, 데이터 경로 등을 설정할 수 있습니다.

2. 학습 실행
```bash
python src/train.py
```

## 주요 기능

- 4비트 양자화를 통한 메모리 효율적인 학습
- LoRA를 사용한 효율적인 파인튜닝
- 데이터 증강을 통한 학습 데이터 다양화
- 상세한 로깅 및 에러 처리

## 설정 파일 설명

### 모델 설정
- `model.path`: 기본 모델 경로
- `model.output_dir`: 학습 중간 결과 저장 경로
- `model.final_output_dir`: 최종 모델 저장 경로

### 학습 설정
- `training.epochs`: 학습 에포크 수
- `training.batch_size`: 배치 크기
- `training.learning_rate`: 학습률
- 기타 학습 관련 파라미터

### LoRA 설정
- `lora.r`: LoRA 랭크
- `lora.alpha`: LoRA 알파
- `lora.target_modules`: LoRA 적용 대상 모듈

### 데이터 설정
- `data.input_file`: 입력 데이터 파일 경로
- `data.augmentation`: 데이터 증강 설정

## 주의사항

- 충분한 GPU 메모리가 필요합니다.
- 데이터셋은 CSV 형식이어야 하며, 'instruction', 'input', 'output' 컬럼이 필요합니다.
- 학습 전에 모델과 데이터셋이 올바른 경로에 있는지 확인하세요. 