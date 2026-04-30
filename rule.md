# Assignment 3 Rules

기준 문서: `assignment3.pdf`

## 핵심

- 주제: `Neural Network Verification with Marabou`
- 대상 모델: `resources` 디렉토리 외부의 외부 모델 (소형 FC 또는 CNN 권장)
- 핵심 개념:
  - `SMT(Satisfiability Modulo Theories)` 기반 신경망 검증
  - `ℓ∞-ball` 입력 제약 하에서의 `robustness property` 검증
  - 검증 결과 `SAT` / `UNSAT` / `TIMEOUT` 해석
- 실행 파일: `test.py`
- 최종 제출물: `requirements.txt`, `test.py`, `report.pdf`, `README.md`, 모델 파일 또는 생성 스크립트

## 일정 / 제출

- 배포일: `Wednesday, April 29, 2026`
- 마감일: `Wednesday, May 13, 2026, 11:59 PM`
- 제출 방식: GitHub 업로드 후 Uclass 링크 제출
- 지각 제출: 시스템 / 이메일 모두 불가

## 프로젝트 구성

- `requirements.txt` 또는 `Dockerfile` 포함
- 외부 모듈 / Python dependency 전부 기록
- 코드 이해용 주석 포함
- `test.py` 포함
- `test.py` 실행 시 Marabou를 모델에 대해 실행하고 검증 결과를 보여줄 수 있어야 함
- 테스트 형식 자유, 기능과 타당성 확인 가능 상태

## Problem 1: Marabou 리소스 디렉토리 탐색

### 배경

- Marabou는 신경망의 속성을 형식적으로 증명하는 SMT 기반 검증 도구
- GitHub 저장소의 `resources` 디렉토리에는 샘플 모델, 데이터셋, 검증 쿼리 예제가 포함되어 있음
- 이 디렉토리를 탐색하여 Marabou가 지원하는 포맷과 기능을 파악하는 것이 목표

### 수행 내용

- [Marabou GitHub](https://github.com/NeuralNetworkVerification/Marabou)의 `resources` 디렉토리 탐색
- 아래 항목을 조사하여 **표 형식**으로 문서화

#### ① 제공 모델 목록

| 모델명 | 아키텍처 | 포맷 | 입력 크기 | 비고 |
|--------|----------|------|-----------|------|
| (예) acasxu_1_1 | FC | `.nnet` | 5 | ACAS Xu 항공 충돌 회피 |
| ... | ... | ... | ... | ... |

#### ② 포함된 데이터셋 / 입력 명세

| 파일명 | 형식 | 설명 |
|--------|------|------|
| (예) ... | `.csv` / `.npy` | ... |

#### ③ 검증 쿼리 예시

| 쿼리 파일 / 예제 | 검증 속성 | 결과 (SAT/UNSAT) |
|-----------------|-----------|-----------------|
| (예) ... | L∞ robustness | ... |

- 지원 포맷 확인: `.nnet`, `.onnx`, `.pb`

## Problem 2: 외부 모델로 Marabou 실행

### 배경

- Marabou의 실제 활용 능력을 검증하기 위해, `resources`에 없는 외부 모델에 직접 검증 쿼리를 적용
- 모델 변환, 쿼리 작성, 결과 해석까지 전 과정을 직접 수행해야 함

### 수행 내용

#### 1. Marabou 설치

- 저장소 클론 후 README 빌드 지침 따르기
- 제공된 예제 실행으로 설치 확인
- 설치 중 발생한 이슈 및 해결 방법 문서화

#### 2. 외부 모델 및 데이터셋 선택

- `resources` 디렉토리에 없는 모델 사용
- 작은 크기 권장 (단순 FC 네트워크 또는 소형 CNN)
- 지원 포맷(`.onnx`, `.nnet`, `.pb`)으로 변환 필요 시 전처리 수행

**모델 파일 제공 방식 (둘 중 하나 필수):**

| 방식 | 설명 | 예시 |
|------|------|------|
| 파일 직접 포함 | 변환된 모델 파일을 repo에 포함 | `model/my_model.onnx` |
| 생성 스크립트 제공 | 모델을 학습·저장하는 스크립트 포함 | `train_and_export.py` |

**데이터셋 / 샘플 입력 제공 방식 (둘 중 하나 필수):**

| 방식 | 설명 | 예시 |
|------|------|------|
| 샘플 파일 직접 포함 | 검증에 사용할 입력 샘플을 repo에 포함 | `data/sample_input.npy` |
| 다운로드 스크립트 제공 | 데이터셋 자동 다운로드 코드 포함 | `prepare_data.py` (torchvision 등 활용) |

> 채점자가 추가 데이터 준비 없이 `test.py`만 실행해도 재현 가능해야 함

**권장 모델/데이터셋 조합 예시:**

| 모델 | 데이터셋 | 포맷 | 비고 |
|------|----------|------|------|
| 3-layer FC (784→128→10) | MNIST | `.onnx` | 가장 간단, 권장 |
| 3-layer FC (4→8→3) | Iris | `.onnx` | 입력 차원 가장 작음 |
| 소형 CNN (2 conv) | Fashion-MNIST | `.onnx` | 검증 시간 주의 |

#### 3. 검증 쿼리 작성

- 입력 제약 정의: 특정 입력 주변 ℓ∞-ball, 반경 ε
- 출력 제약 정의: 섭동 범위 내에서 동일 클래스 예측 유지
- 예시: "digit 3으로 분류된 입력 x에 대해, ‖x̃ − x‖∞ ≤ 0.01 내 모든 입력도 digit 3으로 분류됨을 검증"

#### 4. 실행 및 결과 해석

- Python API(`maraboupy`) 또는 CLI로 검증 쿼리 실행
- 결과 기록: `UNSAT` 또는 `SAT`
- `SAT`인 경우 adversarial input 시각화 또는 설명
- 검증 시간 및 리소스 사용량 기록

**SAT/UNSAT 해석 시 주의사항:**

| 결과 | 의미 | 주의할 점 |
|------|------|-----------|
| `UNSAT` | 해당 쿼리 범위 내 반례를 찾지 못함 (전체 안전성 보장 아님) | ε이 너무 작으면 trivially UNSAT일 수 있음. ε을 점차 키워가며 실험할 것 |
| `SAT` | 속성을 위반하는 반례(adversarial input)가 존재함 | 반례가 입력 범위를 벗어나지 않는지 등 실질적 의미 확인 필요 |
| `TIMEOUT` | 제한 시간 내 결론 미도달 | 결과가 없는 것이지 UNSAT이 아님. 모델 축소 또는 ε 감소 필요 |

## 채점 / 제출 결과

### Problem 1: 탐색 보고서 (`20%`)

- Marabou `resources` 디렉토리 내 모델, 데이터셋, 검증 쿼리 예시 정리
- 표 형식으로 문서화

### Problem 2: 구현 및 결과 (`50%`)

#### 1. Code & reproducibility

- `test.py` 실행 가능
- clear setup instructions 포함
- 보고 항목:
  - 선택한 모델과 데이터셋, 선택 이유
  - 호환성을 위한 수정 / 전처리 내용
  - 검증 결과 (`SAT` / `UNSAT`), 실행 시간
  - `SAT`인 경우 반례 저장 (`results/adversarial_input.npy` 등)

#### 2. `test.py` 예상 출력 형식 (최소 요건)

```
[INFO] Loading model: model/my_model.onnx
[INFO] Input sample index: 42 | True label: 3
[INFO] Perturbation radius (ε): 0.01
[INFO] Running Marabou verification...

=== Verification Result ===
Status  : UNSAT          # 또는 SAT
Runtime : 12.4 seconds
Property: All inputs within L∞ ε=0.01 ball classified as label 3

# SAT인 경우 추가 출력:
Counterexample saved to: results/adversarial_input.npy
Predicted label on counterexample: 7
```

- 결과를 콘솔에 출력하고, 가능하면 `results/` 폴더에 저장
- `SAT`일 경우 반례 입력값을 파일로 저장 (`numpy` 배열 또는 이미지)
- 실행 시간은 반드시 측정하여 출력

### Report (`30%`)

- 분량: `1~2페이지` PDF
- 포함 내용:
  - 모델, 데이터셋, 검증 쿼리 설명
  - 결과 및 해석
  - Marabou의 장단점 본인 경험 기반으로 서술 (확장성, 사용 편의성, 지원 포맷 등)

## AI 사용 / Git 히스토리

- AI 도구 사용 가능
- 전제: genuine understanding 필요
- 저장소 요구:
  - meaningful commit history
  - single bulk commit 지양
  - git log inspect 가능성 고려
- 보고서 요구:
  - own interpretation and reasoning
  - generic description 지양
  - specific observations 포함 (예상치 못한 결과 포함)

## 최종 제출물

- `requirements.txt` 또는 `Dockerfile`
- `test.py`
- `report.pdf`
- `README.md`
- 모델 파일 (`.onnx` 등) 또는 생성 스크립트 (`train_and_export.py`)
- 샘플 입력 파일 (`data/sample_input.npy`) 또는 다운로드 스크립트

## README 요구사항

- Marabou 설치 및 실행 방법 포함
- 모델 변환 / 전처리에 가한 수정 사항 설명 포함