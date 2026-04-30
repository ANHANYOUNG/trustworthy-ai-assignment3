# Problem 1: Marabou Resources Directory 탐색

Marabou GitHub 저장소 `resources/` 디렉토리의 지원 포맷, 모델 목록, 입력 명세, 검증 쿼리 예시 정리.

---

## 디렉토리 구조

```
resources/
├── nnet/
│   ├── acasxu/       ACAS Xu FC 네트워크 45개
│   ├── coav/         무인기 회피 기동 벤치마크 500개
│   ├── mnist/        MNIST FC 네트워크 7개
│   └── twin/         Lipschitz 분석용 쌍둥이 네트워크 81개
├── onnx/
│   ├── acasxu/       acasxu/ ONNX 버전 45개 (.nnet과 동일 모델)
│   ├── cifar10/      CIFAR-10 분류기 6개 (base/deep/wide × 원본/간소화)
│   ├── layer-zoo/    레이어 단위 기능 테스트 모델 20+개
│   ├── vnnlib/       VNN-COMP 쿼리 파일 및 대응 ONNX 모델
│   └── (기타)        MNIST, CNN, FC 소형 모델 다수
├── keras/            Keras .h5 원본 모델 4개
├── properties/mnist/ MNIST ℓ∞-ball 쿼리 파일 81개
├── mps/              LP 문제 파일 2개
├── bnn_queries/      Binary Neural Network 쿼리 2개
└── target/           Targeted adversarial attack 쿼리 (.ipq)
```

---

## ① 제공 모델 목록

| 모델명 | 아키텍처 | 포맷 | 입력 크기 | 출력 크기 | 비고 |
|--------|----------|------|-----------|-----------|------|
| ACASXU_v2a_i_j (45개) | FC, 6 hidden × 50, ReLU | `.nnet`, `.onnx` | 5 | 5 | 항공 충돌 회피; i=1~5(이전 조언), j=1~9(속도 비율) |
| reluBenchmark* (500개) | FC, 5 hidden (6→40→40→38→19→1), ReLU | `.nnet` | 6 | 1 | 무인기 회피(COAV); 파일명에 검증 시간과 예상 결과 포함 |
| mnist10x10, mnist10x20 | FC, 10 hidden × 10/20, ReLU | `.nnet` | 784 | 10 | MNIST 소형 분류기 |
| mnist20x20, mnist20x40 | FC, 20 hidden × 20/40, ReLU | `.nnet` | 784 | 10 | MNIST 중형 분류기 |
| mnist2x256, mnist4x256, mnist6x256 | FC, 2/4/6 hidden × 256, ReLU | `.nnet` | 784 | 10 | VNN-COMP 2020 MNIST 벤치마크 출처 |
| twin_ladder-* (81개) | 동일 FC 네트워크 2개를 병렬 연결한 구조 | `.nnet` | 2/4/5 | 1 | Lipschitz 계수 추정 및 robustness 분석 목적 |
| fc1.onnx | FC, 2 hidden, ReLU | `.onnx` | 2 | 2 | abs(x)+abs(y), x²+y² 함수 근사 학습 |
| fc_2-2-3.onnx | FC, 2→2→3, ReLU | `.onnx` | 2 | 3 | 소형 3-class 분류기 단위 테스트용 |
| fc_2-2sigmoids-3.onnx | FC, 2→2→3, Sigmoid | `.onnx`, `.h5` | 2 | 3 | Sigmoid 활성화 지원 확인용 |
| KJ_TinyTaxiNet.onnx | CNN (소형) | `.onnx` | 8×16×1 | 2 | 활주로 중심선 이탈 예측 |
| cnn_max_mninst2/3.onnx | CNN + MaxPool | `.onnx`, `.h5` | 28×28×1 | 10 | MaxPool 포함 MNIST CNN |
| conv_mp1.onnx | Conv + MaxPool | `.onnx` | 가변 | 가변 | Conv+MaxPool 레이어 기능 테스트용 |
| mnist2x10.onnx | FC, 2 hidden × 10, ReLU | `.onnx` | 784 | 10 | 소형 MNIST |
| mnist2x5_sigmoid.onnx | FC, 2 hidden × 5, Sigmoid | `.onnx` | 784 | 10 | Sigmoid MNIST |
| mnist5x20_leaky_relu.onnx | FC, 5 hidden × 20, Leaky ReLU | `.onnx` | 784 | 10 | Leaky ReLU 지원 확인용 |
| cifar_base/deep/wide_kw.onnx (6개) | CNN (깊이·너비 조합) | `.onnx` | 32×32×3 | 10 | CIFAR-10; `_simp`는 간소화 버전 |
| robust_model_sigmoid_linear.onnx | FC, Sigmoid+Linear | `.onnx`, `.h5` | 784 | 10 | Adversarial training 적용 MNIST |
| model-german-traffic-sign-fast.onnx | FC/CNN | `.onnx` | 가변 | 43 | 독일 교통표지판 분류 |
| layer-zoo/* (20+개) | 레이어 단위 | `.onnx` | 가변 | 가변 | relu, sigmoid, tanh, batchnorm, conv, transpose 등 개별 레이어 테스트용 |
| vnnlib/* (5개) | 소형 FC | `.onnx` | 가변 | 가변 | VNN-COMP 표준 호환성 테스트용 |

---

## ② 포함된 데이터셋 / 입력 명세

실제 데이터셋 파일(.csv, .npy)은 없음. 입력은 검증 쿼리 파일 내 범위(bound) 형태로만 명세.

| 파일/경로 | 형식 | 설명 |
|-----------|------|------|
| `properties/mnist/image{1~3}_target{0~9}_epsilon{0.005,0.05,0.1}.txt` | 텍스트 쿼리 | MNIST 이미지 3장 × target 클래스 × ε 3종 조합, 총 81개 |
| `onnx/vnnlib/acasxu_prop1.vnnlib` | VNNLib (SMT-LIB2) | ACAS Xu 입력 5개의 정규화 범위 구간 명세 |
| `onnx/vnnlib/test_nano/tiny/small_vnncomp.vnnlib` | VNNLib | VNN-COMP 호환성 확인용 최소 입력 명세 |
| `target/mnist-bnn_index2_eps0.001_target9_unsat.ipq` | `.ipq` | MNIST index 2, ε=0.001 조건의 targeted attack 쿼리 |
| `fashion.ipq` | `.ipq` | Fashion-MNIST 검증 쿼리 |

**ACAS Xu 입력 변수**

| 변수 | 의미 | 단위 |
|------|------|------|
| ρ (X_0) | 침입기와의 거리 | feet |
| θ (X_1) | 침입기 방위각 (ownship 기준) | rad |
| ψ (X_2) | 침입기 heading 각도 | rad |
| v_own (X_3) | 자기 항공기 속도 | feet/s |
| v_int (X_4) | 침입기 속도 | feet/s |

출력 5개: COC(Clear of Conflict), Weak Left, Weak Right, Strong Left, Strong Right

---

## ③ 검증 쿼리 예시

| 쿼리 파일 | 검증 속성 | 포맷 | 결과 |
|-----------|-----------|------|------|
| `onnx/vnnlib/acasxu_prop1.vnnlib` | ρ∈[55947,60760] 등 특정 입력 범위에서 COC 출력 ≥ 1500 여부 | `.vnnlib` | SAT |
| `properties/mnist/image1_target1_epsilon0.005.txt` | image1 기준 ε=0.005 ℓ∞-ball 내에서 target=1 misclassify 가능 여부 | 텍스트 | 모델 의존 |
| `target/mnist-bnn_index2_eps0.001_target9_unsat.ipq` | BNN, image2, ε=0.001 내에서 target=9 유도 가능 여부 | `.ipq` | UNSAT (파일명에 명시) |
| `onnx/vnnlib/test_sat_vnncomp.vnnlib` | VNN-COMP SAT 케이스 | `.vnnlib` | SAT |
| `onnx/vnnlib/test_unsat_vnncomp.vnnlib` | VNN-COMP UNSAT 케이스 | `.vnnlib` | UNSAT |

**쿼리 포맷별 구조**

```
# .vnnlib (SMT-LIB2 기반, VNN-COMP 표준)
(declare-const X_0 Real)          ; 입력 변수 선언
(assert (>= X_0 0.6))             ; 입력 하한  ┐ ℓ∞-ball의
(assert (<= X_0 0.679857769))     ; 입력 상한  ┘ 한 차원
(assert (>= Y_0 3.991125))        ; 출력 제약 (UNSAFE 조건)

# .txt (Marabou 자체 쿼리 형식)
x0 >= -0.005    ┐ pixel 0에 대한 ±ε 범위
x0 <= 0.005     ┘ (원본 픽셀값 기준 offset)
x1 >= -0.005
x1 <= 0.005
...             ; 784개 픽셀 반복 → ℓ∞-ball
```

---

## ④ 지원 포맷

| 포맷 | 확장자 | 특징 | 비고 |
|------|--------|------|------|
| NNet | `.nnet` | Stanford 2016, 텍스트 기반, 정규화 범위 포함 | ReLU 전용; ACAS Xu 연구에서 주로 사용 |
| ONNX | `.onnx` | 현재 표준, 다양한 레이어·활성화 지원 | 복잡한 연산자(attention 등)는 미지원 가능 |
| Keras | `.h5` | Marabou 내부 파싱 지원 | 최신 버전에서는 ONNX 변환 권장 |
| VNNLib | `.vnnlib` | SMT-LIB2 기반 속성 기술 언어, VNN-COMP 표준 | 모델 파일과 쌍으로 사용 |
| MPS | `.mps` | LP 문제 명세 | 신경망 검증이 아닌 LP 직접 풀이용 |
| IPQ | `.ipq` | Marabou 독자 Incremental Property Query | 이식성 낮음 |

`.pb`(TensorFlow frozen graph)는 Marabou 문서에서 언급되나 `resources/` 내 예시 파일 없음.
