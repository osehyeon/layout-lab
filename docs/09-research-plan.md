# 09. 연구 계획 — RTX 5090 + RTX PRO 6000(sm_120)으로 Blackwell 연구하기

> 작성일: 2026-09-11. 보유 장비: **RTX 5090 1장, RTX PRO 6000 Blackwell 1장**(둘 다 sm_120). B200(sm_100)은 없음.
> 이 문서는 00~08 문서의 조사 결과를 실행 계획으로 옮긴 것이다. 수치 표기 규칙(값마다 [dense]/[sparse])은 [README](README.md)를 따른다.

## 0. 기본 전략

1. **연구 주제를 sm_120에서 유리한 쪽으로 잡는다.** sm_120은 소프트웨어 공백이 크고(07 문서), 같은 다이 두 제품을 함께 가진 환경은 흔하지 않다.
2. **코드는 처음부터 B200으로 이식할 수 있게 짠다.** CuTe DSL / CUTLASS를 쓰고, `sm_120f`와 `sm_100f`를 모두 타깃한다.
3. **sm_100 전용 부분(tcgen05/TMEM)은 마지막에 클라우드 B200을 짧게 빌려 검증한다.**

---

## 1. 두 카드의 차이와 역할 분담

### 1.1 기능 차이

**ISA(명령어 수준)는 같다.** 둘 다 CC 12.0이라 같은 cubin이 돌고, 커널 코드도 그대로 옮겨진다. 차이는 **제품 구성, 드라이버, 플랫폼 기능** 쪽에 있다.

| 구분 | RTX 5090 | RTX PRO 6000 (Workstation 기준) | 연구에 주는 영향 | 근거 |
|---|---|---|---|---|
| SM / 클럭 | 170 SM, boost 2,407 MHz | 188 SM, boost 2,617 MHz | wave quantization, 타일 수 튜닝이 다름 | 03 §4.1 |
| 메모리 | 32 GB GDDR7 | **96 GB** GDDR7 | 모델 크기 상한 | 03 §4.1 |
| 대역폭 | 1,792 GB/s | 1,792 GB/s (Server Edition은 1,597) | decode 상한은 같음 | 03 §2.1 |
| L2 | 96 MB | 128 MB | 캐시 적중률 차이 | 03 §4.1 |
| **FP32 누산 FP16/BF16/FP8** | **FP16 누산의 절반 속도** | full rate | BF16 학습 이론치 **약 2.4배** 차이. FP4·INT8은 제한 없음 | 00 §2, GeForce 백서 Table 3 |
| TF32 | SM·clock당 PRO의 절반 | full rate | 동상 | 03 §4.1 |
| MIG | **없음** | **최대 4개**(4×24 / 2×48 / 1×96 GB). Workstation은 DisplayModeSelector로 compute 모드 전환 필요(디스플레이 꺼짐) | 멀티테넌트 실험은 PRO만 | 03 §3.1 |
| ECC | GDDR7 on-die ECC 상시 동작, 사용자 토글 없음 | ECC 지원(Server Edition은 기본 Enabled) | 장시간 학습 신뢰성 | 04 §2.1, 03 §2.1 |
| GPU 간 P2P(PCIe) | **미지원**(NVIDIA 포럼 답변) | NVIDIA 포럼상 "Quadro RTX and Data Center" 계열은 지원. RTX PRO 6000 Blackwell 실측은 (미확인) | 두 장을 묶으면 통신이 호스트 메모리 경유 | 04 §3.2 |
| TMA multicast | CUTLASS: "GeForce에는 multicast 없음" | (미확인) — PTX상 허용, 권장 대상은 아님 | cluster 최적화 실험의 전제 | 06 §3.2 |
| vGPU | 없음 | **Server Edition만** | 보유 에디션 확인 필요 | 03 §3.2 |
| Confidential Computing | 없음 | **Server Edition, 단일 GPU만** | 동상 | 03 §3.3 |
| NVENC / NVDEC | 3 / 2 | 4 / 4 | 비디오+AI 파이프라인 | 03 §4.1 |
| 드라이버 계열 | GeForce(Game Ready / Studio) | RTX Enterprise(pro) | 안정성, 인증 | 03 |
| 전력 | 575 W | 600 W (Max-Q 300 W) | 전력 효율 실험 | 03 §1.2 |

### 1.2 역할 분담

- **RTX PRO 6000 = 주력.** 큰 모델(70B급 FP8/NVFP4), 학습, MIG 실험, 최종 벤치마크를 여기서 한다. FP32 누산이 full rate라 BF16/FP8 수치를 대표값으로 쓸 수 있다.
- **RTX 5090 = 커널 개발·비교용.** 같은 커널을 먼저 짜 보고, GeForce 제한이 주는 영향을 측정하는 대조군으로 쓴다.
- **두 장 동시 사용**: P2P를 가정하지 않는다(5090 미지원). 멀티 GPU 실험은 "PCIe + 호스트 경유" 조건을 명시하고 진행한다.

---

## 2. sm_120 결과 중 B200까지 유효한 것

| B200까지 유효 (여기서 해도 됨) | sm_120에서 불가 (빼거나 B200 검증 단계로 미룸) |
|---|---|
| NVFP4/MXFP 수치 연구(포맷 동일) | `tcgen05` / TMEM / 2-CTA MMA 커널 |
| 알고리즘 수준 연구(attention 변형, MoE, speculative decoding) | TMA multicast, 대형 cluster |
| 파이프라인 구조: TMA, mbarrier, warp specialization, `setmaxnreg`, CLC, PDL (sm_120f에 모두 있음) | NVLink 스케일링, 대규모 사전학습 |
| CuTe/CUTLASS 추상화 코드(계산 루프만 교체) | HW stochastic rounding(`cvt.rs`), FP64 Tensor Core |

근거: [06-sm-features.md](06-sm-features.md) §3.2, [01-architecture.md](01-architecture.md) §2.3.

---

## 3. 단계별 계획

### Phase 0 — 환경 구축과 기초 측정 (1~2주)

목표: 문서에 **(미확인)**으로 남은 항목을 직접 측정해 기준 데이터를 만든다. 결과는 해당 문서(00/03/04/06/07)에 반영한다.

| # | 측정 항목 | 방법 | 두 카드 모두? |
|---|---|---|---|
| 0-1 | SM당 최대 상주 block 수: 문서마다 24와 32로 엇갈림 | `cudaDeviceGetAttribute(cudaDevAttrMaxBlocksPerMultiprocessor)` + occupancy API | O |
| 0-2 | FP32 누산 대 FP16 누산 GEMM 처리량 (BF16, FP16, FP8, TF32, FP4, INT8) | cuBLASLt(또는 CUTLASS profiler)로 큰 정사각 GEMM. 결과를 각 카드 [dense] peak 대비 %로 표시 | O |
| 0-3 | PCIe P2P 지원 여부와 대역폭 | `nvidia-smi topo -m`, CUDA samples `p2pBandwidthLatencyTest`, `nccl-tests` | 두 장 조합 |
| 0-4 | PRO 6000의 TMA multicast 실효 | cluster 2×1 multicast TMA vs 비multicast 마이크로벤치 (CUDA C++ / CuTe) | PRO 우선 |
| 0-5 | Triton FP8 반속 우회 | `tl.dot`(FP8, FP32 누산) vs unit-scale `tl.dot_scaled`(MXFP8) — #11320 재현 | O |
| 0-6 | 전력-성능 곡선 | `nvidia-smi -pl`로 전력 상한 스윕, decode/prefill 각각 tok/s·W | O |
| 0-7 | MIG 구성 | PRO 6000에서 compute 모드 전환 후 4×24 GB 생성, 인스턴스별 GEMM/서빙 처리량 | PRO |

메모리 쪽 측정 항목과 확인 메트릭은 [10-memory-model.md](10-memory-model.md) §6에 따로 정리했다. Phase 1의 커널 튜닝은 그 문서의 우선순위(파이프라이닝 → coalescing → 스필 → … → 뱅크 충돌) 순서를 따른다.

### Phase 1 — 방향 A: sm_120 FP4/FP8 커널 (핵심, 1~3개월)

- **문제**: 공개된 sm_120 NVFP4 GEMM 튜토리얼이 FP4 [dense] peak의 약 60%에 머문다(03 §5.3). FlashAttention-4는 sm_120에서 동작하지 않는다(FA4 이슈 #2307).
- **단계**:
  1. 기준선: cuBLASLt NVFP4/MXFP8, CUTLASS `79_blackwell_geforce_gemm`, CuTe DSL `blackwell_geforce` 예제의 % of peak 측정
  2. GEMM 최적화: 타일·stage 수를 99 KB SMEM에 맞추고, warp specialization(pingpong/cooperative)과 `setmaxnreg`, CLC persistent scheduling 효과를 분리 측정
  3. attention: FP8/NVFP4 prefill·decode 커널. FlashInfer의 SM120 경로와 비교
- **도구**: CuTe DSL(빠른 반복) → 필요 시 CUTLASS C++. 비교군으로 Triton `dot_scaled`, TileLang NVFP4
- **B200 이식 포인트**: 수치 포맷, scale layout, TMA 로딩, 파이프라인 구조는 유지한다. 계산 루프는 `tcgen05`로 교체하는 것을 전제로 모듈화한다.

### Phase 2 — 방향 B: 같은 다이 통제 비교 (Phase 0-2의 확장)

- **질문**: 이론 격차(BF16·FP8 FP32 누산 약 2.4배, FP4·INT8 약 1.2배, 대역폭 동일)가 실제 워크로드에서 얼마나 나타나는가?
- **워크로드**: LLM decode(대역폭 병목), prefill(연산 병목), LoRA 학습(BF16), FP4 추론
- **통제**: 같은 드라이버/CUDA/커널, 클럭 고정(`nvidia-smi -lgc`) 여부를 명시, 전력 상한 동일화 실험 포함
- **산출물**: "GeForce 제한이 실제 AI 워크로드에 주는 영향" 정량 보고

### Phase 3 — 방향 C: 저정밀도 수치 연구

- NVFP4 vs MXFP4 vs FP8 vs W4A16: PTQ/QAT 정확도, 모델 크기 의존성. 크기는 PRO 6000 기준 70B급까지.
- KV cache FP8/FP4의 long-context 품질
- **stochastic rounding**: sm_120에는 `cvt.rs`가 없으므로 소프트웨어 SR 구현. 비용과 수렴 영향을 측정하고, B200 HW SR과는 Phase 5에서 비교한다.
- 도구: TensorRT Model Optimizer, torchao, vLLM/SGLang, lm-eval

### Phase 4 — (선택) 시스템 연구

- PRO 6000 MIG 멀티테넌트 서빙: MIG vs MPS vs time-slicing, tail latency
- 5090 + PRO 6000 PCIe 멀티 GPU: TP/PP 분할과 통신 압축(0-3 결과 전제)
- 전력 효율: Max-Q급 300 W 상한 vs 600 W

### Phase 5 — B200 검증 (클라우드, 짧게)

- 전제: Phase 1~3 코드가 `sm_100f` 빌드를 통과하고 수치 검증이 끝난 상태
- 할 일: 계산 루프를 `tcgen05`/TMEM 버전으로 교체한 뒤 정확도 확인, % of peak 측정, SR(HW vs SW) 비교
- 비용 절감: 테스트 스크립트를 로컬에서 완성해 두고, 대여 시간에는 실행과 수집만 한다

---

## 4. 권장 소프트웨어 스택 (2026-09 기준)

| 구성요소 | 권장 | 비고 |
|---|---|---|
| 드라이버 | R580 이상. CUDA 13.4 신기능은 R615+ | Linux는 **open kernel module** 필수 (04 §3.4) |
| CUDA | 13.x (sm_120f family target은 12.9+) | cuTile은 13.1+ |
| PyTorch / Triton | 2.14 / 3.8.0 | cu128 이상 wheel |
| CUTLASS / CuTe DSL | 4.x (현행 4.8.0) | GeForce 예제: `examples/79_*`, `examples/python/CuTeDSL/cute/blackwell_geforce/` |
| TileLang | 0.1.14 | NVFP4만 native |
| 프로파일러 | Nsight Compute / Nsight Systems | |

**빌드 규칙**: 커널 fatbin에는 `sm_120f`(또는 block-scale 등 arch 전용 기능을 쓰면 `sm_120a`)와 `sm_100f`를 함께 넣는다. 예시는 [01 부록 A](01-architecture.md#부록-a-빠른-참조--컴파일-플래그-예). CuTe DSL은 `CUTE_DSL_ARCH`로 타깃을 지정한다.

---

## 5. 측정·보고 규칙

- 처리량 %는 **해당 카드의 [dense] peak** 대비로 적는다(예: PRO 6000 FP4 2,015.2 TFLOPS [dense], 5090 1,676 TFLOPS [dense]).
- 모든 결과에 다음을 기록한다: GPU(에디션 포함), 전력 상한, 클럭 고정 여부, 드라이버/CUDA/라이브러리 버전, 커밋.
- 5090과 PRO 6000 결과를 섞어 "sm_120 성능"으로 보고하지 않는다(FP32 누산 속도가 다름).
- 벤더 수치·서드파티 수치를 인용할 때는 조건을 함께 적는다(02~04 문서 방식).

## 6. 산출물과 성공 기준

| Phase | 산출물 | 성공 기준 (예시) |
|---|---|---|
| 0 | 측정 스크립트, 문서의 (미확인) 해소 | 0-1~0-7 결과를 문서에 반영 |
| 1 | sm_120 FP4/FP8 GEMM·attention 커널 | NVFP4 GEMM에서 공개 튜토리얼(약 60%)보다 높은 % of peak, cuBLASLt 대비 경쟁력 |
| 2 | 통제 비교 보고서 | 워크로드별 실측/이론 격차 표 |
| 3 | 수치 연구 결과 | 포맷별 정확도-속도 곡선, SR 영향 |
| 5 | B200 이식 검증 | 같은 알고리즘의 sm_100 버전 정확도 일치, % of peak |
