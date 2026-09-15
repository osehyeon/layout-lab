# 00. B200 vs RTX PRO 6000 Blackwell vs RTX 5090 — 비교 요약

> 세부 근거와 출처는 각 문서에 있다: [01-architecture](01-architecture.md), [02-b200](02-b200.md), [03-rtx-pro-6000](03-rtx-pro-6000.md), [04-rtx-5090](04-rtx-5090.md).
> 이 문서의 수치는 모두 위 문서에서 가져왔다. 표기 규칙: 처리량은 **NVIDIA 공식 수치를 환산 없이** 적고 값마다 **[sparse]**(2:4 structured sparsity 적용, dense의 2배) 또는 **[dense]** 를 붙인다. 데이터시트·제품 페이지에 공식 표기가 있으면 그 값을, 없으면 백서 표의 dense 값을 쓴다. `(계산값)` = 공식 수치로부터 계산, `(미확인)` = 1차 출처로 확인 못함.
> 자료 수집일: 2026-09-11.

## 1. 한눈에 보기

| | **B200** (HGX B200) | **RTX PRO 6000 Blackwell** (Workstation) | **RTX 5090** |
|---|---|---|---|
| 포지션 | 데이터센터 학습/추론 | 워크스테이션·서버 (그래픽+AI) | 컨슈머 (게이밍/크리에이터) |
| Compute capability | **10.0 (`sm_100`)** | **12.0 (`sm_120`)** | **12.0 (`sm_120`)** |
| 다이 | 듀얼 다이(NV-HBI 10 TB/s), 208B TR, TSMC 4NP | GB202, 92.2B TR, 750 mm², TSMC 4N | GB202 (동일 다이) |
| SM | 148 (미확인, 3자 보도) | **188** | 170 (풀 다이 192) |
| 메모리 | **180 GB HBM3e** | **96 GB GDDR7 (ECC)** | 32 GB GDDR7 (on-die ECC 상시, 토글 없음) |
| 메모리 대역폭 | **7.7 TB/s** | 1,792 GB/s | 1,792 GB/s |
| L2 | (미확인) | 128 MB | 96 MB |
| 전력 | 최대 1,000 W (GB200에선 1,200 W) | 600 W (Max-Q 300 W, Server ≤600 W 설정형) | 575 W |
| 폼팩터 | SXM (HGX 8-GPU 베이스보드) | PCIe Gen5 x16 듀얼슬롯 | PCIe Gen5 x16 |
| 가격 | 비공개 (시스템 단위 판매) | 공식 MSRP 없음. 보도가 $8,565(2025-04) → $16,000(2026-08, NVIDIA US 스토어) | MSRP $1,999 (2025-01-30 출시) |

## 2. 정밀도별 처리량 (GPU 1개)

값은 공식 수치 그대로이며 **[sparse] 값은 dense의 2배**다.
- B200은 데이터시트가 Tensor 수치를 sparse로만 공개해서 Tensor 행이 모두 [sparse]다.
- PRO 6000과 5090은 FP4만 공식 헤드라인(AI TOPS)이 [sparse]이고, 나머지는 백서 표의 [dense] 값이다.
- **같은 행이라도 태그가 다르면 직접 비교하지 말 것.**

| 정밀도 | B200 | RTX PRO 6000 WS | RTX 5090 |
|---|---|---|---|
| FP4 (NVFP4/MXFP4) | **18 PF [sparse]** | **4,000 AI TOPS [sparse]** | **3,352 AI TOPS [sparse]** |
| FP8 (FP16 누산) | 9 PF [sparse] | 1,007.6 TF [dense] | 838 TF [dense] |
| **FP8 (FP32 누산)** | 9 PF [sparse] | **1,007.6 TF [dense]** | **419 TF [dense]** ← 반속 |
| FP16 (FP16 누산) | 4.5 PF [sparse] | 503.8 TF [dense] | 419 TF [dense] |
| **FP16/BF16 (FP32 누산)** | 4.5 PF [sparse] | **503.8 TF [dense]** | **209.5 TF [dense]** ← 반속 |
| TF32 | 2.2 PF [sparse] | 251.9 TF [dense] | 104.8 TF [dense] ← SM·clock당 반속 |
| INT8 | 9 POPS [sparse] | 1,007.6 TOPS [dense] | 838 TOPS [dense] |
| FP32 (non-tensor) | 75 TF [dense] | 126 TF [dense] | 104.8 TF [dense] |
| FP64 | **37 TF [dense]** (FP64 Tensor 동일) | ≈1.97 TF [dense] (1/64, 계산값) | ≈1.64 TF [dense] (1/64, 계산값) |

**꼭 알아둘 점**
- "Blackwell NVFP4 20 PF [sparse], 1,200 W"는 **GB200급** 수치다. HGX B200(1,000 W)은 18 PF [sparse].
- **GeForce 제한**: RTX 5090은 FP32 누산 FP16/BF16/FP8 GEMM을 FP16 누산의 **절반 속도**로 수행한다(NVIDIA GeForce 백서 Table 3). 같은 GB202인 PRO 6000은 full rate라서 BF16 학습 GEMM 이론치가 **약 2.4배**(503.8 vs 209.5 TF, 둘 다 [dense]) 차이 난다. FP4와 INT8은 제한되지 않는다 → FP4 추론에서는 5090과 PRO의 격차가 SM·clock 차이(약 1.2배) 수준으로 줄어든다.
- RTX PRO 6000 Max-Q(300 W)는 FP4 3,511 AI TOPS [sparse], Server Edition은 대역폭이 **1,597 GB/s**로 낮다. 자세한 에디션별 표는 [03 §2](03-rtx-pro-6000.md).

### 2.1 Roofline 관점 (ridge point = dense FLOPS ÷ 대역폭, FLOP/byte, 계산값)

| | B200 | PRO 6000 WS | RTX 5090 |
|---|---|---|---|
| BF16 (FP32 누산) | 292 | 281 | 117 |
| FP8 (FP32 누산) | 584 | 562 | 234 |
| FP4 | 1,169 | 1,124 | 935 |

- B200과 PRO 6000은 **연산/대역폭 비율이 거의 같다**. 절대 규모만 약 4.3~4.5배 차이 난다(대역폭 7.7/1.79 = 4.3배, BF16 dense 2,250(= 4.5 PF [sparse] ÷ 2)/503.8 = 4.5배). 따라서 PRO 6000에서 관찰한 "memory-bound인가 compute-bound인가"는 B200에서도 대체로 유지된다. 단, 커널 구조가 다르므로(§4) 효율은 다를 수 있다.
- RTX 5090은 BF16/FP8(FP32 누산)의 ridge가 절반 이하라서 같은 워크로드가 **더 일찍 memory-bound에서 벗어나 compute-bound가 된다**.
- 단일 배치 decode 상한(가중치 크기 ÷ 대역폭)은 PRO 6000과 5090이 같다(1,792 GB/s). 차이는 메모리 용량(96 vs 32 GB)에서 난다.

## 3. 기능 지원 비교

| 기능 | B200 | RTX PRO 6000 | RTX 5090 |
|---|---|---|---|
| 5세대 Tensor Core, NVFP4/MXFP4/MXFP6/MXFP8 | O | O | O |
| **`tcgen05.mma` / Tensor Memory(TMEM, 256 KB/SM)** | **O** | X | X |
| **2-CTA (CTA pair) MMA** | **O** | X | X |
| Block-scaled MMA 방식 | `tcgen05.mma ... block_scale` (비동기, TMEM 누산) | `mma.sync ... block_scale` (warp 동기, 레지스터 누산) | 좌동 |
| Shared memory / block | 227 KB | 99 KB | 99 KB |
| 최대 상주 warp / SM | 64 | 48 | 48 |
| Thread block cluster | 최대 8 (opt-in 16), TMA multicast O | 지원. TMA multicast는 PTX상 허용되나 NVIDIA 권장 대상 아님(실효 성능 (미확인)) | cluster 1×1×1 (multicast 없음, CUTLASS 기준) |
| 하드웨어 stochastic rounding FP4 변환 (`cvt.rs`) | O (sm_100a) | X | X |
| FP64 Tensor Core | O | X (FP64 1/64) | X (FP64 1/64) |
| NVLink | **NVLink 5, 1.8 TB/s/GPU** (NVSwitch 8-GPU) | X | X |
| PCIe P2P | (NVLink로 대체) | (미확인) | **X** (NVIDIA 포럼 2025-03, NCCL은 host 경유) |
| MIG | **최대 7** (1g.23gb) | **최대 4** (4×24 / 2×48 / 1×96 GB, "Universal MIG" = 그래픽+컴퓨트). WS/Max-Q는 compute 모드 전환 필요 | X |
| vGPU | (미확인) | **Server Edition만** (SR-IOV 48 VF) | X |
| ECC | O (HBM3e) | O (GDDR7 ECC) | on-die ECC 상시 (사용자 토글 없음) |
| Confidential Computing | **O** (TEE-I/O, NVLink 인라인 암호화, 멀티 GPU) | Server Edition, **단일 GPU만** | X (지원 표기 없음) |
| Decompression Engine (최대 800 GB/s) | O | X | X |
| RAS Engine | O | — | — |
| RT Core (4세대), neural shaders / Cooperative Vectors, DLSS 4 | X (그래픽 없음) | O | O |
| NVENC / NVDEC | (미확인) | 4 / 4 | 3 / 2 |
| 디스플레이 출력 | X | O (Server Edition은 없음) | O |

## 4. 핵심 차이: `sm_100` ≠ `sm_120`

- **sm_120은 sm_100의 하위 집합이 아니다.** 같은 "Blackwell"이라도 Tensor Core 프로그래밍 모델이 다르다.
  - `sm_100` (B200/B300): `tcgen05.mma` 단일 스레드 발행, 비동기, 누산기는 TMEM, 2-CTA MMA, TMA multicast, 대형 SMEM.
  - `sm_120` (RTX 5090/PRO 6000/GB10): Ampere/Ada식 warp-level `mma.sync` + Blackwell의 FP4/FP6/block-scale 데이터 타입 + TMA. CUTLASS 기준 block-scaled GEMM은 **TN 레이아웃만**.
- **바이너리 호환 없음**: `sm_100a`/`sm_100f` cubin은 RTX에서 돌지 않고, `sm_120a`/`sm_120f`는 B200에서 돌지 않는다. 배포 코드는 두 계열을 fatbin에 모두 넣고 런타임 dispatch해야 한다(예시 플래그: [01 부록 A](01-architecture.md#부록-a-빠른-참조--컴파일-플래그-예)).
- **실무 함의**: RTX 5090/PRO 6000에서 개발한 NVFP4 커널은 **수치 포맷, scale layout, 정확도 검증까지는 B200에 그대로 이어지지만, mainloop(파이프라인·타일·누산기 위치)는 B200용으로 재작성해야 한다.**

## 5. 소프트웨어 지원 (요약)

| 구성요소 | 최소 버전 | B200 (sm_100) | PRO 6000 / 5090 (sm_120) |
|---|---|---|---|
| CUDA / 드라이버 | 12.8 + R570 (Linux ≥570.26). `f` family target은 12.9+. 13.x는 ≥580 | O | O (Linux는 **open kernel module** 필수) |
| PyTorch | 2.7 + cu128 wheel | O | O |
| Triton | 3.3 | O | O |
| CUTLASS | 3.8 (SM100), 3.9 (SM120), 4.0+ CuTe DSL | O (가장 성숙) | O (TN만, cluster 1×1×1) |
| TensorRT | 10.8 (FP4) | O | O |
| TensorRT-LLM | — | O (주 타깃) | 부분 (FMHA cubin, MLA, NVFP4 MoE 제한 — 2차 출처) |
| Transformer Engine NVFP4 학습 | TE 2.x | O | (미확인) |
| FlashAttention | FA3 = Hopper 전용 | **FA4 O** | **FA4 X** (이슈 #2307 미해결) |
| DeepGEMM | CUDA 12.9+ | O | **X** |
| vLLM / SGLang / FlashInfer | — | O | SM120 경로 순차 추가 중, 일부 NVFP4는 Marlin W4A16 fallback (2차) |
| NCCL | 2.25.1 | O (NVLink 5) | O (PCIe, 5090은 P2P 없음) |

→ "Blackwell 지원"이라고 적힌 라이브러리도 **sm_100에 먼저 최적화**되어 있고, sm_120 경로는 늦게 오거나 빠져 있는 경우가 많다.

## 6. 선택 가이드

| 목적 | 추천 | 이유 |
|---|---|---|
| 대규모 사전학습, NVFP4 학습 레시피, 멀티 GPU 스케일링 | **B200** | NVLink 5, HW stochastic rounding, TE NVFP4 주 타깃, FP64 |
| tcgen05/TMEM 커널, FA4, DeepGEMM 계열 연구 | **B200** (유일) | sm_100 전용 ISA |
| 70B급 단일 GPU 추론, 대형 모델 LoRA/QLoRA, MIG 실험 | **RTX PRO 6000** | 96 GB, FP32 누산 full rate, MIG 4 |
| 그래픽+AI (neural rendering), 비디오+AI | **RTX PRO 6000 / RTX 5090** | RT core, NVENC, Cooperative Vectors (B200엔 없음) |
| sm_120 NVFP4 커널 개발, 7B–32B 양자화 추론, 저비용 프로토타이핑 | **RTX 5090** | $1,999, PRO 6000과 ISA 동일 |
| FP32 누산 BF16 학습 처리량 | PRO 6000 ≫ 5090 | 5090은 반속 |
| 멀티 GPU 텐서 병렬 | B200 ≫ PRO 6000 > 5090 | PRO/5090은 NVLink 없음, 5090은 P2P도 없음 |

## 7. 수치 사용 시 주의

- **B200 사전 스펙 vs 출하 스펙**: 2024-03 발표 당시 192 GB / 8 TB/s / PCIe Gen6 / FP64 40 TF → 출하 HGX B200은 **180 GB / 7.7 TB/s / Gen5 / 37 TF**. NVIDIA 페이지 간에도 DGX 합계 대역폭이 64 vs 62 TB/s로 다르다.
- NVIDIA의 세대 간 "N배" 주장은 대부분 **B200 FP4 vs Hopper FP8**처럼 정밀도가 다른 비교다.
- 서드파티 LLM 벤치마크(tok/s)는 동시성, 시퀀스 길이, 양자화, 엔진 버전에 따라 몇 배씩 달라진다. 각 문서의 "성능 참고" 절에 조건이 함께 적혀 있다.
- 공정 표기: 데이터센터 다이는 TSMC **4NP**, GB202는 NVIDIA 백서상 **4N**.
