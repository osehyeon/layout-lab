# 01. NVIDIA Blackwell 아키텍처 & 소프트웨어 스택

> 작성일: 2026-09-11 · 범위: B200 / RTX PRO 6000 Blackwell / GeForce RTX 5090이 공유하는 Blackwell 아키텍처와 SW 스택. 제품별 상세 사양은 각 GPU 문서를 참고.
> 표기 규칙: 1차 출처에서 확인하지 못한 수치는 **(미확인)**. 처리량은 NVIDIA 공식 수치를 **환산 없이** 적고 값마다 **[sparse]**(2:4 structured sparsity 적용, dense의 2배) 또는 **[dense]** 를 붙임. 데이터시트·제품 페이지 공식 표기가 있으면 그 값, 없으면 백서 표의 dense 값. 로컬 사본은 `docs/sources/architecture/` 에 있음([SOURCES.md](sources/architecture/SOURCES.md)).

---

## 0. TL;DR

1. "Blackwell" 안에는 **서로 다른 두 개의 SM ISA**가 있다. 데이터센터용 **sm_100 계열(B200/GB200=CC 10.0, B300/GB300=CC 10.3)** 과 클라이언트/워크스테이션용 **sm_120 계열(RTX 50, RTX PRO Blackwell=CC 12.0, DGX Spark GB10=CC 12.1)**. **sm_120은 sm_100의 상위집합이 아니다.**
2. sm_100 전용 기능: `tcgen05.*` 명령, **Tensor Memory (TMEM, 256 KB/SM)**, **2-CTA(cta_group::2) MMA**, 228 KB SMEM/SM, 비이식 cluster 16. TMA multicast는 PTX 문법상 sm_120에서도 허용되지만 NVIDIA 권장 대상이 아니고, CUTLASS는 GeForce에서 cluster를 1×1×1로 고정한다. sm_120은 tcgen05/TMEM/2-CTA가 모두 없고, **warp 단위 `mma.sync` 에 block-scaled FP4/FP6/FP8 확장(kind::mxf8f6f4 / mxf4 / mxf4nvf4)** 을 추가한 형태다(SMEM 최대 100 KB/SM, 99 KB/block).
3. 두 계열 모두 5세대 Tensor Core로 **FP4(NVFP4·MXFP4)/FP6/FP8(MX 포함)** 을 지원하므로 "NVFP4 추론"은 양쪽에서 가능하지만, **커널 바이너리와 커널 설계는 호환되지 않는다**(sm_100a용 커널은 sm_120에서 동작하지 않음).
4. 최소 SW: **CUDA 12.8**(sm_100/sm_120 최초 지원, 드라이버 R570), family target(`f`)은 **CUDA 12.9**, sm_103/sm_121도 12.9부터. **PyTorch 2.7 + cu128**, Triton 3.3, cuDNN·NCCL 2.25.1·TensorRT 10.8부터 Blackwell 지원.

---

## 1. Blackwell 제품군 개요

### 1.1 다이(die)와 공정

| 구분 | 다이 | 공정 | 트랜지스터 | 비고 |
|---|---|---|---|---|
| 데이터센터 (B200/GB200) | 2개의 reticle-size 다이 | **TSMC 4NP** | **208B** (다이 2개 합) | 두 다이를 **NV-HBI 10 TB/s** 로 연결해 하나의 CUDA GPU로 동작 ([Tech Brief](https://nvdam.widen.net/s/xqt56dflgh/nvidia-blackwell-architecture-technical-brief.pdf)) |
| 데이터센터 (B300/GB300, Blackwell Ultra) | 2 다이 | TSMC 4NP | 208B | 최대 160 SM, 288 GB HBM3e, NVFP4 15 PF [dense] ([Blackwell Ultra 블로그](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)) |
| 클라이언트/프로 **GB202** (RTX 5090, RTX PRO 6000) | 단일 다이 | **TSMC 4N** (NVIDIA custom) | **92.2B**, 750 mm² | Full: 12 GPC / 96 TPC / 192 SM, L2 128 MB ([RTX Blackwell WP](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf), [RTX PRO Blackwell WP](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf)) |
| GB203 (RTX 5080 등) | 단일 다이 | TSMC 4N | 45.6B, 378 mm² | 7 GPC / 42 TPC / 84 SM (RTX WP Appendix B) |
| GB205 (RTX 5070 등) | 단일 다이 | TSMC 4N | 31B | 192-bit 메모리 (RTX WP Appendix C) |

- 데이터센터 다이는 **4NP**, GeForce/RTX PRO 다이는 **4N**으로 whitepaper에 표기되어 있다(서로 다름에 유의).
- 데이터센터 Blackwell의 208B는 Hopper(80B) 대비 2.6x ([Ultra 블로그 Table 2](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)).

### 1.2 데이터센터 GPU당 처리량 (NVIDIA 1차 자료)

[Blackwell Ultra 블로그 Table 2](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)와 [Tech Brief GB200 Table 1](https://nvdam.widen.net/s/xqt56dflgh/nvidia-blackwell-architecture-technical-brief.pdf)(GPU 2개짜리 superchip 값을 GPU 수 2로 나눔 — sparsity 환산 아님):

| 항목 (GPU 1개) | Hopper (H100) | Blackwell (GB200 내 B200) | Blackwell Ultra (B300) |
|---|---|---|---|
| NVFP4 | – | 10 PF [dense] | 15 PF [dense] (같은 표의 sparse는 20 PF — 2배 관계가 아닌 예외) |
| FP8 | 2 PF [dense] | 5 PF [dense] | 5 PF [dense] |
| FP16/BF16 | – | 2.5 PF [dense] (GB200 superchip 5 PF [dense] ÷ GPU 2) | (미확인) |
| TF32 | – | 1.25 PF [dense] (superchip 2.5 ÷ 2) | (미확인) |
| INT8 | – | 5 POPS [dense] (superchip 10 ÷ 2) | tcgen05 `kind::i8` 미지원 (아래 2.4 참조) |
| FP64 (Tensor) | – | 45 TF [dense] (superchip 90 ÷ 2) | (미확인) |
| HBM | 80 GB / 3.35 TB/s | 192 GB HBM3e / 8 TB/s | 288 GB HBM3e / 8 TB/s |
| NVLink | 900 GB/s | 1.8 TB/s | 1.8 TB/s |
| 최대 TGP | 700 W | 1,200 W | 1,400 W |
| SFU EX2 (softmax용) | 4.5 TeraExp/s | 5 TeraExp/s | 10.7 TeraExp/s |

- 위 Blackwell 값은 **GB200(액체냉각, 최대 1.2 kW) 기준**이다. HGX/DGX B200(8-GPU, 공랭) SKU는 클럭·전력이 달라 수치가 낮을 수 있으니 B200 문서의 SKU별 표를 따를 것.
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)는 "B200 HBM3/HBM3e **최대 180 GB**", "GB200 L2 **126 MB**"로 적고 있어 블로그의 192 GB와 차이가 난다(제품 SKU별 가용 용량 차이로 추정, 확인 필요).

### 1.3 GeForce RTX 5090 참고치 (GB202, 170 SM)

[RTX Blackwell WP Appendix A](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) (Boost clock 2407 MHz 기준):

| Tensor 연산 | RTX 5090 |
|---|---|
| FP4 (FP32 accumulate) | 3352 AI TOPS [sparse] (제품 페이지 헤드라인) |
| FP8 (FP16 accumulate) | 838 TFLOPS [dense] |
| FP8 (FP32 accumulate) | 419 TFLOPS [dense] |
| FP16/BF16 (FP32 accumulate) | 209.5 TFLOPS [dense] |
| TF32 | 104.8 TFLOPS [dense] |
| INT8 | 838 TOPS [dense] |
| FP32 (non-Tensor) | 104.8 TFLOPS [dense] |

- GeForce RTX 5090은 **FP32 누산 시 FP8/FP16 Tensor 처리량이 FP16 누산의 절반**이다(WP 표 기준). RTX PRO/데이터센터 제품에는 이 제한이 없는 것으로 알려져 있으나 (미확인) — 제품별 문서의 사양표로 확인할 것. FP64는 SM당 FP64 코어 2개, **FP32의 1/64** (정확성 보장용) ([RTX WP](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)).
- RTX PRO 6000 Blackwell 수치는 [RTX PRO WP](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf) Appendix 참조(각 GPU 문서에서 다룸).

---

## 2. Compute Capability와 ISA 분기 — **sm_100 ≠ sm_120**

### 2.1 Compute capability 매핑

| CC / 타깃 | 제품 | 계열 | 최초 지원 CUDA | 출처 |
|---|---|---|---|---|
| **10.0** `sm_100` | B200, GB200 | 데이터센터 Blackwell | 12.8 | [CUDA GPUs](https://developer.nvidia.com/cuda-gpus), [CUDA 12.8 RN](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html) |
| **10.3** `sm_103` | B300, GB300, GB300 DGX Station | Blackwell Ultra | 12.9 | [CUDA Features Archive](https://docs.nvidia.com/cuda/pdf/CUDA_Features_Archive.pdf) |
| 10.7 `sm_107` | Rubin (참고: 차세대) | sm_100f 패밀리에 포함 | 13.4 (developer preview) | [CUDA 13.4 RN](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) |
| **11.0** `sm_110` (구 `sm_101`) | Jetson Thor | 임베디드 Blackwell | 12.8(sm_101) → 13.0에서 sm_110로 개명 | [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) ("sm_101a Renamed to sm_110a from PTX ISA 9.0") |
| **12.0** `sm_120` | GeForce RTX 50, RTX PRO Blackwell | 클라이언트/프로 Blackwell | 12.8 | [CUDA GPUs](https://developer.nvidia.com/cuda-gpus) |
| **12.1** `sm_121` | NVIDIA GB10 (DGX Spark) | sm_120f 패밀리 | 12.9 | [CUDA GPUs](https://developer.nvidia.com/cuda-gpus), [Features Archive](https://docs.nvidia.com/cuda/pdf/CUDA_Features_Archive.pdf) |

### 2.2 `a` / `f` suffix와 이식성

[CUDA Programming Guide – Compute Capabilities](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html), [NVIDIA 블로그(2025-05)](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/):

| 타깃 | 사용 가능한 기능 | 실행 가능한 GPU |
|---|---|---|
| `compute_100` / `sm_100` (suffix 없음) | 기본 기능 세트 (arch-specific 기능 불가) | PTX는 10.0 이상 모든 GPU(JIT), cubin은 10.x |
| `compute_100f` (family, **CUDA 12.9+**) | 패밀리 공통 arch-specific 기능 | **CC 10.0, 10.3, 10.7** |
| `compute_103f` | | 10.3, 10.7 |
| `compute_107f` (CUDA 13.4, Rubin 프리뷰) | | 10.7 |
| `compute_110f` | | 11.0 |
| `compute_120f` | | **CC 12.0, 12.1** |
| `compute_121f` | | 12.1 |
| `compute_100a` / `sm_100a` | 해당 CC의 **모든** arch-specific 기능 | **CC 10.0 전용** — 앞/뒤 호환 없음 |

- 포함 관계: `compute_100a` ⊃ `compute_100f` ⊃ `compute_100` (Programming Guide).
- **`sm_100f` 와 `sm_120f` 는 서로 다른 패밀리**이다. 따라서 tcgen05/TMEM 기반 커널(`sm_100a`/`sm_100f`)은 RTX 5090/RTX PRO 6000에서 실행되지 않는다. [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)도 "`sm_100a`/`compute_100a` 로 만든 PTX는 forward/backward 호환되지 않는다"고 명시한다.
- 실무 권장: 각 타깃의 cubin(`sm_100a`, `sm_103a`, `sm_120a` 또는 `sm_100f`, `sm_120f`)을 fatbin에 넣고, 이식용 PTX(`compute_100` 등 suffix 없는 것)를 추가한다. 직접 PTX를 쓰면 `__CUDA_ARCH_SPECIFIC__` / `__CUDA_ARCH_FAMILY_SPECIFIC__` 매크로로 fallback 경로를 둔다.
- PyTorch 확장 빌드 예: `TORCH_CUDA_ARCH_LIST="10.0a;12.0a"` 식으로 **두 계열을 모두 명시**해야 한다(표기법은 버전별 차이가 있음, (미확인)).

### 2.3 기능 비교: sm_100 vs sm_120

| 기능 | sm_100 / sm_103 (B200/B300) | sm_120 / sm_121 (RTX 50, RTX PRO, GB10) | 근거 |
|---|---|---|---|
| MMA 명령 | **`tcgen05.mma`** (single-thread 발행, 비동기, 결과는 TMEM) | **`mma.sync.aligned` (warp 단위, 결과는 레지스터)** + `.kind::f8f6f4`, `.kind::mxf8f6f4/mxf4/mxf4nvf4 .block_scale` 확장 | [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), [CUTLASS 문서](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md) |
| Hopper `wgmma` | 없음 (tcgen05로 대체) | 없음 | CUTLASS 문서 |
| Tensor Memory (TMEM) | **있음**: CTA당 128 lane × 512 column × 32-bit = **256 KB/SM** | **없음** | PTX ISA ("sm_100a/sm_100f … 512 columns and 128 rows per CTA"), [Ultra 블로그](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) |
| 2-CTA (CTA pair) MMA | **`cta_group::2`** — 인접 SM 2개가 하나의 MMA를 협동 수행 (예: 256×N 타일) | 없음 | CUTLASS `KernelTmaWarpSpecialized2SmSm100` |
| Thread block cluster | 지원. portable 최대 8, **B200은 opt-in으로 16** | 지원(CC 9.0+ 공통). 단 GeForce는 **multicast가 없어 CUTLASS가 cluster 1×1×1로 고정** | [Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html), CUTLASS 문서 |
| TMA | 있음 (+multicast, 권장 대상) | 있음. multicast는 PTX상 허용되나 권장 대상 아님, CUTLASS는 GeForce "no multicast" | Programming Guide Table 29, PTX ISA, CUTLASS 문서 |
| SMEM 최대 / SM | **228 KB** (통합 L1/SMEM 256 KB) | **100 KB** (통합 L1/SMEM 128 KB) | Programming Guide Table 31/32 |
| SMEM 최대 / block | 227 KB | **99 KB** | 동 |
| 최대 상주 warp / SM | 64 | **48** | Tuning Guide |
| 레지스터 파일 / SM | 64K × 32-bit (256 KB) | 동일 | Tuning Guide, RTX WP |
| FP32:FP64 | FP64 Tensor Core 지원 (B200) | 1/64 (호환성용) | Programming Guide Table 30, RTX WP |
| Block-scaled 레이아웃 | TN/NT/NN/TT (타입에 따라 다름) | **TN만** | CUTLASS 문서 |
| NVFP4/MXFP4 처리량 (상대) | Hopper FP8 대비 4x (`mxf4`, `mxf4nvf4`) | Ada FP8 대비 2x (FP32 누산 시 4x로 표기) | CUTLASS 문서 |
| 하드웨어 stochastic rounding 변환 (`cvt.rs` → e2m1x4 등) | sm_100a, sm_103a, sm_107a | **표에 없음** | PTX ISA `cvt` target notes |

> 결론: **sm_120 = "Hopper 이전(Ampere/Ada) 스타일 warp-level MMA + Blackwell의 FP4/FP6/block-scale 데이터 타입 + TMA"**, **sm_100 = "tcgen05 + TMEM + 2-CTA 기반의 완전히 새로운 비동기 Tensor Core 프로그래밍 모델"** 이다. 같은 NVFP4 GEMM이라도 커널 구조(파이프라인, 누산기 위치, 타일 크기, SMEM 예산)가 완전히 다르다.

### 2.4 계열 내부 차이 (sm_100 vs sm_103, Thor, GB10)

- **sm_103 (B300)**: PTX ISA상 `tcgen05.mma .kind::i8` 은 `sm_100a`, `sm_110a` 에서만 지원되고 family target(`sm_100f`)에서도 "except .kind::i8"로 제외된다 → **B300에는 tcgen05 INT8 MMA가 없다**. 한편 Programming Guide 13.4 Table 33에서 CC 10.3 행이 10.0보다 하나 적은 항목은 INT8이 아니라 **FP64**다(INT8은 "Yes"). 즉 **B300에는 FP64 Tensor Core가 없고**, Table 30의 FP32:FP64 비율도 **64:1**이다(B200은 2:1). INT8은 PTX(tcgen05 `kind::i8` 없음)와 Table 33(INT8 Yes)이 달라서, tcgen05가 아닌 경로(`mma.sync` 등)로 지원되는 것으로 추정한다(미확인). HGX B300 페이지의 INT8 합계 3 POPS, FP64 합계 10 TFLOPS와도 일치하는 방향이다. 상세는 [06-sm-features.md](06-sm-features.md) 참조.
- **`.scale_vec::1X/2X/4X`** 는 `sm_100a` 전용이고, 패밀리 이식용으로는 `.block16/.block32` 표기를 써야 한다(PTX ISA).
- **Jetson Thor (sm_110)**: tcgen05/TMEM을 **지원**(PTX target notes에 sm_110a/sm_110f 포함) — 즉 임베디드 Blackwell은 데이터센터 쪽 ISA를 따른다.
- **DGX Spark GB10 (sm_121)**: sm_120f 패밀리 → tcgen05/TMEM **없음**. RTX 50/RTX PRO용 커널(`sm_120f`)이 그대로 동작.

---

## 3. 5세대 Tensor Core와 2세대 Transformer Engine

### 3.1 지원 정밀도

| 정밀도 | sm_100 (B200) | sm_120 (RTX 50 / RTX PRO) | 비고 |
|---|---|---|---|
| FP64 (Tensor) | O (**B300/CC 10.3은 X**, Table 33) | 최소한만(정확성용) | RTX WP: "very minimal number of FP64 Tensor Cores" |
| TF32 | O | O | |
| FP16 / BF16 | O | O | |
| FP8 E4M3 / E5M2 | O | O | |
| FP6 (E3M2 / E2M3) | O | O | Blackwell 신규 |
| FP4 E2M1 | O | O | Blackwell 신규 |
| MXFP8 / MXFP6 / MXFP4 (UE8M0 scale, 32개당) | O | O | OCP MX 호환 |
| NVFP4 (UE4M3 scale, 16개당) | O | O | NVIDIA 고유 |
| INT8 | O (tcgen05 `kind::i8`은 `sm_100a` 전용 → B300은 tcgen05 INT8 없음. 단 Table 33은 10.3도 INT8 "Yes") | O | §2.4 참조 |

출처: [Programming Guide Table 33](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html), [RTX WP](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) ("RTX Blackwell Tensor Cores support FP16, BF16, TF32, INT8, and Hopper's FP8 Transformer Engine … adds FP4 and FP6"), [CUTLASS 문서](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md).

### 3.2 `tcgen05.mma` 종류 (sm_100)

[CUTLASS 문서](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md) 기준(처리량은 Hopper 대비 상대치):

| `kind` | 입력 | 상대 처리량 |
|---|---|---|
| `tf32` | TF32 | 2x Hopper TF32 |
| `f16` | FP16/BF16 | 2x Hopper FP16 |
| `i8` | INT8/UINT8 | 2x Hopper INT8 |
| `f8f6f4` | FP4/FP6/FP8 혼합 (scale 없음) | 2x Hopper FP8 |
| `mxf8f6f4.block_scale` | MXFP4/6/8 혼합 | 2x Hopper FP8 |
| `mxf4.block_scale` | MXFP4 | 4x Hopper FP8 |
| `mxf4nvf4.block_scale.scale_vec_size::[2X\|4X]` | MXFP4 또는 NVFP4 | 4x Hopper FP8 |

모든 kind에 `.sp`(2:4 sparse) 변형이 있으며, 모두 `cta_group::1|2`를 지원.

### 3.3 Micro-tensor scaling과 NVFP4 vs MXFP4

2세대 Transformer Engine은 "micro-tensor scaling"이라 부르는 세분화된 스케일링으로 FP4 AI를 가능하게 한다([Tech Brief](https://nvdam.widen.net/s/xqt56dflgh/nvidia-blackwell-architecture-technical-brief.pdf)).

| 항목 | FP4 (E2M1) | **MXFP4** (OCP MX) | **NVFP4** |
|---|---|---|---|
| 원소 | E2M1 (부호1, 지수2, 가수1), 표현 범위 약 ±6 | E2M1 | E2M1 |
| 블록 크기 | – | **32** 원소 | **16** 원소 |
| 블록 scale 형식 | – | **UE8M0** (2의 거듭제곱) | **FP8 E4M3** (분수 스케일 가능; 하드웨어 타입명 `ue4m3`) |
| 2단계 scale | – | 없음 | **per-tensor FP32** scale |
| 유효 비트/원소 | 4 | 4 + 8/32 = 4.25 | 4 + 8/16 = **4.5** |

- NVFP4는 FP16 대비 약 3.5x, FP8 대비 약 1.8x 메모리 절감. DeepSeek-R1-0528 기준 FP8 대비 정확도 저하 1% 이하 ([NVFP4 블로그](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)).
- E4M3 scale은 범위가 좁아서(최대 448) per-tensor FP32 scale로 텐서 분포를 먼저 맞춘 뒤 블록 scale을 인코딩한다. MXFP4의 E8M0는 per-tensor scale이 필요 없지만 amax 근처 양자화 오차가 커질 수 있다(NVFP4 블로그).
- MX 포맷 정의: [OCP MX v1.0 스펙](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) (로컬 미보관, [arXiv 2310.10537](https://arxiv.org/abs/2310.10537)로 대체).
- 하드웨어 레벨 스케일 벡터: MX 타입은 32개(sparse 64개)당 `float_ue8m0_t`, NV 타입은 16개(sparse 32개)당 `float_ue4m3_t`. K 방향으로 SFA(M×⌈K/32⌉), SFB(N×⌈K/32⌉) 행렬을 둔다(CUTLASS 문서).

---

## 4. 메모리 계층

| 계층 | sm_100 (B200) | sm_120 (GB202) | 출처 |
|---|---|---|---|
| 레지스터 | 64K × 32-bit / SM, 스레드당 최대 255 | 동일 (SM당 256 KB) | Tuning Guide, RTX WP |
| **TMEM** | **256 KB/SM** (128 lane × 512 col × 32-bit), `tcgen05.alloc` 으로 warp가 32-column 단위 할당, `tcgen05.ld/st/cp` 로 이동, MMA 누산기 보관 | 없음 | PTX ISA, Ultra 블로그 |
| L1 + SMEM (통합) | 256 KB / SM; SMEM carveout 0–228 KB | 128 KB / SM; SMEM carveout 0–100 KB | Programming Guide Table 32, Tuning Guide |
| 정적 SMEM 한도 | 48 KB (단 `sm_100a/103a` 는 정적 228 KB까지) | 48 KB (`sm_120a/121a` 는 100 KB까지) | PTX ISA §5.1.7 |
| L2 | GB200 126 MB | Full GB202 128 MB, RTX 5090 96 MB | Tuning Guide, RTX WP |
| DRAM | HBM3e, B200 8 TB/s (GB200 superchip 16 TB/s), 192 GB (B300 288 GB) | GDDR7: RTX 5090 32 GB, 512-bit, 28 Gbps → 1.792 TB/s; RTX PRO 6000 96 GB, 1.792 TB/s | Ultra 블로그, RTX/RTX PRO WP |
| TMA | bulk tensor copy + multicast | bulk tensor copy (multicast 없음) | CUTLASS 문서 |
| Cluster / DSMEM | 지원, portable 8 / 비이식 16 (B200) | 지원 (CUTLASS GEMM은 1×1×1) | Tuning Guide, CUTLASS |

- DSMEM은 L2 접근과 동시에 쓸 수 있어 SM 간 통신 대역폭을 늘릴 수 있고, 32-byte 정렬·coalesced 접근을 권장(Tuning Guide).
- GDDR7은 PAM3 신호(1.5 bit/cycle), GDDR7 제품은 ECC(SEC)와 EDR 지원(RTX WP v1.1).

---

## 5. 기타 기능

### 5.1 데이터센터 (GB100 계열)

| 기능 | 내용 | 출처 |
|---|---|---|
| **NVLink 5** | GPU당 18 링크, **1.8 TB/s 양방향** (방향당 900 GB/s), NVLink 4 대비 2x | [Tech Brief](https://nvdam.widen.net/s/xqt56dflgh/nvidia-blackwell-architecture-technical-brief.pdf) |
| **NVLink Switch** | 72-GPU NVLink 도메인에서 130 TB/s, 최대 576 GPU 확장, SHARP FP8 지원 | Tech Brief |
| NVLink-C2C | Grace–GPU 900 GB/s 코히런트 | Tech Brief |
| PCIe | **Gen 6 ×16** (256 GB/s 양방향) — Blackwell/Blackwell Ultra | [Ultra 블로그](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/), Tech Brief(GB200: "2x 256 GB/s Gen6") |
| **Decompression Engine** | 최대 **800 GB/s** 압축 해제, DB 쿼리/Spark 가속용. 지원 포맷(LZ4/Snappy/Deflate 등) (미확인) | Tech Brief |
| **RAS Engine** | 조기 결함 예측·진단용 전용 엔진 | Tech Brief |
| **Confidential Computing** | 업계 최초 **TEE-I/O** 가능 GPU, NVLink inline 보호, 비암호 모드와 거의 동일 처리량 | Tech Brief |
| MIG | GPU당 최대 7 인스턴스 (GB200: "2x7") | Tech Brief |
| 미디어 | GB200 superchip: 2×7 NVDEC, 2×7 NVJPEG | Tech Brief |
| DRAM 암호화, Reduced Bandwidth Mode | CUDA 12.8 NVML에 query/control 추가 | [CUDA 12.8 RN](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html) |
| Blackwell Ultra 추가 | attention용 SFU(EX2) 처리량 2x, 개선된 GigaThread Engine | Ultra 블로그 |

### 5.2 RTX 쪽 (GB20x)

| 기능 | 내용 | 출처 |
|---|---|---|
| **4세대 RT Core** | Triangle Cluster 교차 엔진, **Mega Geometry** 지원, 3세대 대비 ray-triangle 교차율 2x, Linear Swept Spheres | [RTX WP](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) |
| **Neural Shaders / Cooperative Vectors** | 셰이더 안에서 작은 신경망 실행. Cooperative Vectors API(Slang 등과 결합; 지원 그래픽 API 목록은 (미확인))로 셰이더에서 Tensor Core 직접 사용. SER도 신경망 셰이딩 가속 | RTX WP |
| **DLSS 4 Multi Frame Generation** | 렌더 프레임 1장당 AI 프레임 최대 3장 추가 생성("up to three additional frames per every traditionally rendered frame"), DLSS 3/3.5 대비 최대 2x 프레임률 | RTX WP |
| AI Management Processor (AMP) | AI 모델과 그래픽 작업의 GPU 스케줄링 | RTX WP |
| **9세대 NVENC** | AV1/HEVC 품질 5% 개선, 4:2:2 지원. RTX 5090: 3× NVENC, RTX PRO 6000 WS: 4× NVENC | RTX WP, RTX PRO WP |
| **6세대 NVDEC** | H.264 디코드 2x. RTX 5090: 2× NVDEC, RTX PRO 6000: 4× NVDEC | RTX WP, RTX PRO WP |
| PCIe | **Gen 5** | RTX WP |
| DisplayPort | 2.1b (UHBR20, 최대 80 Gbps) | RTX WP |
| 메모리 | **GDDR7** (데이터센터는 HBM3e) | RTX WP |
| MIG | **RTX PRO Blackwell은 MIG 지원** (구성표는 RTX PRO WP Table 3). GeForce는 해당 없음 | RTX PRO WP |

---

## 6. 소프트웨어 스택 지원 매트릭스

> 표의 "sm_100"은 B200/GB200(및 대부분 B300), "sm_120"은 RTX 50/RTX PRO Blackwell(및 GB10 sm_121) 기준. **1차 출처**와 **커뮤니티(2차)** 를 구분했다.

| 구성요소 | 최소/권장 버전 | sm_100 (B200) | sm_120 (RTX 5090 / RTX PRO 6000) | 출처 |
|---|---|---|---|---|
| **CUDA Toolkit** | **12.8** (sm_100/101/120 최초), **12.9** (sm_103, sm_121, `f` family target), 13.0 (sm_101→sm_110 개명, pre-Turing 제거). 현행 13.4 | O | O | [12.8 RN](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html), [Features Archive](https://docs.nvidia.com/cuda/pdf/CUDA_Features_Archive.pdf) |
| **드라이버** | CUDA 12.8 GA: Linux **≥570.26** / Windows ≥570.65. CUDA 12.9 GA: ≥575.51.03. CUDA 13.x: ≥580 (minor-version compat), 13.4 신기능은 R615+. 13.4부터 Linux 드라이버가 Toolkit에 번들되지 않음 | | | [12.8 RN](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html), [13.4 RN](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) |
| **PyTorch** | **2.7.0 + cu128 wheel** 이 Blackwell 최초 공식 지원 (당시 "[Prototype]" 표기) | O | O | [PyTorch 2.7 블로그](https://pytorch.org/blog/pytorch-2-7/) |
| **Triton** | **3.3** (PyTorch 2.7 동봉)부터 Blackwell 지원 | O | O (tcgen05 경로는 sm_100 전용이므로 sm_120 성능 특성은 다름 — (미확인)) | PyTorch 2.7 블로그 |
| **cuBLAS / cuBLASLt** | 12.8: Blackwell 지원 + block-scaled FP4/FP8 matmul API ("CC 10.0 and higher") | O (NVFP4/MXFP8, Grouped GEMM NVFP4는 CC 10.x/11.0) | O (GeForce FP8 관련 버그 수정 이력 있음, DGX Spark MXFP8/NVFP4 최적화) | [12.8 RN](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html), [13.4 RN](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) |
| **cuDNN** | 현행 9.26.0: CUDA 13.x 패키지가 CC 10.0/10.3/10.7/12.0/12.1 지원 (Linux 드라이버 ≥615.71.09). Blackwell 최초 지원 버전 (미확인) | O | O | [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html) |
| **CUTLASS** | **3.8.0** (2025-01): SM100 (tcgen05, TMEM, 2SM) 도입. SM120 GeForce 커널은 CHANGELOG상 3.9.0에 등장(문서는 "CUTLASS 4.0 has added support"로 표기). **4.0.0** (2025-06): **CuTe DSL**(Python). 4.4.2: SM120f 컴파일. 현행 4.8.0 (2026-08-25) | O (가장 성숙) | O (`examples/79_blackwell_geforce_gemm`, TN만, cluster 1×1×1) | [CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md), [CUTLASS 문서](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md) |
| **TensorRT** | **10.8**: Blackwell 지원(GeForce 50 정식; B200/GB200은 early access), E2M1 FP4(explicit quantization) | O | O | [TRT 10.8 RN](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.8.0.html) |
| **TensorRT-LLM** | 정식 버전 번호 (미확인) | O (주 타깃, trtllm-gen 커널) | 부분 지원: trtllm-gen FMHA cubin이 SM120/121용으로 없음, MLA 모델·NVFP4 W4A16 MoE 제한 (2차) | [TRT-LLM #11799](https://github.com/NVIDIA/TensorRT-LLM/issues/11799) |
| **Transformer Engine** | NVFP4 학습 레시피 `NVFP4BlockScaling` (TE 2.x). 버전 하한 (미확인) | O (학습 레시피의 주 타깃) | NVFP4 학습 공식 지원 여부 (미확인). stochastic rounding용 `cvt.rs` 는 PTX상 sm_100a/103a/107a만 | [TE NVFP4 문서](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html), PTX ISA |
| **FlashAttention** | FA2: README상 Ampere/Ada/Hopper 명시(Blackwell 공식 표기 없음 — sm_120 동작 여부 (미확인)). **FA3: Hopper 전용** (H100/H800, CUDA ≥12.3). **FA4 (CuTeDSL, `pip install flash-attn-4`)**: Hopper + Blackwell(B200) | FA4 O (SM100/SM103) | **FA4 X** (TMEM/tcgen05 의존; 이슈 #2307 2026-03 개설, 미해결) | [FA README](https://github.com/Dao-AILab/flash-attention), [#2307](https://github.com/Dao-AILab/flash-attention/issues/2307) |
| **cuDNN attention / FlashInfer** | FlashInfer: SM100용 trtllm-gen 경로 + SM120용 별도 경로(XQA NVFP4 KV 커널은 SM120 전용). SM120 `mm_fp4` 실패 이슈 존재 (2차) | O | 부분 | [FlashInfer #2577](https://github.com/flashinfer-ai/flashinfer/issues/2577) |
| **vLLM / SGLang** | Blackwell 지원 버전 하한 (미확인) | O | NVFP4 MoE/GEMM·NVFP4 KV cache 등 SM120 경로가 순차적으로 추가 중. 일부 NVFP4 체크포인트가 Marlin W4A16으로 fallback (2차) | [vLLM #31085](https://github.com/vllm-project/vllm/issues/31085), [#47749](https://github.com/vllm-project/vllm/issues/47749), [#50288](https://github.com/vllm-project/vllm/pull/50288) |
| **DeepGEMM** | "NVIDIA **SM90 or SM100**", **CUDA 12.9+**. SM100은 UE8M0 packed scale 사용 | O | **X** | [DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM) |
| **NCCL** | **2.25.1**: "Added support for Blackwell" (CUDA 12.2/12.4/12.8 빌드) | O (NVLink 5 도메인 최적화) | O (PCIe; GeForce는 NVLink 없음) | [NCCL 2.25.1 RN](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-25-1.html) |
| cuFFT | 12.8: SM120은 legacy callback 커널에 한해 PTX JIT만 지원 | O | 주의 | 12.8 RN |

**요약**: "Blackwell 지원"이라고 적힌 라이브러리라도 **사실상 sm_100에 먼저 최적화**되어 있고(tcgen05/TMEM 기반 커널), sm_120은 **별도 커널 경로가 필요하므로 지원 시점과 성숙도가 뒤처지는 경우가 많다**. FA4와 DeepGEMM은 현재 sm_100 전용이다.

---

## 7. 연구 관점 핵심 포인트

1. **NVFP4 저정밀 학습·추론**
   - NVIDIA의 [NVFP4 pretraining 논문](https://arxiv.org/abs/2509.25149): 12B hybrid Mamba-Transformer를 **10T 토큰** NVFP4로 학습해 FP8과 비슷한 loss와 정확도를 얻음(MMLU-pro 62.58% vs FP8 62.62%). 레시피는 (1) 민감한 층을 고정밀로 유지, (2) **Random Hadamard Transform**으로 블록 outlier 분산, (3) **weight 2D(16×16) block scaling**으로 fwd/bwd 표현 일치, (4) **gradient stochastic rounding**. TE `NVFP4BlockScaling` 기본값: weight 2D, activation/gradient 1D ([TE 문서](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html)).
   - 연구 주제: MXFP4(E8M0/32) vs NVFP4(E4M3/16+FP32) 오차 비교, per-tensor scale 계산 비용, RHT 커널 fusion, SR 하드웨어 유무에 따른 sm_100과 sm_120 학습 차이(PTX상 `cvt.rs` 는 sm_100a/103a/107a 전용).
2. **TMEM 기반 커널 설계 (sm_100)**
   - 누산기를 레지스터 대신 TMEM에 두기 때문에 레지스터 압박이 줄고 epilogue(TMEM→RF→SMEM→TMA store)와 다음 타일의 MMA를 겹칠 수 있다. `tcgen05.alloc/dealloc` 로 명시적 할당(32 column 단위, 2의 거듭제곱), `tcgen05.commit` + mbarrier로 완료를 추적한다(PTX ISA).
   - Attention(FA4)에서는 S/P 행렬을 TMEM에 두고 softmax와 MMA를 파이프라인화하는 설계가 핵심 — B300의 SFU 2x는 softmax 병목을 직접 겨냥한다.
3. **Warp specialization**: TMA producer warp, MMA 발행 warp(단일 스레드 발행), epilogue warpgroup으로 역할을 나누는 구조가 SM100 CUTLASS 커널의 기본이다(`KernelTmaWarpSpecialized*Sm100`). SM120은 Hopper식 pingpong/cooperative(4 또는 8 MMA warp) 스케줄을 쓴다(CUTLASS 문서).
4. **2-CTA MMA와 cluster**: `cta_group::2` 로 SM 두 개가 B 타일을 나눠 로드하고 256×N MMA를 수행 → SMEM 사용량과 L2 트래픽 절감. TMA multicast와 cluster 16 (B200 비이식)을 조합한 타일링 탐색이 가능. **sm_120에서는 불가능**하므로 이식성 연구 주제가 된다.
5. **sm_120의 작은 SMEM(99 KB/block)과 레지스터 누산**: 같은 NVFP4 GEMM이라도 타일 크기가 작아지고(CUTLASS SM120 128×128×128 등) FP32 누산 시 GeForce 처리량이 절반으로 떨어짐 → 워크스테이션/컨슈머 GPU용 저정밀 커널 최적화(Colfax 등)가 활발하다.
6. **Decompression Engine (800 GB/s)**: 압축된 데이터(DB, 가중치/KV 압축 등)를 GPU에서 바로 푸는 파이프라인. LLM 가중치 압축 로딩에 쓸 수 있는지는 공개 API 범위 (미확인).
7. **Family target 이식성**: `sm_100f` 커널은 B200·B300·Rubin(10.7)까지 이어지는 반면 RTX/Spark는 `sm_120f` 로 따로 빌드해야 한다. 연구 코드를 배포할 때 **두 계열 모두 fatbin에 넣고 런타임 dispatch** 하는 것이 사실상 필수다.
8. **Blackwell Ultra 특이점**: NVFP4 dense 1.5x, SFU EX2 2x, INT8 tcgen05 제거, **FP64 Tensor Core 제거(FP32:FP64 64:1)** → "추론·attention 특화"로 가는 방향. INT8 양자화와 FP64 HPC 연구는 B300에서 재검토가 필요하다.

---

## 부록 A. 빠른 참조 — 컴파일 플래그 예

```bash
# B200(10.0) + B300(10.3) 공통 tcgen05 커널, RTX 50/RTX PRO/GB10 공통 커널, + 이식용 PTX
nvcc -gencode arch=compute_100f,code=sm_100f \
     -gencode arch=compute_120f,code=sm_120f \
     -gencode arch=compute_100,code=compute_100 \
     -c kernels.cu
# B200 한 기종의 모든 arch-specific 기능(예: scale_vec::4X) 사용 시
nvcc -gencode arch=compute_100a,code=sm_100a -c kernels.cu
```
(`f` 는 CUDA 12.9+ 필요. 출처: [NVIDIA 블로그](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/), [Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html))

## 부록 B. 추가 참고 (2차 연구)

- Jarmusch et al., "Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks", [arXiv 2507.10789](https://arxiv.org/abs/2507.10789) — 로컬 사본 있음. 수치는 본문에 인용하지 않았다.
- Colfax Research, [NVFP4 blockscaled GEMM on RTX PRO 6000 (SM120)](https://research.colfax-intl.com/optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/).
