# 06. SM 타깃별 신규 기능 표 (sm_70 Volta → sm_121 GB10)

> 작성일: 2026-09-11 · 범위: compute capability(CC) 7.0~12.1의 SM 타깃(`sm_XX`, `sm_XXa`, `sm_XXf`)마다 **새로 추가된 기능**을 정리한다. Blackwell 계열 내부의 상세 비교와 `a`/`f` suffix 설명은 [01-architecture.md §2](01-architecture.md)를 따른다.
> 표기 규칙: 1차 출처로 확인하지 못한 값은 **(미확인)**, 공식 수치에서 계산한 값은 **(계산값)**, 커뮤니티 자료는 **2차 출처**. 이 문서는 처리량 수치를 거의 다루지 않는다(다룰 때는 저장소 규칙대로 [sparse]/[dense] 표기). 로컬 사본: `docs/sources/architecture/` ([SOURCES.md](sources/architecture/SOURCES.md)), `docs/sources/sm-features/` ([SOURCES.md](sources/sm-features/SOURCES.md)).

**주요 1차 출처** (아래 표에서는 약칭으로 인용):

- **PTX ISA 9.4**: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html. 명령별 "Target ISA Notes", Table 71 (PTX Release History), Table 72 (Arch-specific/Family-specific PTX Features Release History), `.target` 절(Table 70 Architecture Families), §13 Release Notes. 셀 안의 "(PTX x.y)"는 해당 기능이 처음 들어간 PTX ISA 버전이다.
- **PG 13.4**: CUDA Programming Guide 13.4, Compute Capabilities 부록 Table 28~33. https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- **PG 12.9**: 구판 CUDA C++ Programming Guide 12.9. 13.x에서 빠진 CC 5.x~7.2 값이 여기 있다. https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html
- **Tuning Guide**:
  - [Volta](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html)
  - [Turing](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)
  - [Ampere](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
  - [Ada](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)
  - [Hopper](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
  - [Blackwell](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- **백서(WP)**:
  - [V100](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
  - [Turing](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf)
  - [A100](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
  - [GA102](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)
  - [Ada](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)
  - [H100](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)
  - [RTX Blackwell](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- **릴리스 노트**:
  - [CUDA 12.8 RN](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
  - [CUDA 13.0 RN](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)
  - [CUDA 13.4 RN](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
  - [CUDA Features Archive](https://docs.nvidia.com/cuda/pdf/CUDA_Features_Archive.pdf)
- **기타**:
  - [CUDA GPUs 목록](https://developer.nvidia.com/cuda-gpus)
  - [CUTLASS Blackwell 문서](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md)
  - [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)
  - [Hopper Compatibility Guide](https://docs.nvidia.com/cuda/hopper-compatibility-guide/index.html)
  - [NVIDIA 블로그: family-specific (2025-05)](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/)
  - [Blackwell Ultra 블로그](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)

---

## 0. TL;DR

1. SM 세대마다 Tensor Core를 쓰는 프로그래밍 모델이 바뀌었다.
   - **sm_70** `wmma`
   - → **sm_75/80** warp 단위 `mma.sync` + `ldmatrix` + `cp.async`
   - → **sm_90a** warpgroup 단위 `wgmma` + TMA + cluster
   - → **sm_100a/f** 단일 스레드가 발행하는 `tcgen05.mma` + TMEM + CTA pair
   - **sm_120** 은 이 흐름에서 한 단계 뒤로 가서 `mma.sync` 에 FP4/FP6 block-scale 확장을 붙인 구조다.
2. **`wgmma` 는 `sm_90a` 에서만 쓸 수 있고, 이후 어느 아키텍처에도 없다.** `setmaxnreg`·`tensormap.replace` 도 sm_90a 전용으로 시작했지만, Blackwell에서는 `sm_100f`/`sm_110f`/`sm_120f` family 기능으로 이어졌다. 그래서 **RTX 50에서도 `setmaxnreg` 를 쓸 수 있다**.
3. **B300(CC 10.3)은 B200의 상위집합이 아니다.** 빠진 것과 새로 생긴 것이 둘 다 있다.
   - 빠진 것: `tcgen05 .kind::i8` 이 없다(PTX). **PG 13.4 Table 33에서 CC 10.3 행은 FP64 Tensor Core가 공란**이고, Table 30의 FP32:FP64는 **64:1**(B200 10.0은 2:1)이다.
   - 새로 생긴 것: `tcgen05.ld.red`, K=96 MMA, `cvt.rs` 가 추가됐다.
4. `clusterlaunchcontrol`(CLC), `st.bulk`, packed FP32(`add/mul/fma.f32x2`), 256-bit ld/st는 **base `sm_100` 기능**이다. onion 모델에 따라 **sm_120(RTX 50/PRO)에서도 쓸 수 있다**.
   - 반면 `tcgen05`/TMEM/`cta_group::2`/`cvt.rs`/`redux.sync.f32` 는 10.x(와 일부 11.0) 전용이다.
5. CUDA 13.0에서 **Maxwell·Pascal·Volta(sm_50~sm_72)의 오프라인 컴파일과 라이브러리 지원이 제거**됐다. 따라서 현행 툴킷의 최저 타깃은 sm_75다.
   - 새로 생긴 타깃: **sm_101 → sm_110 개명**(Jetson Thor, CUDA 13.0/PTX 9.0), **sm_88**(PTX 9.0, 제품 미확인), **sm_107**(PTX 9.4/CUDA 13.4, Rubin developer preview).

---

## 1. Compute Capability / SM 타깃 개요

| CC / SM 타깃 | 아키텍처 | 대표 GPU ([CUDA GPUs](https://developer.nvidia.com/cuda-gpus)) | 최초 지원 CUDA | 최초 PTX ISA | `a` / `f` | 비고 |
|---|---|---|---|---|---|---|
| 5.x `sm_50/52/53` | Maxwell | GTX 900, Jetson TX1(5.3) | 6.0 / 6.5 / 7.0 | 4.0 / 4.1 / 4.2 | – | **CUDA 13.0에서 제거**(12.x로만 빌드) |
| 6.x `sm_60/61/62` | Pascal | P100(6.0), GTX 10(6.1), TX2(6.2) | 8.0 | 5.0 | – | **CUDA 13.0에서 제거** |
| **7.0** `sm_70` | Volta | V100, TITAN V | 9.0 | 6.0 | – | **CUDA 13.0에서 제거** |
| **7.2** `sm_72` | Volta (Tegra) | Jetson AGX Xavier (현행 CUDA GPUs 페이지에는 없음, (미확인)) | 9.1 | 6.1 | – | **CUDA 13.0에서 제거** |
| **7.5** `sm_75` | Turing | T4, RTX 20, T400 등 | 10.0 | 6.3 | – | CUDA 13.x가 지원하는 최저 CC |
| **8.0** `sm_80` | Ampere (GA100) | A100 | 11.0 | 7.0 | – | |
| **8.6** `sm_86` | Ampere (GA10x) | A10, RTX 30, RTX A6000 | 11.1 | 7.1 | – | |
| **8.7** `sm_87` | Ampere (Tegra) | Jetson AGX Orin / Orin NX / Orin Nano | 11.4 | 7.4 | – | |
| 8.8 `sm_88` | (미확인) | (미확인) — CUDA GPUs 페이지와 PG 13.4 표에 없음 | 13.0 | 9.0 | – | PTX는 "Baseline feature set for sm_88"만 기술 |
| **8.9** `sm_89` | Ada Lovelace | L4, L40, L40S, RTX 40 | 11.8 | 7.8 | – | FP8 `mma`는 PTX 8.4(CUDA 12.4)에서야 추가 |
| **9.0** `sm_90` / `sm_90a` | Hopper | H100, H200, GH200 | 11.8 (`sm_90`) / 12.0 (`sm_90a`) | 7.8 / 8.0 | **`a` 최초 도입** | `compute_90a` PTX는 Blackwell에서 실행 불가 |
| **10.0** `sm_100/100a/100f` | Blackwell (GB100) | B200, GB200 | 12.8¹ (`f`는 12.9) | 8.6 (`f`는 8.8) | a, f | `compute_100f` → CC 10.0·10.3·10.7 |
| **10.3** `sm_103/103a/103f` | Blackwell Ultra | B300, GB300, GB300 DGX Station | 12.9 | 8.8 | a, f | `compute_103f` → 10.3·10.7 |
| 10.7 `sm_107/107a/107f` | Rubin ([CUDA 13.4 RN](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) "Developer Preview") | (미확인) | 13.4 | 9.4 | a, f | sm_10x 패밀리. `compute_107f` → 10.7만 |
| **11.0** `sm_110/110a/110f` (구 `sm_101`) | Blackwell (Tegra) | Jetson T5000 / T4000 (Thor) | 12.8(`sm_101`) → 13.0(`sm_110`) | 8.6(`sm_101`) → 9.0(`sm_110`) | a, f | [CUDA 13.0 RN](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html): "SM101 has been renumbered as SM110". PTX Table 70: sm_11x 패밀리 = `sm_110f`, `sm_101f` |
| **12.0** `sm_120/120a/120f` | Blackwell (GB20x) | GeForce RTX 50, RTX PRO 6000 Blackwell | 12.8 (`f`는 12.9) | 8.7 (`f`는 8.8) | a, f | `compute_120f` → 12.0·12.1 |
| **12.1** `sm_121/121a/121f` | Blackwell (GB10) | NVIDIA GB10 (DGX Spark) | 12.9 | 8.8 | a, f | `compute_121f` → 12.1만 |

¹ PTX Table 71은 PTX ISA 8.6(sm_100/101 최초)을 "CUDA 12.7"에 대응시킨다. 하지만 공개 툴킷에서 SM_100/SM_101/SM_120 컴파일러 지원을 처음 발표한 것은 [CUDA 12.8 RN](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)이다. sm_103·sm_121·family target은 [Features Archive](https://docs.nvidia.com/cuda/pdf/CUDA_Features_Archive.pdf)의 12.9 절 기준이다.

**Pre-Volta 기준선 (한 줄)**:
- sm_50~53 (Maxwell): sm_53에서 `.f16`/`.f16x2` 산술 추가.
- sm_60~62 (Pascal): `atom.add.f64`, atomic `.scope`, sm_61에서 `dp4a`/`dp2a` 추가.
- Tensor Core·통합 L1/SMEM 이전 세대이며, CUDA 13.0부터 오프라인 컴파일 대상이 아니다([Features Archive §5.2](https://docs.nvidia.com/cuda/pdf/CUDA_Features_Archive.pdf), PTX `.target` 표).

---

## 2. SM별 신규 기능 표 (직전 세대 대비 **새로 생긴 것 / 없어진 것**)

- 열 구성: **TC** = Tensor Core / MMA 명령·데이터 타입, **메모리** = 메모리·데이터 이동, **실행** = 실행·동기화 모델, **기타** = 그 외·제거 항목.
- 근거: 별도 표기가 없으면 PTX ISA target notes와 §13 Release Notes, PG 13.4 Table 29~33, 해당 세대 Tuning Guide/백서다.

### 2.1 Volta ~ Ada (sm_70 ~ sm_89)

| SM | TC / MMA | 메모리·데이터 이동 | 실행·동기화 | 기타 (제거·주의 포함) |
|---|---|---|---|---|
| **sm_70** (V100) | **1세대 Tensor Core 8개/SM**: FP16 입력, FP16/FP32 누산 ([V100 WP](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)). `wmma.load/mma/store` (PTX 6.0). `mma.sync.m8n8k4 .f16` (PTX 6.4) — PTX는 이를 "sm_70에 최적화, 다른 아키텍처에서는 성능이 크게 낮을 수 있음"이라고 적는다 | **통합 L1/SMEM 128 KB**, SMEM 최대 96 KB/SM·block(48 KB 초과는 dynamic + opt-in) ([Volta TG](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html)). 메모리 일관성 모델 공식화: `.relaxed/.acquire/.release` + scope, `fence` (PTX 6.0) | **Independent Thread Scheduling**. `shfl.sync`/`vote.sync`/`match.sync`, `bar.warp.sync`, `nanosleep` 추가. `.sync` 없는 `shfl`/`vote`는 sm_70+ 타깃에서 **제거**(PTX 6.4) | HW 가속 MPS(V100 WP). `atom.add.f16`, `atom.cas.b16`. FP32:FP64 = 2:1 (SM당 FP32 64 / FP64 32). **CUDA 13.0에서 드롭** |
| **sm_72** (Xavier) | **정수 `wmma`** (`.s8/.u8` 입력, `.s32` 누산) — PTX 타깃 표 "Adds support for integer multiplicand and accumulator matrices in wmma" | 한도는 7.0과 같음: 64 warps, 32 blocks, SMEM 96 KB/SM ([PG 12.9](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html) Table 28) | – | `cvt.pack` 추가. **CUDA 13.0에서 드롭** |
| **sm_75** (Turing) | **2세대 TC 8개/SM**: `wmma` **sub-byte INT4(`.s4/.u4`)·1-bit(`.b1`, xor)** 추가. `mma.sync` 신규 shape `.m16n8k8` f16, `.m8n8k16` s8, `.m8n8k32` s4, `.m8n8k128` b1 (PTX 6.5). Volta cubin의 TC는 Turing 피크의 절반만 낸다(재컴파일 권장, [Turing TG](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)) | **`ldmatrix`** (PTX 6.5), `movmatrix` (PTX 7.8 명령, sm_75+). `ld .level::prefetch_size`. 통합 L1/SMEM 96 KB, **SMEM 64 KB/SM로 감소** | **최대 상주 warp 32/SM**(Volta 64에서 감소), block 16/SM | `tanh`, `ex2.approx.f16` HW 명령. `cvt.pack` sub-byte. FP32:FP64 = **32:1** ([Turing WP](https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf): "1/32nd … to ensure any programs with FP64 code operates correctly"). RT Core(그래픽) |
| **sm_80** (A100) | **3세대 TC 4개/SM**(TC당 성능 2배) ([A100 WP](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)). **BF16·TF32·FP64(DMMA) Tensor Core**. `mma` `.m16n8k16` f16/bf16, `.m16n8k8` tf32, `.m8n8k4` f64, INT8 `.m16n8k32`, INT4 `.m16n8k64`, b1 `.m16n8k256` + `.and` (PTX 7.0/7.1). **`mma.sp` 2:4 structured sparsity** (PTX 7.1). `wmma` bf16/tf32/f64 | **`cp.async`** (LDGSTS: global→shared 비동기, 레지스터 경유 없음·L1 우회 가능) + `commit_group/wait_group`. **통합 192 KB, SMEM 164 KB/SM·163 KB/block**. **L2 residency control**, `createpolicy`/`.L2::cache_hint`/`applypriority`/`discard` (PTX 7.4). Compute data compression (A100 WP) | **`mbarrier`** (HW split arrive/wait barrier = `cuda::barrier`). **`redux.sync`** (warp reduce, 정수). 64 warps, 32 blocks/SM | BF16 연산 지원(Table 29). `cvt` `.relu`/bf16/tf32, `min/max .NaN`. **MIG**, 3세대 NVLink, task graph 가속(A100 WP). FP32:FP64 = 2:1 |
| **sm_86** (GA10x) | 3세대 TC 4개/SM(sparsity 포함, [GA102 WP](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)). **FP64 TC 가속 없음**(Table 33 8.6 행 FP64 공란) | 통합 128 KB, **SMEM 100 KB/SM·99 KB/block** | **48 warps, 16 blocks/SM** | **SM당 FP32 2배**(8.0 대비, [Ampere TG](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)). PTX 타깃 표상 유일한 ISA 추가는 `min/max .xorsign.abs`. FP32:FP64 = 64:1 |
| **sm_87** (Orin) | 3세대 TC(개수 (미확인)). FP64 TC 없음(Table 33) | **SMEM은 8.0급**: 통합 192 KB, 164 KB/SM, 163 KB/block | 48 warps, 16 blocks/SM (8.6과 같음) | PTX는 "Baseline feature set"(신규 명령 없음). FP32:FP64 = 64:1 (Table 30) |
| *(sm_88)* | – | – | – | PTX 9.0(CUDA 13.0)에서 target만 추가됨. 제품·한도는 (미확인) |
| **sm_89** (Ada) | **4세대 TC 4개/SM + FP8(E4M3/E5M2)** ("Hopper FP8 Transformer Engine", [Ada TG](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)). PTX: `mma`/`mma.sp` `.e4m3/.e5m2` (PTX 8.4). `cvt` e4m3x2/e5m2x2 (PTX 8.1에서 sm_89로 확장) | SMEM 100 KB/SM(8.6과 같음). **L2 확대**: AD102 98,304 KB (Ada TG) | **24 blocks/SM**(8.6은 16), 48 warps | FP32 2배/SM(8.0 대비). FP32:FP64 = 64:1. Ada WP: 소수의 FP64 유닛은 "FP64 Tensor Core code" 정확성 보장용. SER·Opacity Micromap(그래픽) |

### 2.2 Hopper ~ Blackwell (sm_90 ~ sm_121)

| SM | TC / MMA | 메모리·데이터 이동 | 실행·동기화 | 기타 (제거·주의 포함) |
|---|---|---|---|---|
| **sm_90** (H100, base) | **4세대 TC 4개/SM** ([H100 WP](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)). FP8은 sm_89부터 base `mma.sync`로 쓸 수 있음. `mma .f64` `.m16n8k4/k8/k16` (sm_90). **INT4 TC 가속 제외**: PG Table 33에서 9.0 이후 INT4 공란. PTX 문법상 `.s4` mma는 sm_80+로 남아 있어 에뮬레이션 여부는 (미확인) | **TMA**: `cp.async.bulk`, `cp.async.bulk.tensor`(1D~5D, tile/im2col), `cp.reduce.async.bulk{.tensor}`, `.multicast::cluster` (PTX 8.0). **DSMEM**: `.shared::cluster`, `mapa`, `getctarank` (PTX 7.8). `st.async`/`red.async` (PTX 8.1). `multimem.*` (NVLS, PTX 8.1). **`stmatrix`**. `prefetch.tensormap`. **통합 256 KB, SMEM 228 KB/SM·227 KB/block**. 128-bit atomics, float2/float4 `atomicAdd` (Table 29) | **Thread Block Cluster** (portable 8, H100 non-portable 16, [Hopper TG](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)), `barrier.cluster`, cluster 특수 레지스터. **mbarrier transaction count** (`expect_tx/complete_tx`), `try_wait`, `.cluster` scope (PTX 8.0). **`elect.sync`**. **`griddepcontrol`** (Programmatic Dependent Launch, PTX 7.8). `fence.proxy.async` | **DPX 명령 HW 네이티브**(Table 29: 9.0·10.x만 "Native"). BF16 `add/mul/fma` 네이티브, bf16 atomics. **Confidential Computing**, 2세대 MIG, NVLink 4 (H100 WP). FP32:FP64 = 2:1 |
| **sm_90a** | **`wgmma.mma_async`**(warpgroup = 4 warps, A/B는 SMEM descriptor, 누산은 레지스터. FP16/BF16/TF32/FP8/INT8 등) + `.sp` (PTX 8.0/8.2). **sm_90a 전용 — 이후 어떤 타깃에도 없음**. PTX 9.2 명확화: FP8 `wgmma` + `.f32` 누산은 "half보다 높고 single보다 낮은 정밀도"로 누산 | `.multicast::cluster`는 "sm_90a에 최적화, 다른 타깃은 성능이 크게 낮을 수 있음"(PTX 8.2 명확화). `tensormap.replace` (PTX 8.3). `ldmatrix .m8n16 .s8.s4` (PTX 9.4) | **`setmaxnreg`** (warpgroup 간 레지스터 재분배 → warp specialization) | `compute_90a` PTX는 앞/뒤 호환 없음. "PTX compiled for compute_90a are not supported on the Blackwell architecture" ([Blackwell Compat Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html)) |
| **sm_100** (B200, base) | base 타깃에서는 **tcgen05 불가**. `mma.sync`/`wmma`만(onion). FP64 TC 가속 있음 (Table 33 10.0) | `cp.async.bulk{.tensor}`의 **`.shared::cta` 목적지**, `.cp_mask` (PTX 8.6). `.tile::gather4`/`.im2col::w` (`.shared::cta` 목적지일 때). **`st.bulk`**. **256-bit `ld/st`** (`.v8.b32/.v4.b64`, PTX 8.8). `multimem.st.async/red.async`, `fabric.*`, `createpolicy.range.fabric` (PTX 9.3~9.4) | **`clusterlaunchcontrol.try_cancel/query_cancel`** (CLC: 실행 중 cluster 취소를 통한 work stealing, PTX 8.6). `st.async/red.async`의 `.global`·`.release`·`.mmio`. `fence` `.acquire/.release` | **packed FP32 `add/sub/mul/fma.f32x2`**, mixed `add/sub/fma.f32.{f16,bf16}`, 3-input `min/max`, `cvt.satfinite.tf32`. **128-bit 부동소수 연산**(Table 29: 10.x+). DPX 네이티브. FP32:FP64 = 2:1 |
| **sm_100f** (10.0·10.3·10.7 공통) | **`tcgen05.mma{.sp}{.ws}`**: `.kind::f16/tf32/f8f6f4/mxf8f6f4/mxf4/mxf4nvf4` (단 `.sp`의 mxf4* 제외), **`cta_group::1/2`(CTA pair)**, `.block16/.block32` block scaling, `scale-input-d`. **Tensor Memory**: `tcgen05.alloc/dealloc/ld/st/cp/commit/fence/wait` — CTA당 128 lane × 512 column × 32-bit = **256 KB** | TMA `.tile::gather4/scatter4`, `.im2col::w{::128}` (`.shared::cluster` 목적지), TMA `.cta_group`. `ldmatrix .m16n16/.m8n16/.b8` + `.src_fmt/.dst_fmt`, `stmatrix .m16n8/.b8`. `multimem` FP8 타입·`.acc::f16`. `tensormap.replace` + `.swizzle_atomicity` | `setmaxnreg`. CLC `.multicast::cluster::all`. **`redux.sync .f32`**(+`.abs/.NaN`) — 12.x·11.0에는 없음 | `cvt` FP4/FP6/UE8M0 계열(`.e2m1x2/.e2m3x2/.e3m2x2/.ue8m0x2`), bf16x2↔FP8/FP6/FP4 (PTX 9.1~9.2) |
| **sm_100a** (10.0 전용) | **`tcgen05 .kind::i8`**, `.scale_vec::1X/2X/4X` 표기, `tcgen05.mma.sp .kind::mxf4/mxf4nvf4`, **`tcgen05.shift`** | – | – | **`cvt.rs` (HW stochastic rounding, `.e2m1x4/.e4m3x4/.e5m2x4/.e3m2x4/.e2m3x4` ← f32)** (PTX 8.7). `cvt .s2f6x2` (9.1). FP8/FP6/FP4 x4 packed `add/sub/mul/fma` (9.4, 100a·103a). **제거**: `wgmma` 없음(tcgen05로 대체) |
| **sm_103** (B300) `103a/103f` | 100f 기능 전부 + **`tcgen05.ld.red`** (103f, PTX 8.8). **K=96** MMA shape (103a). mma.sp mxf4/mxf4nvf4, `tcgen05.shift` (103a). **없는 것: `tcgen05 .kind::i8`** (100a·110a만). PG Table 33의 **10.3 행은 FP64 공란(= FP64 Tensor Core 없음)이고 INT8은 "Yes"** — INT8은 warp-level `mma.sync` 경로로 추정(미확인) | tensormap `.swizzle_mode` 96B (103a). SMEM 한도는 10.0과 같음(228/227 KB) | 10.0과 같음(64 warps, 32 blocks) | `cvt.rs` (103a). tcgen05 SMEM descriptor의 leading-dim byte address mode (103a). **FP32:FP64 = 64:1**(Table 30. 10.0은 2:1). SFU EX2 10.7 TeraExp/s, TMEM 256 KB/SM ([Ultra 블로그](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)) |
| **sm_107** (Rubin, dev preview) `107a/107f` | `.kind::ti16`, **UE5M3 scale 타입**, `.decompress::lut::b`, `.collector::b::*`, K=64(dense f8f6f4)·K=128(mxf4) 신규 shape (107f). K=96, `spcompress/spdecompress`, `tcgen05.ld.spcompress` (107a). **INT8 Tensor Core 없음**(Table 33 10.7 행 공란), FP64 TC 있음 | **SMEM 328 KB/SM·327 KB/block**(oversized opt-in `cudaSharedMemoryModeAllowOversizedSharedMemory`), 통합 336 KB (Table 31/32). `.multicast::cluster::32b`(mbarrier·`cp.async.bulk`·`tcgen05.commit`), `applypriority.async.bulk{.tensor}`, TMA `.override::global_address`, `.report_mechanism`, `.im2col_no_offs::w` | Table 30 기준 **최대 32 warps·16 blocks·1024 threads/SM**(표기 그대로. 10.0의 절반) | `cvt .ue5m3x2`, `.pzo`, FP8/6/4 `.rz`, `.scaled::n1::ue8m0`. packed int `set`, `f16x2/bf16x2/f32x2` 혼합 `add/fma`. **FP32:FP64 = 4:1**. `cvt.rs` (107a). ⚠ PTX 문서 안에서 `tcgen05.alloc .exclusive` 대상이 Table 72(sm_107f)와 target note(sm_100f/sm_110f)로 서로 다르게 적혀 있음 |
| **sm_110** (Thor, 구 sm_101) `110a/110f` | **데이터센터 ISA 계열**: tcgen05 전체(110f), TMEM, `cta_group::2`, `.block16/32`, **`tcgen05.ld.red`**(101f→110f). 110a: **`.kind::i8`**, `tcgen05.shift`, mma.sp mxf4*, `cvt .s2f6x2`. **FP64 TC 없음**(Table 33) | SMEM 228 KB/SM·227 KB/block, 통합 256 KB (10.0과 같음) | **48 warps, 24 blocks/SM**(12.x와 같음) | **없는 것**: `cvt.rs`, `.scale_vec::NX`, `redux.sync.f32`, **DPX 네이티브**(Table 29: "Multiple Instr."). FP32:FP64 = 64:1. Thor SoC에는 DLA 없음(CUDA 13.0 RN) |
| **sm_120** (RTX 50 / RTX PRO, base) | **tcgen05·TMEM·wgmma 모두 없음**. Hopper 이전 방식의 warp-level `mma.sync`(onion). `mma.sp` `.m16n8k32` + FP8 입력 + `.f16` 누산은 "requires sm_120". FP64 TC는 정확성용 소량 ([RTX WP](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf): "very minimal number of FP64 Tensor Cores"), Table 33 12.x FP64 공란 | TMA·cluster·DSMEM **있음**(Table 29). **TMA multicast**: PTX 문법상 sm_90+ 허용이지만 권장 타깃 목록에 sm_120이 없고, CUTLASS는 "GeForce에는 multicast 기능이 없어 cluster shape를 1x1x1로 고정" ([CUTLASS](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md)). SMEM **100 KB/SM·99 KB/block** | CLC(base sm_100 기능) 사용 가능. **48 warps**. blocks/SM: PG 13.4 Table 30은 **24**, Blackwell TG·PG 12.9는 **32**(출처 간 불일치) | base sm_100 기능(`f32x2`, `st.bulk`, 256-bit ld/st, 128-bit FP)을 onion으로 상속. **DPX 비네이티브**. FP32:FP64 = 64:1 (SM당 FP64 코어 2개) |
| **sm_120f** (12.0·12.1 공통) | **`mma.sync` FP4/FP6 + block scaling**: `.kind::f8f6f4` (e2m1/e2m3/e3m2/e4m3/e5m2), `.kind::mxf8f6f4 .block_scale`, `.kind::mxf4/.kind::mxf4nvf4 .block_scale .scale_vec::2X/4X`(PTX 8.7/8.8, 9.1에서 mxf4nvf4 + ue8m0 4X). `mma.sp`도 같음(mxf4* 제외). block-scaled는 **TN 레이아웃만**(CUTLASS) | `ldmatrix/stmatrix` 신규 shape·`.b8`, `tensormap.replace`(`.swizzle_atomicity`는 sm_120a 제약 있음). TMA에서 `.shared::cluster` 목적지로 sub-byte 타입(`.b4x16` 등) 복사는 **sm_120a 미지원** (PTX "Restriction on Tensor Copy instructions") | **`setmaxnreg`**, CLC `.multicast::cluster::all` | `cvt` FP4/FP6/UE8M0. `add/sub/min/max .u8x4/.s8x4`, `add.sat .u16x2/.s16x2/.u32` (PTX 9.2, 120f) |
| **sm_120a** | **`mma.sp .kind::mxf4/mxf4nvf4`** (a 전용, 120a·121a) | `multimem` FP8 타입·`.acc::f16` (120a·121a만. `sm_120f`로는 불가) | – | `cvt .s2f6x2`. **없는 것**: `cvt.rs`, `redux.sync.f32`, tcgen05 전부 |
| **sm_121** (GB10) `121a/121f` | sm_120f 패밀리. 121a는 120a와 같은 arch-specific 항목(mma.sp mxf4*, multimem FP8, `.s2f6x2`)을 가짐 | PG 13.4는 12.x를 묶어 표기(SMEM 100 KB/SM) | 48 warps, 24 blocks/SM (12.x) | `compute_120f` 바이너리가 그대로 동작. `compute_121f`는 12.1만 |

---

## 3. 기능 × SM 매트릭스

범례:
- **O**: suffix 없는 base 타깃으로 사용 가능. 이후 CC에도 onion으로 상속.
- **f**: family 타깃(`sm_XXf`, 또는 같은 패밀리의 `a`)이 필요.
- **a**: 해당 CC의 `sm_XXa` 전용이며 패밀리 이식 불가.
- **△**: 조건부(PTX 문법상 허용되나 HW 가속·권장 대상 아님 등, 셀 주석 참고).
- **X**: 불가 또는 없음.
- CUDA 13.x는 7.0/7.2를 컴파일하지 않는다(12.x 툴킷 필요).

### 3.1 Volta ~ Hopper

| 기능 | 7.0 | 7.2 | 7.5 | 8.0 | 8.6 | 8.7 | 8.9 | 9.0 |
|---|---|---|---|---|---|---|---|---|
| Independent Thread Scheduling (`*.sync` 필수) | O | O | O | O | O | O | O | O |
| `wmma` FP16 | O | O | O | O | O | O | O | O |
| `wmma` INT8 | X | O | O | O | O | O | O | O |
| `wmma`/`mma` INT4·INT1 (sub-byte) | X | X | O | O | O | O | O | △ (Table 33 INT4 가속 없음) |
| `mma.sync .m8n8k4 .f16` | O | O | O | △ | △ | △ | △ | △ (sm_70 최적화, 이후 성능 저하 가능) |
| `mma.sync` `.m16n8k8` f16 · INT8 `.m8n8k16` | X | X | O | O | O | O | O | O |
| `mma.sync` `.m16n8k16` f16/bf16 · tf32 · f64 `.m8n8k4` | X | X | X | O | O | O | O | O |
| `ldmatrix` / `movmatrix` | X | X | O | O | O | O | O | O |
| `stmatrix` | X | X | X | X | X | X | X | O |
| BF16 / TF32 Tensor Core | X | X | X | O | O | O | O | O |
| FP64 Tensor Core (DMMA) | X | X | X | O | △ | △ | △ | O |
| `mma.sp` 2:4 sparse | X | X | X | O | O | O | O | O |
| FP8 E4M3/E5M2 (`mma`, `cvt`) | X | X | X | X | X | X | O | O |
| `cp.async` (LDGSTS) | X | X | X | O | O | O | O | O |
| `mbarrier` (arrive / test_wait) | X | X | X | O | O | O | O | O |
| `redux.sync` (정수) | X | X | X | O | O | O | O | O |
| L2 residency control / `createpolicy` | X | X | X | O | O | O | O | O |
| BF16 산술 네이티브 (`add/mul.bf16`) | X | X | X | △ (`fma`·`min/max`·`cvt`만) | △ | △ | △ | O |
| TMA `cp.async.bulk{.tensor}` | X | X | X | X | X | X | X | O |
| TMA `.multicast::cluster` | X | X | X | X | X | X | X | O (sm_90a 권장) |
| Thread Block Cluster / DSMEM | X | X | X | X | X | X | X | O |
| mbarrier `expect_tx` / `try_wait` | X | X | X | X | X | X | X | O |
| `elect.sync` | X | X | X | X | X | X | X | O |
| `griddepcontrol` (PDL) | X | X | X | X | X | X | X | O |
| `st.async`/`red.async`, `multimem.*` | X | X | X | X | X | X | X | O |
| 128-bit atomics | X | X | X | X | X | X | X | O |
| DPX 네이티브 | X | X | X | X | X | X | X | O |
| `wgmma.mma_async` | X | X | X | X | X | X | X | **a** |
| `setmaxnreg` | X | X | X | X | X | X | X | **a** |
| `tensormap.replace` | X | X | X | X | X | X | X | **a** |

- FP64 TC의 △: 8.6/8.7/8.9는 PTX상 `mma .f64`를 컴파일할 수 있지만 PG Table 33에 FP64 TC 가속이 없다. 소수의 FP64 유닛은 정확성 보장용이다([Ada WP](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf), [GA102 WP](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf)).
- DPX X: PG 13.4 Table 29 기준 "Multiple Instr."(소프트웨어 에뮬레이션).

### 3.2 Hopper ~ Blackwell (GPU 기준, 셀 = 그 GPU에서 필요한 최소 타깃 종류)

| 기능 | 9.0 H100 | 10.0 B200 | 10.3 B300 | 10.7 Rubin† | 11.0 Thor | 12.0 RTX 50/PRO | 12.1 GB10 |
|---|---|---|---|---|---|---|---|
| Hopper base 기능 (TMA, cluster/DSMEM, mbarrier tx, `elect.sync`, `griddepcontrol`, `stmatrix`, `mma.sync`, `cp.async`) | O | O | O | O | O | O | O |
| `wgmma` | a | X | X | X | X | X | X |
| `setmaxnreg` | a | f | f | f | f | f | f |
| `tensormap.replace` | a | f | f | f | f | f | f |
| TMA `.multicast::cluster` | O (90a 권장) | O (100f/a 권장) | O (103f/a 권장) | O (107f/a 권장) | O (110f/a 권장) | △ (권장 목록 외; CUTLASS "GeForce multicast 없음") | △ |
| TMA `.tile::gather4`·`.im2col::w` → `.shared::cta` | X | O | O | O | O | O | O |
| TMA `.tile::gather4/scatter4`·`.im2col::w{::128}` → `.shared::cluster`, TMA `.cta_group` | X | f | f | f | f | X | X |
| `cp.async.bulk .cp_mask`, `st.bulk`, 256-bit `ld/st` | X | O | O | O | O | O | O |
| `clusterlaunchcontrol` (CLC) | X | O | O | O | O | O | O |
| CLC `.multicast::cluster::all` | X | f | f | f | f | f | f |
| `add/sub/mul/fma.f32x2`, 3-input `min/max`, 128-bit FP | X | O | O | O | O | O | O |
| **`tcgen05.mma`** (f16/tf32/f8f6f4/mxf8f6f4/mxf4/mxf4nvf4) | X | f | f | f | f | **X** | **X** |
| **Tensor Memory** (`tcgen05.alloc/ld/st/cp`) | X | f | f | f | f | X | X |
| **`cta_group::2`** (CTA pair MMA) | X | f | f | f | f | X | X |
| tcgen05 `.block16/.block32` (`.scale_vec::NX` 표기는 100a만) | X | f (a) | f | f | f | X | X |
| tcgen05 **`.kind::i8`** | X | **a** | **X** | X | **a** | X | X |
| `tcgen05.mma.sp .kind::mxf4/mxf4nvf4` | X | a | a | a | a | X | X |
| `tcgen05.shift` | X | a | a | a | a | X | X |
| `tcgen05.ld.red` | X | X | f | f | f | X | X |
| tcgen05 K=96 | X | X | a | a | X | X | X |
| tcgen05 `.kind::ti16`, UE5M3 scale | X | X | X | f | X | X | X |
| `mma.sync .kind::f8f6f4` / `mxf8f6f4` / `mxf4` / `mxf4nvf4 .block_scale` | X | X | X | X | X | **f** | **f** |
| `mma.sp .kind::mxf4/mxf4nvf4` | X | X | X | X | X | a | a |
| `ldmatrix .m16n16/.m8n16/.b8`, `stmatrix .m16n8/.b8` | X | f | f | f | f | f | f |
| `cvt` ↔ FP4/FP6/UE8M0 (`.e2m1x2` 등) | X | f | f | f | f | f | f |
| **`cvt.rs`** (HW stochastic rounding) | X | a | a | a | X | X | X |
| `redux.sync .f32` | X | f | f | f | X | X | X |
| `multimem` FP8 타입·`.acc::f16` | X | f | f | f | f | a | a |
| FP6 / FP4 Tensor Core (Table 33) | X | O | O | O | O | O | O |
| FP64 Tensor Core (Table 33) | O | O | **X** | O | X | X (정확성용 소량) | X |
| INT8 Tensor Core (Table 33) | O | O | O | **X** | O | O | O |
| DPX 네이티브 (Table 29) | O | O | O | O | X | X | X |

- † 10.7은 CUDA 13.4 developer preview로, PTX 9.4와 PG 13.4 표에만 근거한다.
- "O (… 권장)"은 PTX가 `.multicast::cluster`를 해당 타깃에 쓰라고 권고(advised)한다는 뜻이다. 권고 목록 밖의 타깃에서는 "substantially reduced performance"가 날 수 있다.
- INT8 Tensor Core 행(Table 33)과 tcgen05 `.kind::i8` 행이 10.3에서 서로 다르다. 즉 B300의 INT8은 tcgen05 경로가 아니다(추정: `mma.sync` 경로).

---

## 4. SM별 하드웨어 한도

- 출처(별도 표기가 없으면):
  - 7.5~12.x: PG 13.4 Table 30~32
  - 7.0·7.2: PG 12.9 Table 28, [Volta TG](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html)
  - Tensor Core 개수·세대: 각 백서와 PG 12.9 CC 절
- 모든 CC 공통: 레지스터 **64K × 32-bit (256 KB)/SM**, 스레드당 최대 255개, block당 최대 1024 threads, SMEM 뱅크 32개, SMEM 48 KB 초과 시 dynamic + opt-in.

| CC | 대표 GPU | TC 세대 · 개수/SM | 통합 L1+SMEM | SMEM 최대 /SM | SMEM 최대 /block | 최대 warps /SM | 최대 threads /SM | 최대 blocks /SM | FP32:FP64 (non-Tensor) | FP64 TC |
|---|---|---|---|---|---|---|---|---|---|---|
| 7.0 | V100 | 1세대 · 8 | 128 KB | 96 KB | 96 KB | 64 | 2048 | 32 | 2:1 | X |
| 7.2 | Xavier | 1세대(+INT8) · (미확인) | (미확인) | 96 KB | 96 KB | 64 | 2048 | 32 | (미확인) | X |
| 7.5 | T4, RTX 20 | 2세대 · 8 | 96 KB | 64 KB | 64 KB | 32 | 1024 | 16 | 32:1 | X |
| 8.0 | A100 | 3세대 · 4 | 192 KB | 164 KB | 163 KB | 64 | 2048 | 32 | 2:1 | O |
| 8.6 | A10, RTX 30 | 3세대 · 4 | 128 KB | 100 KB | 99 KB | 48 | 1536 | 16 | 64:1 | X |
| 8.7 | Orin | 3세대 · (미확인) | 192 KB | 164 KB | 163 KB | 48 | 1536 | 16 | 64:1 | X |
| 8.9 | L40S, RTX 40 | 4세대 · 4 | 128 KB | 100 KB | 99 KB | 48 | 1536 | 24 | 64:1 | X |
| 9.0 | H100 | 4세대 · 4 | 256 KB | 228 KB | 227 KB | 64 | 2048 | 32 | 2:1 | O |
| 10.0 | B200 | 5세대 · 4 (PG 12.9 CC 10.x 절) + TMEM 256 KB | 256 KB | 228 KB | 227 KB | 64 | 2048 | 32 | 2:1 | O |
| 10.3 | B300 | 5세대 · 4 (640 TC / 160 SM, (계산값)) + TMEM 256 KB | 256 KB | 228 KB | 227 KB | 64 | 2048 | 32 | **64:1** | **X** |
| 10.7 | Rubin (프리뷰) | (미확인) | **336 KB** | **328 KB**¹ | 327 KB | **32** | **1024** | 16 | **4:1** | O |
| 11.0 | Thor | 5세대 · (미확인), tcgen05/TMEM 지원 | 256 KB | 228 KB | 227 KB | 48 | 1536 | 24 | 64:1 | X |
| 12.x | RTX 50, RTX PRO, GB10 | 5세대 · 4 (RTX WP "Tensor Cores / SM 4 (5th Gen)"), TMEM 없음 | 128 KB² | 100 KB | 99 KB | 48 | 1536 | 24³ | 64:1 | X (정확성용 소량) |

1. 10.7에서 328 KB SMEM 구성을 쓰려면 `cudaFuncAttributeSharedMemoryMode`/launch attribute로 `cudaSharedMemoryModeAllowOversizedSharedMemory`를 켜야 한다 (PG 13.4 Table 32 각주).
2. 12.x 통합 캐시: PG 13.4 Table 32는 128 KB, PG 12.9 CC 12.0 절 본문은 "100 KB", Blackwell Tuning Guide는 "shared memory capacity per SM is 128KB"로 적는다. 128 KB는 통합 L1/SMEM이고 SMEM 최대는 100 KB로 보는 것이 표들과 일관된다.
3. 12.x blocks/SM: PG 13.4 Table 30은 24이고, PG 12.9 Table 28과 [Blackwell TG](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)는 32다. 현행 PG 13.4 값을 우선했지만 실제 GPU에서 `cudaDevAttrMaxBlocksPerMultiprocessor`로 확인할 것을 권장한다.

- FP32:FP64는 PG의 **non-Tensor** 처리량 비율이다. 대표 제품 SM 구성과 맞춰 보면 다음과 같다.
  - 데이터센터(7.0/8.0/9.0/10.0)는 2:1이다.
  - 클라이언트·임베디드(8.6/8.7/8.9/11.0/12.x)는 64:1이다. SM당 FP64 코어 2개로 정확성만 보장한다(GA102/Ada/RTX Blackwell WP).
  - Turing은 32:1이다.
  - **B300(10.3)은 데이터센터 제품인데도 64:1**이다.

---

## 5. 호환성 규칙 요약

[01-architecture.md §2.2](01-architecture.md)와 [NVIDIA 블로그](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/)를 보완하는 요점만 정리한다.

- **cubin**: 같은 major CC이면서 minor가 같거나 높은 GPU에서만 실행된다(예: `sm_80` cubin → 8.6/8.9 OK, 9.0 불가). **PTX**: 더 높은 모든 CC에서 JIT로 실행된다(onion 모델, PTX `.target` 절). Blackwell/Hopper Compatibility Guide 모두 "PTX가 들어 있으면 재빌드 없이 동작"한다고 적는다.
- **`a` 타깃**(9.0부터): 해당 CC에서만 실행되고 앞/뒤 호환이 없다. `compute_90a` PTX는 Blackwell에서 돌지 않는다.
- **`f` 타깃**(10.0부터, CUDA 12.9): 같은 패밀리의 후속 CC에서만 실행된다. PG 13.4 Table 28:

  | 컴파일 타깃 | 호환 CC |
  |---|---|
  | `compute_100f` | 10.0, 10.3, 10.7 |
  | `compute_103f` | 10.3, 10.7 |
  | `compute_107f` | 10.7 |
  | `compute_110f` | 11.0 |
  | `compute_120f` | 12.0, 12.1 |
  | `compute_121f` | 12.1 |

  - 포함 관계는 `a` ⊃ `f` ⊃ base다.
  - family 기능은 같은 패밀리 후속 세대의 `a` 타깃에서도 쓸 수 있다(PTX: "Family-specific features can be used with f-targets as well as a-targets of later generation devices in the same family"). 예를 들어 `sm_103a`는 100f 기능을 모두 포함한다.
- **패밀리 경계**: sm_10x · sm_11x · sm_12x는 서로 다른 패밀리다(PTX Table 70). Thor(11.0)는 tcgen05를 갖지만 `compute_100f` 바이너리를 실행하지 못한다(Table 28).
- **fatbin 권장 예**:
  - B200+B300+Rubin 공통 → `sm_100f`
  - Thor → `sm_110f`
  - RTX 50/PRO/GB10 → `sm_120f`
  - 이식용 PTX → `compute_100` 또는 `compute_90` 등 suffix 없는 것
  - B200 전용 `kind::i8`·`cvt.rs`·`scale_vec::NX`를 쓰려면 `sm_100a`를 따로 추가하고, 소스에서 `__CUDA_ARCH_SPECIFIC__`/`__CUDA_ARCH_FAMILY_SPECIFIC__`로 분기한다.
- **CUDA 13.x에서 드롭된 타깃**: sm_50/52/53, sm_60/61/62, sm_70/72. 오프라인 컴파일과 라이브러리(cuFFT·cuSPARSE 등) 지원이 제거됐고, 이 GPU는 12.x 툴킷으로 빌드해야 한다([Features Archive §5.2](https://docs.nvidia.com/cuda/pdf/CUDA_Features_Archive.pdf), [CUDA 13.0 RN](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)). PTX `.target` 문법 목록에는 여전히 남아 있다.

---

## 6. 연구 관점 메모 — 커널 포팅 시 무엇이 바뀌나

- **sm_80 → sm_90a**:
  - 로드 경로: `cp.async` + `__syncthreads` 파이프라인 → **TMA + mbarrier(`expect_tx`)**.
  - MMA: warp-level `mma.sync` → **`wgmma`**(SMEM descriptor 입력, 비동기).
  - 구조: producer/consumer **warp specialization + `setmaxnreg`**, cluster + TMA multicast로 L2 트래픽을 줄인다.
  - SMEM 예산: 164 → 228 KB.
  - 이 모든 이점은 `sm_90a` 전용 바이너리에서만 나온다.
- **sm_90a → sm_100a/f**:
  - **`wgmma` 코드는 재사용할 수 없다**. `tcgen05.mma`로 바꾸면서 설계가 달라진다.
    - 단일 스레드가 발행한다.
    - 누산기가 레지스터가 아닌 **TMEM**에 있다. epilogue는 `tcgen05.ld`로 읽는다.
    - `tcgen05.commit` → mbarrier로 완료를 추적한다.
    - `cta_group::2`로 256-wide 타일을 쓸 수 있다.
  - 레지스터 압박이 줄어 epilogue와 softmax 설계 여지가 커진다.
  - CLC로 persistent 스케줄링을 할 수 있다.
  - TMA와 SMEM 228 KB는 그대로다.
  - B300까지 한 바이너리로 가려면 `sm_100f`를 쓴다. 이때 `kind::i8`, `scale_vec::NX`, `cvt.rs`는 쓸 수 없다.
- **sm_90a → sm_120(f)**:
  - `wgmma`와 `tcgen05`가 모두 없으므로 **Ampere/Ada 스타일 `mma.sync` + `ldmatrix`** 로 돌아간다. 그 위에 다음이 얹힌다.
    - TMA(multicast 없음)
    - `setmaxnreg`(120f)
    - FP4/FP6 block-scale `mma.sync`(TN만)
  - CUTLASS SM120 커널이 Hopper식 pingpong/cooperative 스케줄을 `mma.sync`로 구현한 사례다.
  - SMEM 99 KB/block과 48 warps/SM 때문에 타일 크기와 스테이지 수를 줄여야 한다.
- **B200 ↔ B300 차이를 활용한 연구 포인트**:
  - INT8 GEMM(tcgen05 `kind::i8`)과 FP64 HPC 커널은 B300에서 성능 특성이 크게 다를 수 있다. B300에는 FP64 TC가 없고 FP32:FP64가 64:1이다.
  - 반대로 B300 전용 `tcgen05.ld.red`(TMEM load와 reduce 결합)와 K=96 shape는 attention·FP4 GEMM 최적화의 여지다.
- **Thor(11.0)**: tcgen05/TMEM을 갖춘 데이터센터식 ISA에 `kind::i8`까지 있는 반면, occupancy(48 warps)와 DPX는 클라이언트 쪽 특성이다. 임베디드 LLM·비전 추론 커널을 sm_100 커널에서 옮길 때 가장 가까운 대상이다.
- **Rubin 프리뷰(10.7)**: PG 표상 **SMEM 328 KB / 32 warps/SM**으로, SMEM은 커지고 동시 warp는 줄었다. 10.x용 타일 설정을 그대로 쓰면 최적이 아닐 가능성이 높다. developer preview라 수치가 바뀔 수 있으니 13.x 이후 문서로 재확인할 것.
