# sm_120 커널 작성 스택 조사 — NVIDIA 공식 계열

> 조사일: 2026-09-11. 대상: sm_120(RTX 5090 / RTX PRO 6000, CC 12.0), 해당되면 sm_121(GB10).
> 값: **O** 지원, **△** 부분 지원 또는 우회 필요, **X** 불가, **(미확인)** 1차 근거 없음.
> 근거 파일은 이 폴더의 [SOURCES.md](SOURCES.md). ISA 사실(sm_120 명령 제약)은 [06-sm-features.md](../../../06-sm-features.md) §2.2/§3.2를 따른다.
> 원래 에이전트 조사가 중단되어, 에이전트가 받아 둔 파일을 바탕으로 직접 정리했다.

---

## A. CUDA C++ (nvcc) + inline PTX + CCCL(`cuda::ptx`, `cuda::barrier`)

| 항목 | 값 | 근거 |
|---|---|---|
| F1 sm_120 지원 | **O**. CUDA 12.8+(`sm_120`, `sm_120a`), family target `sm_120f`는 CUDA 12.9+ | 01-architecture §2, CUDA 12.8 RN |
| F2 FP16/BF16 mma | **O** (`mma.sync`, `wmma`) | PTX ISA |
| F3 FP8 mma | **O** (`mma.sync` e4m3/e5m2, sm_89+ base) | PTX ISA 8.4 |
| F4 FP4/FP6 block-scaled mma | **O** — `mma.sync .kind::f8f6f4 / mxf8f6f4 / mxf4 / mxf4nvf4 .block_scale`. **`sm_120a` 또는 `sm_120f` 타깃 필수**, inline PTX로 작성(CCCL은 `mma` wrapper 없음) | 06 §2.2, `cccl-ptx-mma-less-list.txt` |
| F5 TMA | **O** — `cuda::ptx::cp_async_bulk_tensor`(base SM_90), `cuda::make_tma_descriptor`(CC 9.0+) | `cccl-ptx-cp_async_bulk_tensor.h`, `cccl-make_tma_descriptor.rst` |
| F6 mbarrier/비동기 파이프라인 | **O** — `cuda::barrier`, `cuda::ptx::mbarrier_*`, `expect_tx` | CCCL |
| F7 warp specialization | **O** (직접 구현하는 패턴) | — |
| F8 setmaxnreg | **O** — CCCL wrapper 대상에 SM_120a, SM_120f 명시 | `cccl-ptx-setmaxnreg.h` |
| F9 cluster / DSMEM | **O** (PG Table 29). multicast는 PTX상 허용되나 권장 대상 아님 | 06 §3.2 |
| F10 CLC / persistent | **O** — `clusterlaunchcontrol.try_cancel`은 base SM_100 → sm_120 사용 가능. `.multicast::cluster::all`은 120a/120f | `cccl-ptx-clusterlaunchcontrol.h` |
| F11 PDL | **O** (`griddepcontrol`, sm_90+ base) | PTX ISA |
| F12 2:4 sparse mma | **O** (`mma.sp`, inline PTX). `mma.sp .kind::mxf4/mxf4nvf4`는 `sm_120a` 전용 | 06 §2.2 |
| F13 ldmatrix/stmatrix, swizzle | **O** (신규 shape `.m16n16/.b8` 등은 120f) | 06 §3.2 |
| F14 저수준 escape hatch | **O** (native) | — |
| F15 PyTorch 연동 | `torch.utils.cpp_extension`(`load_inline`, `CUDAExtension`), `TORCH_CUDA_ARCH_LIST`로 12.0 지정. `a`/`f` 타깃 표기법은 버전별 차이 (미확인) | `pytorch-cpp_extension.py` |
| F16 autotuning | **X** (내장 없음) | — |
| F17 sm_120 레퍼런스 | Colfax NVFP4 block-scaled GEMM 튜토리얼(sm12x): FP4 dense peak의 약 60% | `sources/rtx-5090`, 03-rtx-pro-6000 §5.3 |
| F18 주의점 | 모든 기능을 쓸 수 있지만 작성량이 가장 많다. `sm_120a` cubin은 이식 불가 | 01 §2.2 |

**요점**: sm_120의 모든 ISA 기능(block-scaled `mma.sync`, TMA, `setmaxnreg`, CLC, PDL)을 쓸 수 있는 유일한 방법이다. 다만 CCCL `cuda::ptx`는 TMA·mbarrier·CLC·`setmaxnreg`만 감싸고 **MMA는 감싸지 않으므로**, Tensor Core 부분은 inline PTX나 CuTe atom을 직접 써야 한다.

---

## B. CUTLASS C++ (3.x/4.x) + CuTe

| 항목 | 값 | 근거 |
|---|---|---|
| F1 sm_120 지원 | **O** — SM120 GEMM(CHANGELOG상 3.9, 문서는 "CUTLASS 4.0 has added support"). `ArchTag = cutlass::arch::Sm120`, `CUTLASS_ARCH_MMA_SM120_SUPPORTED`/`SM121_SUPPORTED`로 게이트. `sm_120f` 컴파일은 4.4.2+ | `cutlass-79a…cu`, 01 §6 |
| F2 FP16/BF16 mma | **△** — SM120 전용 collective builder는 **F8F6F4만 지원**("Non-blockscaled collective builder only supports F8F6F4 MMA"). FP16/BF16 GEMM은 SM80 계열 커널(cp.async 기반)을 쓰는 것으로 보임 (추정) | `cutlass-sm120_mma_builder.inl` |
| F3 FP8 mma | **O** — `SM120_16x8x32_TN<f4/f6/f8 조합>` atom | `cutlass-cute-arch-mma_sm120.hpp` |
| F4 FP4/FP6 block-scaled mma | **O** — mxf8f6f4, mxf4, nvf4(`scale_vec::2X/4X`). **TN 레이아웃만**. tile shape 표(Table 16~19), FP6 출력은 leading 차원 tile 128 | `cutlass-blackwell_functionality-20260908.md` §SM120 |
| F5 TMA | **O** — `KernelTmaWarpSpecialized*Sm120` | 같은 문서 |
| F6 mbarrier/파이프라인 | **O** (multi-stage) | 같은 문서 |
| F7 warp specialization | **O** — pingpong(4 MMA warp × 2그룹) / cooperative(8 MMA warp). `KernelScheduleAuto` → cooperative | 같은 문서 |
| F8 setmaxnreg | **O** — `reg_reconfig.h`가 `__CUDA_ARCH__ == 1200`(SM120_ALL 또는 family)에서 활성 | `cutlass-arch-reg_reconfig.h` |
| F9 cluster / DSMEM | **△** — builder가 `size(ClusterShape) == 1` 강제("no programmatic multicast on this arch") | `cutlass-sm120_mma_builder.inl` |
| F10 CLC / persistent | **△** — persistent tile scheduler는 O(79a "Warp-Specialized persistent kernel"). SM120에서 CLC 기반 scheduler 사용 여부는 (미확인)(문서의 CLC 절은 SM100 대상) | 79a, 문서 |
| F11 PDL | **O** — `grid_dependency_control.h`가 1200/1210에서 GDC 활성(`CUTLASS_ENABLE_GDC_FOR_SM100` 정의 시) | `cutlass-arch-grid_dependency_control.h` |
| F12 2:4 sparse mma | **O** — `mma_sm120_sparse.hpp`, `sm120_sparse_mma_builder.inl`, `sm120_blockscaled_sparse_mma_builder.inl`, 단위 테스트(f4/f6/f8) | GitHub 저장소 트리 (링크만) |
| F13 ldmatrix/swizzle | **O** (CuTe copy atom, swizzle layout) | CuTe |
| F14 저수준 escape hatch | **O** (C++ 템플릿 + inline PTX) | — |
| F15 PyTorch 연동 | C++ 확장으로 감싸기. CUTLASS Python Operator API의 sm_120 지원은 (미확인) | — |
| F16 autotuning | **△** — CUTLASS Profiler(오프라인 탐색) | CUTLASS 문서 |
| F17 sm_120 레퍼런스 | `examples/79_blackwell_geforce_gemm`(79a NVFP4→BF16 등), grouped(`sm120_array_mma_builder`), blockwise(`sm120_blockwise_mma_builder`) builder, `test/unit/gemm/device/sm120_*` | 저장소 |
| F18 주의점 | TN만, cluster 1, 템플릿 컴파일이 무겁다. SM100 커널(tcgen05)은 그대로 쓸 수 없다 | — |

**요점**: sm_120에서 **가장 성숙한 FP8/FP4 GEMM 경로**다(block-scaled, sparse, grouped까지 있음). 반면 SM120 전용 builder가 F8F6F4 타입만 받으므로 FP16/BF16 고성능 GEMM은 별도 경로가 필요하다.

---

## C. CuTe DSL (Python, CUTLASS 4.x)

| 항목 | 값 | 근거 |
|---|---|---|
| F1 sm_120 지원 | **O** — GeForce 전용 예제(`Sm120GemmKernel`, `Sm120BlockScaledGemmKernel`). `MmaSM120BlockScaledOp.admissible_archs = [sm_120a, sm_120f, sm_121a, sm_121f]`, `CUTE_DSL_ARCH` 환경 변수로 타깃 지정. 최초 지원 버전 (미확인) | `cutedsl-nvgpu-warp-mma.py`, geforce 예제 |
| F2 FP16/BF16 mma | **O** — `MmaF16BF16Op`(sm_80+). 예제는 fp16/bf16 입력, fp32 누산 | 같은 파일, `cutedsl-geforce-dense_gemm.py` |
| F3 FP8 mma | **O** — `MmaFP8Op`(sm_89+) | `cutedsl-nvgpu-warp-mma.py` |
| F4 FP4/FP6 block-scaled mma | **O** — `MmaMXF4Op` 등(sm_120a/120f). 예제는 E2M1 입력, scale은 vec32=E8, vec16=E8 또는 E4M3(→ MXFP4/NVFP4) | geforce blockscaled 예제 |
| F5 TMA | **O** — `CopyBulkTensorTileG2SOp`(sm_90+), 두 예제 모두 TMA load/store | `cutedsl-nvgpu-cpasync-copy.py` |
| F6 mbarrier/파이프라인 | **O** — multi-stage pipeline | dense_gemm 예제 |
| F7 warp specialization | **O** — DMA warp group + MMA warp group 2개 pingpong | blockscaled 예제 |
| F8 setmaxnreg | **O** — `cute.arch.setmaxregister_increase` | dense_gemm 예제 |
| F9 cluster / DSMEM | **△** — 예제 클래스는 `cluster_shape_mnk = (1, 1, 1)` 고정. multicast op 자체는 존재 | 두 예제 |
| F10 CLC / persistent | **△** — persistent tile scheduling O. CLC의 sm_120 사용은 (미확인) | blockscaled 예제 |
| F11 PDL | **△** — Pipeline 문서에 "Programmatic Dependent Launch" 절이 있음. sm_120 검증은 (미확인) | `cute-dsl-limitations.txt` 목차 |
| F12 2:4 sparse mma | **△** — `MmaF16BF16SparseOp`(sm_80+). FP8/FP4 sparse의 sm_120 op는 (미확인) | `cutedsl-nvgpu-warp-mma.py` |
| F13 ldmatrix/swizzle | **O** — blockscaled 예제가 `ldmatrix` 사용, CuTe swizzle layout | 같은 예제 |
| F14 저수준 escape hatch | **△** — `cute.arch.*` primitive 제공. 임의 inline PTX 삽입은 (미확인) | — |
| F15 PyTorch 연동 | **O** — DLPack/프레임워크 텐서 변환, TVM-FFI 컴파일 경로 | `cutedsl-tvm_ffi_compilation.rst` |
| F16 autotuning | **X** (내장 없음, 파라미터를 직접 스윕) | — |
| F17 sm_120 레퍼런스 | `examples/python/CuTeDSL/cute/blackwell_geforce/kernel/{dense_gemm, blockscaled_gemm}`, `cute_ext/.../sm120_dense_block_scaled_gemm_persistent_pingpong.py` | 저장소 |
| F18 주의점 | layout은 32-bit shape/stride만, Windows 미지원, JIT 코드 단계 디버깅 불가. 문서의 유틸리티 목차는 "Hopper (SM90)", "Blackwell (SM100)"만 있음 | `cute-dsl-limitations.txt` |

**요점**: CUTLASS C++와 **같은 CuTe 추상화를 Python으로** 쓰면서 컴파일이 빠르다. sm_120 NVFP4 persistent pingpong 예제가 공식으로 있어 **sm_120 FP4 커널 연구의 출발점으로 가장 적합**하다.

---

## D. cuTile Python / CUDA Tile IR

| 항목 | 값 | 근거 |
|---|---|---|
| F1 sm_120 지원 | **O** — Tile IR 아키텍처 목록에 SM_120·SM_121이 **CUDA 13.1부터**. cuTile Python은 CC 8.x~12.x, driver r580+. 최신 1.6.0(2026-09-08) | `cuda-tile-AttrDefs.td`, `cutile-python-quickstart.txt`, CHANGELOG |
| F2 FP16/BF16 mma | **O** — `ct.mma`(컴파일러가 Tensor Core로 매핑) | Warp FAQ의 cuTile 설명, 문서 |
| F3 FP8 mma | **O** — `float8_e4m3fn`/`float8_e5m2` 입력(`use_fast_acc`는 Hopper 전용 옵션) | CHANGELOG 1.4.0 |
| F4 FP4/FP6 block-scaled mma | **△** — `ct.mma_scaled`, `float4_e2m1fn`·`float8_e8m0fnu`(CTK 13.3 / cuTile 1.4.0). sm_120에서의 성능·지원 조합은 (미확인). CUDA 13.1 RN은 "limited low-precision support"라고 적음 | CHANGELOG, `cuda-13.1-release-notes.txt` |
| F5 TMA | **△ 자동** — 구조화된 tile load는 "지원 GPU에서" 컴파일러가 TMA로 내림. 사용자가 직접 제어하지 않음 | `warp-faq.rst` |
| F6 mbarrier/파이프라인 | **△ 자동** (컴파일러 관리, `num_worker_warps` 힌트만) | CHANGELOG 1.4.0 |
| F7 warp specialization | **X** (직접 제어 불가, 스레드 매핑이 숨겨짐) | quickstart |
| F8 setmaxnreg | **X** (직접 제어 불가) | — |
| F9 cluster / DSMEM | (미확인) | — |
| F10 CLC / persistent | (미확인) | — |
| F11 PDL | **O** — `ct.grid_dependency_control_launch_dependents/wait`, `ct.launch(programmatic_dependent_launch=...)` (CTK 13.4 / cuTile 1.6.0) | CHANGELOG 1.6.0 |
| F12 2:4 sparse mma | (미확인) | — |
| F13 ldmatrix/swizzle | **X** (추상화로 숨겨짐) | — |
| F14 저수준 escape hatch | **X** | — |
| F15 PyTorch 연동 | **O** — 배열 인자(DLPack 호환 텐서), JAX는 `ct.jax.cutile_call` | CHANGELOG |
| F16 autotuning | **O** (1.6.0에서 개선) | CHANGELOG 1.6.0 |
| F17 sm_120 레퍼런스 | (미확인) — 성능 가이드에 sm_120 수치 없음 | `cutile-python-performance.txt` |
| F18 주의점 | CUDA 13.1(2025-12)에 나온 신생 스택이다. 스레드·메모리 매핑이 컴파일러에 숨겨져 있어 튜닝 여지가 작다. `tileiras` 버전마다 지원 GPU가 다르다(README: 13.2는 Blackwell과 Ampere/Ada) | README |

**요점**: Triton과 비슷한 수준의 tile 추상화를 NVIDIA가 직접 제공한다. sm_120을 공식 지원하고 PDL, autotuning도 있지만, **warp specialization·`setmaxnreg`·swizzle을 직접 제어할 수 없어** 최대 성능을 뽑는 도구는 아니다.

---

## E. NVIDIA Warp

| 항목 | 값 | 근거 |
|---|---|---|
| F1 sm_120 지원 | **△** — 커널을 CUDA C++로 생성해 NVRTC로 컴파일하므로 설치된 CUDA가 지원하면 동작. sm_120 명시는 (미확인) | `warp-faq.rst` |
| F2 FP16/BF16 mma | **△** — tile 연산 일부가 "Tensor Core를 쓸 수 있는 디바이스 라이브러리"를 호출 | 같은 문서 |
| F3~F4 FP8 / FP4 | (미확인) | — |
| F5 TMA | **X** — "Warp's tile load and store implementation does not currently use TMA" | 같은 문서 |
| F6~F13 | **X** (SIMT + 협력 tile 추상화, 저수준 제어 없음) | — |
| F15 PyTorch 연동 | **O** (텐서 상호 변환) | 문서 |
| F16 autotuning | **X** | — |
| F18 주의점 | 물리 시뮬레이션·기하 처리와 미분 가능 프로그래밍용이다. GEMM/attention 커널 연구용이 아니다 | — |

---

## F. Numba-CUDA

| 항목 | 값 | 근거 |
|---|---|---|
| F1 sm_120 지원 | **O** — CUDA 13 기준 CC 7.5~12.1 | `numba-cuda-installation.rst` |
| F2~F4 Tensor Core | **△** — 네이티브 API 없음. CUDA C++ 디바이스 함수를 FFI로 링크해야 함 (추론) | `numba-cuda-cuda_ffi.rst` |
| F5~F13 | **X** (FFI로 CUDA C++를 부르는 것 외에는 불가) | — |
| F14 저수준 escape hatch | **△** — `.cu` 파일/디바이스 함수 링크 | 같은 문서 |
| F15 PyTorch 연동 | **O** (`__cuda_array_interface__`) | — |
| F18 주의점 | **유지보수 모드**: CUDA 13 수명 동안 보안·치명적 버그만 수정하며, 신규 기능은 Numba-CUDA-MLIR로 이전을 권장 | `numba-cuda-docs-index.txt` |

---

## G. 라이브러리 호출 경로 (커널 작성 X, 호출 O)

| 라이브러리 | sm_120 관련 사실 | 근거 |
|---|---|---|
| **cuBLASLt** | block-scaled matmul: FP4는 `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`(NVFP4), FP8은 `VEC32_UE8M0`(MXFP8). 12.8 RN 기준 "CC 10.0 and higher"(12.0 포함). 128-element 실험 모드는 CC 10.x/11.0만. **FP8 matmul은 CC 12.x에서 TN 형식만**. **FP64 에뮬레이션(fixed-point, Ozaki 계열)이 CC 12.x를 CUDA 13.0u2+부터 지원** | `cublas-docs.txt` 541·9489·10876행, 01 §6 |
| **cuDNN frontend** | 대상이 Hopper(H100/H200)와 Blackwell(B200/GB200/GB300)로 명시되어 있다. sm_120 지원은 (미확인). FROST GEMM 엔진(block-scaled FP4/FP8)도 Blackwell 데이터센터 대상으로 보임 (미확인) | `cudnn-frontend-README.md` |
