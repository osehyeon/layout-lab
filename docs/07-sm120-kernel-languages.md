# 07. sm_120 커널 작성 언어·DSL 지원 매트릭스 (RTX 5090 / RTX PRO 6000)

> 조사일: 2026-09-11. 대상: **sm_120**(CC 12.0, RTX 5090 / RTX PRO 6000 Blackwell). sm_121(GB10)은 대부분 같은 `sm_120f` 패밀리라 함께 적는다.
> 도구별 상세 근거 표(F1~F18)는 [`sources/kernel-languages/nvidia/findings.md`](sources/kernel-languages/nvidia/findings.md)와 [`sources/kernel-languages/community/findings.md`](sources/kernel-languages/community/findings.md)에 있고, 원본 파일·URL·고정 커밋은 각 폴더의 `SOURCES.md`에 있다.
> sm_120 ISA 자체의 제약(tcgen05/TMEM 없음, `mma.sync` block-scale, SMEM 99 KB/block, multicast 비권장)은 [06-sm-features.md](06-sm-features.md) §2.2·§3.2를 전제로 한다.
> 표기: **O** 지원, **△** 부분 지원 또는 우회 필요, **X** 불가, **?** 미확인. 셀 근거는 소스 코드·릴리스 노트 우선이며, GitHub 이슈는 **2차 출처**다.

## 0. TL;DR

1. **sm_120에서 모든 기능(block-scaled FP4 `mma.sync`, TMA, `setmaxnreg`, CLC, PDL, sparse)을 쓸 수 있는 것은 CUDA C++/PTX, CUTLASS, CuTe DSL뿐이다.** 공식 sm_120 FP4 예제가 있는 것도 CUTLASS(`examples/79_blackwell_geforce_gemm`)와 CuTe DSL(`blackwell_geforce/` 예제)이다.
2. **Triton은 sm_120을 지원하지만 "Ampere식 MMAv2 + sm_120a" 수준이다.**
   - 되는 것: 네이티브 `tl.dot_scaled`(FP4, MXFP8)는 3.6.0부터 된다.
   - 안 되는 것:
     - cluster는 코드에서 명시적으로 막혀 있다(`supportClusterOps()`가 12.x 제외).
     - warp specialization은 sm_120에서 테스트되지 않는다.
     - block-scaled 튜토리얼은 CC 10/11 전용이다.
3. **TileLang이 커뮤니티 도구 중 sm_120 경로가 가장 명시적이다.**
   - README에 "SM70 through SM120"이라고 적혀 있고, SM120 NVFP4 예제와 자동 warp specialization이 있다.
   - 다만 FP4는 NVFP4(`mxf4nvf4` + `ue4m3`)만 되고, MXFP8/MXFP4는 PR 단계다.
4. **"Blackwell 지원"이라고 적힌 여러 도구가 실제로는 sm_100 전용이다.** 대상: ThunderKittens 커널, Triton TLX, Gluon의 `tcgen05_mma_scaled`, JAX Mosaic GPU의 `tcgen05`, Inductor의 Blackwell 템플릿(`100 ≤ arch < 110`).
5. **고수준 도구는 FP4를 결국 라이브러리로 넘긴다.** MAX(Mojo)는 sm_120 block-scaled matmul을 cuBLASLt로 보내고, cuTile은 warp 수준 제어를 숨긴다.

---

## 1. 도구 × 기능 매트릭스 (F1~F12)

| 도구 (확인 버전) | F1 sm_120 지원 | F2 FP16/BF16 MMA | F3 FP8 MMA | F4 FP4/FP6 block-scaled | F5 TMA | F6 mbarrier·다단 파이프라인 | F7 warp specialization | F8 `setmaxnreg` | F9 cluster·DSMEM | F10 CLC·persistent | F11 PDL | F12 2:4 sparse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **CUDA C++ + PTX + CCCL** (CUDA 12.8+/13.x) | O | O | O | O (inline PTX, `sm_120a/f`) | O | O | O (직접 구현) | O | O (multicast 비권장) | O | O | O |
| **CUTLASS C++** (4.x) | O | △ (SM120 builder는 F8F6F4 전용) | O | O (TN만) | O | O | O (pingpong/cooperative) | O | △ (cluster 1 강제) | △ (persistent O, CLC ?) | O | O |
| **CuTe DSL** (CUTLASS 4.x) | O | O | O | O (NVFP4/MXFP4) | O | O | O | O | △ (예제 1×1×1) | △ | △ | △ (F16만 확인) |
| **cuTile / CUDA Tile** (1.6.0, CUDA 13.1+) | O | O | O | △ (`mma_scaled`, sm_120 성능 ?) | △ 자동 | △ 자동 | X | X | ? | ? | O | ? |
| **Triton** (3.8.0) | O | O | O (#11320: RTX 50에서 반속) | △ (같은 포맷끼리만 네이티브) | O | O | △ (sm_120 미검증) | △ | X | △ | O | X |
| **Gluon** (Triton 3.8.0) | △ (`mma_v2` 경로) | O | ? | X (`tcgen05_mma_scaled`는 sm_100) | O | O | O | △ | X | △ | ? | X |
| **TLX** (fbtriton) | X (sm90/sm100 전용) | ? | ? | ? | ? | ? | ? | ? | ? | X | ? | X |
| **TileLang** (0.1.14) | O | O | O | △ (NVFP4만) | O | O | O (자동 WS 패스) | O | O/△ | △ | O | △ |
| **ThunderKittens** (`cb21f34`) | △ (primitive만, 커널 없음) | O | O | X | O | O | △ | O | O | O | O | X |
| **Helion** (1.4.0) | O (Triton 백엔드, TileIR 10/12) | O | O | △ | O | O | △ | X | X | △ | ? | X |
| **torch.compile / Inductor** (PyTorch 2.14) | O | O | O/△ | △ (torchao 경유) | △ | O | X | X | X | △ | X | X |
| **JAX Pallas** (0.11.1) | △ (Mosaic `tcgen05`는 10/11만) | △ | △ (`mma.sync` FP8) | X | O | O | △ | ? | ? | ? | ? | X |
| **Mojo / MAX** (26.5) | △ ("development" 호환) | ? | ? | △ (→ cuBLASLt) | ? | ? | ? | ? | ? | ? | ? | ? |
| *Warp / Numba-CUDA* | △ / O | △ / △(FFI) | ? / X | ? / X | X / X | X | X | X | X | X | X | X |

### 핵심 셀의 근거 (원본 확인분)

| 주장 | 근거 |
|---|---|
| Triton은 12.x에서 cluster 비활성 | `triton-TargetFeatures.h`: `supportClusterOps() { return computeCapability >= 90 && computeCapability / 10 != 12; }` |
| Triton은 sm_120에서 MMAv2 + `sm_120a` | `triton-AccelerateMatmul.cpp` "Exclude consumer Blackwell (sm120)", `triton-nvidia-backend-compiler.py` `suffix = "a" if capability >= 90` |
| Triton 3.6.0 sm_120 기능 | `triton-3.6.0-release-notes.md` "SM120 Features": native FP4 scaled_dot(#8494), native MXFP8 scaled_dot(#7918 등), TMA gather4 on sm_120/121(#8498) |
| TileLang sm_120 | `tilelang-README.md` "code paths from SM70 through SM120", `tilelang-mma_block_scale.h` "Only kind::mxf4nvf4 is supported", "requires sm_120a and CUDA 12.8" |
| Inductor Blackwell 템플릿은 sm_100 계열만 | `pytorch-inductor-cuda_env.py` `is_datacenter_blackwell_arch()`: `100 <= arch < 110` |
| Helion TileIR | `helion-_compat.py` "only device with compute capability 10.x and 12.x support tileir backend", autotune fallback `sm120 → sm100 → sm90` |
| JAX Mosaic GPU `tcgen05` | `jax-mosaic-gpu-tcgen05.py` `assert arch.major in {10, 11}` |
| ThunderKittens sm_120 | `thunderkittens-util.cuh` `KITTENS_SM120` → `MAX_SHARED_MEMORY = 99 * 1024`, FP8 `mma.sync` 가드. sm_120 포트 PR #203은 머지되지 않음(2차) |
| MAX sm_120 FP4 → cuBLASLt | `max-block_scaled_quantization.mojo`: "consumer Blackwell (sm_120 / sm_121) has no SM100 kernel, so it is skipped at compile time and control falls straight to the vendor fallback" |
| CUTLASS SM120 제약 | `cutlass-sm120_mma_builder.inl`: "no programmatic multicast on this arch", "Only TN layout is supported", "Non-blockscaled collective builder only supports F8F6F4 MMA" |
| CuTe DSL sm_120 block-scaled | `cutedsl-nvgpu-warp-mma.py` `MmaSM120BlockScaledOp.admissible_archs = [sm_120a, sm_120f, sm_121a, sm_121f]` |
| cuTile sm_120 | `cuda-tile-AttrDefs.td` `SM_120`(CUDA 13.1), `cutile-python-quickstart.txt` "compute capability 8.x … 12.x" |

---

## 2. 사용성 비교 (F14~F16 중심)

| 도구 | 추상화 수준 | 저수준 escape hatch | PyTorch 연동 | autotuning | 컴파일·반복 속도 |
|---|---|---|---|---|---|
| CUDA C++ + PTX | 스레드/명령 | native | `torch.utils.cpp_extension` | X | 느림(nvcc) |
| CUTLASS C++ | 템플릿 컴포넌트 | inline PTX | C++ 확장 | △ (CUTLASS Profiler) | 매우 느림(템플릿) |
| CuTe DSL | CuTe layout/atom (Python) | `cute.arch.*` | DLPack, TVM-FFI | X | 빠름(JIT) |
| cuTile | tile (스레드 숨김) | X | DLPack, JAX FFI | O | 빠름 |
| Triton / Helion | tile (block 단위) | `inline_asm_elementwise` 수준 | 네이티브(`torch.compile` 기본 백엔드) | O (`@triton.autotune`, Helion은 autotune 중심) | 빠름 |
| Gluon | 명시적 layout·warp (Triton 방언) | 제한적 | Triton과 같음 | ? | 빠름 |
| TileLang | tile + 명시적 스케줄 | T.* intrinsic | DLPack | O | 빠름 |
| ThunderKittens | C++ tile primitive | CUDA C++ | C++ 확장 | X | 느림 |

---

## 2-1. 제어 범위 비교: Triton vs CuTe DSL (sm_120)

> 기준: Triton 3.8.0(main `66aa2f8`), CUTLASS 4.x CuTe DSL. 근거는 [community/findings.md](sources/kernel-languages/community/findings.md) §1·§2, [nvidia/findings.md](sources/kernel-languages/nvidia/findings.md) §C, [08-helion-triton-gluon.md](08-helion-triton-gluon.md) §2.
> 범례:
> - **●** 사용자가 직접 제어
> - **◐** 힌트나 옵션만 주고 결정은 컴파일러가 함
> - **○** 컴파일러가 결정하며 사용자 제어 불가
> - **✕** 기능을 쓸 수 없음
> - **?** sm_120에서 미확인

| 분류 | 항목 | Triton | CuTe DSL | Gluon에서는 |
|---|---|---|---|---|
| **데이터 배치** | 레지스터 tile의 스레드↔원소 매핑(layout) | ○ `coalesce`·`remove_layout_conversions` 패스가 결정 | ● TV layout, tiled MMA/copy | ● `BlockedLayout` 등 |
| | shared memory layout·swizzle | ○ `NVMMASharedLayout` 자동 | ● smem layout·swizzle 명시 | ● |
| | shared memory 할당·재사용 | ○ | ● | ● |
| **데이터 이동** | global→shared 경로(`cp.async` 대 TMA) | ◐ tensor descriptor를 쓰면 TMA, lowering은 컴파일러 | ● `CopyG2SOp` 대 `TmaCopyOp` | ● |
| | TMA box·swizzle 모드 | ◐ block shape만 지정, swizzle 자동 | ● | ● |
| | TMA scatter | ✕ ptxas 실패(#11344, 2차) | ? | — |
| | shared→레지스터 로드(`ldmatrix` 등) | ○ `supportLdMatrix` 게이트로 자동 | ● copy atom 선택 | ● |
| | epilogue 경로(레지스터→shared→TMA store) | ○ | ● | ● |
| **연산** | MMA 명령·shape 선택 | ○ `accelerate_matmul`이 MMAv2 고정 | ● `MmaF16BF16Op`·`MmaFP8Op`·`MmaMXF4Op` + atom layout | ● `mma_v2`만 |
| | 누산 정밀도(FP16 대 FP32) | ● `tl.dot(..., out_dtype=)` | ● | — |
| | FP4/MXFP8 block-scaled 조합 | ◐ `tl.dot_scaled`는 같은 포맷끼리만 native, `num_ctas==1` | ● NVFP4 / MXFP4 선택 | ✕ sm_120 API 없음 |
| | scale factor layout | ○ 컴파일러가 재배치 | ● `sm120_make_smem_layout_sfa/sfb` | — |
| | 2:4 sparse `mma.sp` | ✕ lowering 없음 | ◐ F16/BF16 op만 확인, FP8/FP4는 ? | — |
| **동기·스케줄** | mbarrier 배치·동기화 지점 | ○ pipeliner가 삽입 | ● | ● |
| | 파이프라인 stage 수·순서 | ◐ `num_stages` 힌트, 스케줄은 컴파일러 | ● | ● |
| | warp specialization(역할·warp 수) | ◐ `warp_specialize=True`만, sm_120 미검증(#10284, 2차) | ● DMA warp group과 MMA warp group을 직접 구성 | ● `gl.warp_specialize` |
| | 역할별 레지스터 재분배(`setmaxnreg`) | ○ 노출 안 됨(`maxnreg`는 커널 전체 상한) | ● `setmaxregister_increase/decrease` | ◐ `worker_num_regs` |
| | persistent 루프·tile 순회 순서 | ● 커널 안에서 직접 작성 | ● tile scheduler | — |
| | CLC | ◐ `clc=True`, 코드상 12.x 허용, 실측 ? | ? | — |
| | PDL(`griddepcontrol`) | ● `gdc_wait`/`gdc_launch_dependents` + `launch_pdl` | ◐ 문서에 있음, sm_120 ? | — |
| **실행 구성** | cluster·DSMEM | ✕ `supportClusterOps()`가 12.x 차단 | ◐ API는 있으나 예제는 1×1×1(multicast는 sm_120 비권장) | ✕ |
| | CTA당 warp 수 | ● `num_warps` | ● | — |
| | 컴파일 타깃(`sm_120a`/`sm_120f`) | ○ 자동 `sm_120a`, `f` 선택 불가 | ● `CUTE_DSL_ARCH` | ○ |
| **기타** | inline PTX | ◐ elementwise만 | ◐ `cute.arch.*`, 임의 PTX는 ? | — |
| | autotuning | ● `@triton.autotune` | ✕ 없음 | — |
| | PyTorch 연동 | ● 네이티브 | ● DLPack / TVM-FFI | — |

**요약**
- **Triton이 제어하지 못하는 것(○·✕)**은 대부분 성능을 좌우하는 하드웨어 결정이다: layout, shared memory 배치, MMA 명령, 동기화 지점, 레지스터 재분배, cluster, sparse, 컴파일 타깃. sm_120에서는 warp specialization 미검증과 혼합 포맷 FP4 미지원이 더해진다.
- **CuTe DSL이 제어하지 못하거나 불확실한 것**은 편의 기능 쪽이다: autotuning 없음, sm_120에서 CLC·PDL과 FP8/FP4 sparse는 미확인.
- **Gluon**은 Triton의 ○ 항목 대부분을 ●로 바꾼다. 하지만 sm_120에서 FP4 block-scaled MMA가 없고 cluster도 막혀 있다.
- 따라서 **알고리즘 수준 연구는 Triton, 하드웨어 수준 최적화(peak 근접)는 CuTe DSL**로 나누고, Triton 결과를 기준선으로 삼아 부족한 커널만 CuTe DSL로 내리는 방식을 권한다([09-research-plan.md](09-research-plan.md) Phase 1).

---

## 3. 용도별 추천 (sm_120 기준)

| 목적 | 1순위 | 대안 | 이유 |
|---|---|---|---|
| **NVFP4/MXFP4 GEMM 최대 성능** | CUTLASS C++ (79a 계열 커스터마이즈) | CuTe DSL blockscaled pingpong 예제 | block-scaled `mma.sync` + TMA + WS + `setmaxnreg`를 모두 쓰는 공식 경로 |
| **FP4 커널 연구·수정 (빠른 반복)** | **CuTe DSL** | TileLang | Python JIT로 CuTe atom을 직접 다룬다. sm_120 공식 예제가 있다 |
| **Attention 등 새 연산자 프로토타입** | Triton | TileLang, Helion | 생산성과 PyTorch 연동이 가장 좋다. 다만 FP8 반속(#11320), cluster 불가, WS 미검증을 감안해야 한다 |
| **NVFP4 attention / 혼합 커널** | TileLang | CUDA C++ | NVFP4 block-scale MMA와 자동 WS가 있다. MXFP 계열은 PR 대기 중이고 JIT hang 이슈(#2328, 2차)가 있다 |
| **B200(sm_100)으로 이식할 코드** | CUTLASS / CuTe DSL | — | 같은 추상화로 SM100 커널(tcgen05)도 제공한다. Triton·TileLang도 두 계열을 모두 타깃하지만 내부 경로가 다르다 |
| **라이브러리 호출로 충분한 경우** | cuBLASLt (NVFP4 `VEC16_UE4M3`, MXFP8 `VEC32_UE8M0`) | torchao, MAX | FP8은 CC 12.x에서 TN만 된다. MAX도 sm_120에서는 cuBLASLt를 부른다 |
| **FP64가 필요한 HPC** | cuBLAS FP64 에뮬레이션 | 직접 Ozaki 구현 | CC 12.x를 CUDA 13.0u2+부터 지원한다 |

**피할 것 (sm_120 기준)**: TLX, ThunderKittens(커널 없음), Gluon `tcgen05_*`, JAX Mosaic GPU `tcgen05`, Warp·Numba-CUDA(Tensor Core 연구용 아님). Inductor의 Blackwell 전용 템플릿과 CuTe/CUTLASS 백엔드 경로는 sm_100 계열에서만 켜진다.

---

## 4. sm_120 공통 주의점

1. **SMEM 99 KB/block**: Triton 등 sm_90/sm_100 기준 설정은 "out of resources"가 난다. 타일 크기와 `num_stages`를 줄여야 한다(ThunderKittens도 `MAX_SHARED_MEMORY = 99 KB`로 둔다).
2. **FP32 누산 FP8/FP16은 GeForce(RTX 5090)에서 반속**이다(00-comparison §2). 같은 커널이라도 RTX PRO 6000에서는 2배 빠를 수 있으므로, 벤치마크에 GPU를 명시해야 한다. Triton #11320(2차)도 이 영향으로 보인다.
3. **cluster/multicast는 쓰지 않는 것이 기본**이다(CUTLASS·Triton 모두 막는다). TileLang의 cluster/TMA 문서는 요구사항에 "RTX 5090"을 포함하고 multicast(`cluster_mask`)도 설명한다. 하지만 이는 CUTLASS의 "GeForce에는 multicast 없음"과 충돌하며, 실효 이득은 미확인이다.
4. **Triton FP8 GEMM 우회**: RTX 50에서 FP8 `tl.dot`(FP32 누산)은 반속이다. 같은 데이터를 unit scale(E8M0 = 127)의 `tl.dot_scaled`(MXFP8, `kind::mxf8f6f4`)로 돌리면 5090에서 약 1.55배 빨랐다는 보고가 있다(#11320, 2차). 수정 PR #11386은 main에 반영되지 않았다.
5. **block-scaled MMA는 `sm_120a` 또는 `sm_120f` 타깃이 필수**다. Triton과 JAX는 `a` 타깃을 자동 선택하고, CuTe DSL은 `CUTE_DSL_ARCH`로 지정한다.
6. **버전이 빠르게 바뀐다**: 이 문서의 커뮤니티 도구 셀은 위 표의 고정 커밋 기준이다. Triton main의 MXFP8 수정(#11386)처럼 머지 대기 중인 PR이 여럿이므로, 사용 전에 해당 버전의 릴리스 노트를 재확인할 것.
