# sm_120 커널 작성 언어 조사 — 커뮤니티 / 서드파티 DSL (조사 메모)

- 작성일: 2026-09-11. 대상: `sm_120` (GeForce RTX 50, RTX PRO Blackwell, CC 12.0). GB10 `sm_121`은 해당될 때만 따로 적었다.
- 범위: OpenAI Triton, Triton Gluon, Meta TLX, TileLang, ThunderKittens, Helion, torch.compile/Inductor, JAX Pallas, Mojo/MAX 등. NVIDIA 공식 스택(CUDA C++/PTX, CUTLASS/CuTe, CuTe DSL, cuTile, Warp, Numba)은 [`../nvidia/findings.md`](../nvidia/findings.md)에 있다.
- 범례: **O** 지원 / **△** 부분 지원·우회 필요 / **X** 미지원 / **(미확인)** 1차 근거 없음. GitHub issue·PR·블로그는 **2차 출처**로 표시했다. "코드상"은 소스의 capability 게이트를 읽고 판단했다는 뜻이고, 실제 sm_120 하드웨어 검증 근거는 아니다.
- 하드웨어 전제(06-sm-features.md §2.2/§3.2, 01-architecture.md §2와 같음): sm_120에는 tcgen05·TMEM·wgmma·2-CTA MMA가 없다. MMA는 warp 단위 `mma.sync`와 `.kind::f8f6f4`/`mxf8f6f4`/`mxf4`/`mxf4nvf4 .block_scale`(sm_120f/a)로 한다. TMA·cluster·DSMEM·`griddepcontrol`·CLC·`setmaxnreg`(f)는 있다. TMA multicast는 권장 대상이 아니다. SMEM은 99 KB/block이다.
- 이 파일은 조사 에이전트의 초안을 그대로 옮긴 것이다. 07 문서에 인용한 핵심 게이트(Triton `supportClusterOps`, MMAv2 선택, 3.6.0 SM120 기능, TileLang NVFP4 전용, Inductor `is_datacenter_blackwell_arch`, Helion TileIR, JAX `tcgen05`, ThunderKittens 99 KB, MAX cuBLASLt fallback, torchao 게이트)는 저장된 원본 파일에서 다시 확인했다.

## 확인한 버전·커밋

| 프로젝트 | 확인한 릴리스 | 확인한 소스 커밋 (clone 시각) |
|---|---|---|
| triton-lang/triton | **v3.8.0** (2026-08-28). PyTorch **2.14.0**(2026-09-02)이 `triton_version.txt = 3.8.0`으로 고정 (`release/2.14` 브랜치) | main `66aa2f8a62` (2026-09-10) |
| facebookexperimental/triton (fbtriton, TLX) | – (PyPI `fbtriton`, `triton-utlx`) | `fd4d8ce1b7` (2026-09-10) |
| tile-ai/tilelang | **v0.1.14** (2026-09-02) | `85fd8fc2d3` (2026-09-10) |
| HazyResearch/ThunderKittens | TK 2.0 (2026-01-11, README) | `cb21f34cdd` (2026-09-10) |
| pytorch/helion | **v1.4.0** (2026-07-29) | `cfb135ef8d` (2026-09-11) |
| pytorch/pytorch | **v2.14.0** (2026-09-02) | main `31527a43db` (2.15.0a0, 2026-09-11) |
| pytorch/ao (torchao) | v0.18.0 (2026-08-03) | `3005bc1d24` (2026-09-11) |
| jax-ml/jax | jax-v0.11.1 (2026-08-17) | `5570f3d6fe` (2026-09-10) |
| modular/modular (Mojo/MAX) | MAX v26.5.0 (2026-08-11) | `da4f8b76ac` (2026-09-11) |

---

## 1. OpenAI Triton (upstream `triton-lang/triton`)

| 항목 | 값 | 근거 |
|---|---|---|
| F1 sm_120 공식 지원 + 최소 버전 | **O**. Blackwell 지원은 3.3(PyTorch 2.7 동봉)부터다. sm_120 MMA 경로는 MMAv2(`mma.sync`)로 고정된다(`getMMAVersionSafe`: `computeCapability < 130 → {2}`, 주석 "Exclude consumer Blackwell (sm120)"). FP8 native는 **3.5.0**, native `dot_scaled`와 TMA gather4는 **3.6.0**부터다. 컴파일 타깃은 `sm_120a`다(`sm_arch_from_capability`: capability ≥ 90이면 suffix `"a"`). 따라서 `sm_120a` 전용 명령(`mma ... .block_scale`)을 낼 수 있다. `sm_120f`를 고르는 옵션은 없다(주석 "TODO: Handle non-"a" sms"). GB10(sm_121) native block-scaled dot은 PR #10010(2026-04-13 머지)으로 들어갔다 | [compiler.py L110-113](https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/third_party/nvidia/backend/compiler.py), [AccelerateMatmul.cpp L44-60](https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp), [3.5.0 RN](https://github.com/triton-lang/triton/releases/tag/v3.5.0) "#7409 Don't promote fp8 MMAv2 dot inputs for sm120", [3.6.0 RN](https://github.com/triton-lang/triton/releases/tag/v3.6.0) "SM120 Features", [PyTorch 2.7 블로그](https://pytorch.org/blog/pytorch-2-7/), [#10010](https://github.com/triton-lang/triton/pull/10010) (2차) |
| F2 FP16/BF16 TC mma | **O**. `tl.dot`는 `mma.sync.m16n8k16`(MMAv2)로 내려간다. wgmma/tcgen05 경로는 타지 않는다 | AccelerateMatmul.cpp L44-60, [MMAv2.cpp](https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp) |
| F3 FP8 mma | **O** (3.5.0+). `mmav2SupportsFp8Operands()`는 cc 89와 12.x에 true다. 주석: "sm120 has hardware support for fp8 operands w/ mmav2". 쓰는 명령은 `mma.sync.m16n8k32 ... e4m3/e5m2`이고 f32·f16 누산 둘 다 있다. 3.4 이전에는 fp16으로 upcast되어 느렸다(#7188). ⚠ 일반 FP8 `mma.sync` + FP32 누산은 RTX 50에서 8-bit 속도의 절반이고, `kind::mxf8f6f4` + unit scale을 쓰면 5090에서 1.55배 빠르다는 보고가 있다(#11320). 이를 반영한 커밋 `fcf734b`(2026-08-20, PR #11386)는 PR로 머지되지 않았고, 2026-09-10 main에서 해당 코드가 보이지 않는다(revert 여부 (미확인)) | AccelerateMatmul.cpp L1003-1009, MMAv2.cpp L366-383, [#7188](https://github.com/triton-lang/triton/issues/7188), [#11320](https://github.com/triton-lang/triton/issues/11320), [#11386](https://github.com/triton-lang/triton/pull/11386) (모두 2차) |
| F4 FP4/FP6 block-scaled (NVFP4/MXFP4/MXFP8) | **△**. `tl.dot_scaled`의 sm_120 native lowering(`ScaledBlockedToMMA`, cc/10==12 전용)이 지원하는 조합은 셋이다. **MXFP8×MXFP8**: `kind::mxf8f6f4.block_scale.scale_vec::1X ... ue8m0`. **MXFP4×MXFP4**: `kind::mxf4nvf4 ... scale_vec::2X ... ue8m0`. **NVFP4×NVFP4**: `kind::mxf4nvf4 ... scale_vec::4X ... ue4m3`. 조건은 `num_ctas==1`, FP4는 K-packed operand(MN-packed는 decomposition fallback, #10726), scale 두 개 모두 필요하다. **Mixed(FP8×FP4, bf16×mx)는 native가 아니다**("TODO: Enable mixed-precision mxfp for sm120" → decompose/upcast 경로). FP6 X. 튜토리얼 `10-block-scaled-matmul.py`는 cc 10/11에서만 돈다(sm_120 예제 아님) | AccelerateMatmul.cpp L715-810, MMAv2.cpp L386-414, [tutorial 10 L142-150](https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/python/tutorials/10-block-scaled-matmul.py), [#7918](https://github.com/triton-lang/triton/pull/7918), [#8494](https://github.com/triton-lang/triton/pull/8494), [#10726](https://github.com/triton-lang/triton/pull/10726) (2차) |
| F5 TMA | **O** (scatter 제외). `tl.make_tensor_descriptor` / host `TensorDescriptor`가 동작한다(`add_tma_lowering`은 cc//10 ≥ 9). gather4는 3.6.0부터(#8498: "All other TMA features except for cluster-related ones are supported on sm_120"). **TMA scatter는 X**: 테스트는 skip하지만 컴파일러는 `tile::scatter4`를 내보내 ptxas에서 실패한다(#11344 open, 거부 PR #11363 open). multicast X(cluster 미지원) | compiler.py L360-361, [test_tensor_descriptor.py L1487](https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/python/test/unit/language/test_tensor_descriptor.py), [#8498](https://github.com/triton-lang/triton/pull/8498), [#11344](https://github.com/triton-lang/triton/issues/11344) (2차) |
| F6 mbarrier / 비동기 파이프라인 | **O**. `num_stages` software pipeliner(cp.async 또는 TMA + mbarrier)를 쓴다. sm12x는 cc//10 ≥ 10 분기(Blackwell 파이프라인)를 탄다. 테스트 주석: "sm12x has no async dot, so its pipeline is one stage shorter". ⚠ SMEM 99 KB 때문에 큰 블록이나 많은 stage에서 "out of resource: shared memory, … Hardware limit: 101376"이 난다 | compiler.py L334-348, test_tensor_descriptor.py L820-842, [#8182](https://github.com/triton-lang/triton/issues/8182) (2차) |
| F7 warp specialization | **△**. `tl.range(..., warp_specialize=True)`로 호출하는 `add_warp_specialize` 패스는 sm12x에서도 실행된다(cc//10 ≥ 10 분기). 하지만 공식 테스트는 `is_hopper_or_blackwell()`(cc 9, 10, 11)로만 게이트되어 **sm_120은 테스트 대상이 아니다**. RTX 5070에서 `warp_specialize=True`로 컴파일이 실패하는 이슈(#10284, 3.6.0)가 open 상태다 | compiler.py L343, [test_warp_specialization.py L14-29](https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/python/test/unit/language/test_warp_specialization.py), `_internal_testing.is_blackwell()` = cc[0] in [10,11], [#10284](https://github.com/triton-lang/triton/issues/10284) (2차) |
| F8 setmaxnreg | **△**. `tl` 언어에서는 직접 노출되지 않는다. WS lowering(`ConvertWarpSpecializeToLLVM`)이 파티션별 레지스터 수가 지정되면 `NVVM::SetMaxRegisterOp`를 낸다. sm_120 전용 게이트나 테스트는 없다. `maxnreg` 옵션은 PTX `.maxnreg`(커널 전체 상한)로 별개다 | ConvertWarpSpecializeToLLVM.cpp L120-126, compiler.py L122-124 |
| F9 cluster / DSMEM | **X** (현 main). `TargetFeatures::supportClusterOps()`가 `computeCapability/10 != 12`로 12.x를 제외한다. 유효한 multi-CTA 커널이 거부되는 이슈 #10973이 open이고, 허용 PR #10974는 머지되지 않고 닫혔다. `dot_scaled` sm120 경로도 `numCTAs != 1`이면 실패한다 | [TargetFeatures.h L36-38](https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/include/triton/Dialect/TritonNvidiaGPU/IR/TargetFeatures.h), [#10973](https://github.com/triton-lang/triton/issues/10973), [#10974](https://github.com/triton-lang/triton/pull/10974) (2차) |
| F10 CLC / persistent | **△**. persistent 커널은 O(tutorial 09, TMA persistent). CLC(`clc=True`)는 옵션 검사가 `capability < 100`만 막고, `add_lower_clc`는 cc//10 ≥ 10에서 돈다. `supports_clc()` = cc[0] ≥ 10이라 코드상 12.x도 허용된다. PTX상 CLC는 base sm_100 기능이라 sm_120에 있다(06 문서). 다만 sm_120 실측 근거는 없다 | compiler.py L226-227·L362-363, `_internal_testing.supports_clc`, [tutorial 09](https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/python/tutorials/09-persistent-matmul.py) L72-73 |
| F11 PDL | **O**. `triton.language.extra.cuda.gdc_wait()` / `gdc_launch_dependents()`(`griddepcontrol`)를 쓰고 launch에 `launch_pdl=True`를 준다. tutorial 11은 cc ≥ 9에서 동작한다 | [gdc.py](https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/third_party/nvidia/language/cuda/gdc.py), tutorial 11 L22-23 |
| F12 2:4 sparse mma | **X**. upstream Triton에 sparse dot/`mma.sp` lowering이 없다(grep 결과 없음) | 소스 grep (`mma.sp`, `sparse_dot`) |
| F13 ldmatrix/stmatrix, swizzle | **△**. 컴파일러가 자동으로 고른다: `supportLdMatrix` ≥ 75, `supportStMatrix` ≥ 90, `supportLdStMatrixB8` ≥ 100(12.x 포함). SMEM swizzle은 `NVMMASharedLayout`이 자동으로 정한다. `tl`에서 사용자가 직접 제어할 수는 없다(Gluon에서는 가능) | TargetFeatures.h L42-44 |
| F14 inline PTX | **O** (elementwise 한정). `tl.inline_asm_elementwise` | [core.py L3489](https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/python/triton/language/core.py) |
| F15 PyTorch 연동 | **O**. `torch.Tensor`를 그대로 인자로 넘긴다. PyTorch 휠에 Triton이 동봉된다(2.14 → 3.8.0). torch.compile의 기본 GPU codegen이다 | [release/2.14 triton_version.txt](https://github.com/pytorch/pytorch/blob/release/2.14/.ci/docker/triton_version.txt) |
| F16 autotuning | **O**. `@triton.autotune`(동시 autotune thread-safety 이슈 #11494 open) | [#11494](https://github.com/triton-lang/triton/issues/11494) (2차) |
| F17 sm_120 예제·% of peak | `test_core.py::test_scaled_dot`에 sm_120 분기가 있다(tolerance 완화). 튜토리얼 03/09는 공통 경로다. 성능 수치는 2차 출처뿐이다: vLLM Llama3-8B e2e(5090)에서 mxfp8 native dot 44.45 s vs emulation 76.44 s(#7918). FP8을 `dot_scaled` unit scale로 돌리면 `tl.dot` 대비 1.55배(#11320). **% of peak: (미확인)** | [test_core.py L4305-4556](https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/python/test/unit/language/test_core.py), [#7918](https://github.com/triton-lang/triton/pull/7918), [#11320](https://github.com/triton-lang/triton/issues/11320) (2차) |
| F18 성숙도/주의점 | ① 99 KB SMEM OOR(#8182). ② TMA scatter ptxas 실패(#11344). ③ cluster/multi-CTA 불가(#10973). ④ WS 미검증·컴파일 실패(#10284). ⑤ FP8 `tl.dot` 절반 속도(#11320). ⑥ mixed mxfp는 native 아님. ⑦ 3.3~3.4 시절 sm_120 codegen/설치 문제(#6216, #6859, #7550 "dot_scaled actually using fp16 mma on the 5090"). ⑧ `triton_kernels`(MoE 라이브러리) sm120 지원 PR #8484(sm80 fallback)는 머지되지 않았다 | 각 이슈 링크 (2차) |

**sm_120에서 Triton을 쓸 때 요점.** sm_120에서 Triton은 "Ampere/Ada식 MMAv2 + TMA descriptor + Blackwell block-scaled `mma.sync`" 컴파일러로 동작한다. B200용 tcgen05·TMEM·2-CTA·cluster 최적화(WS 튜토리얼, block-scaled 튜토리얼 10, TLX 커널)는 그대로 가져올 수 없다. 가장 효과가 큰 기능은 3.6.0 이후의 native `tl.dot_scaled`다(MXFP8, MXFP4, NVFP4를 같은 포맷끼리만). FP8 GEMM은 `tl.dot`의 절반 속도 문제가 있으므로 unit-scale `dot_scaled` 우회를 검토할 만하다. 블록 크기와 `num_stages`는 99 KB SMEM에 맞춰 autotune 공간을 줄여야 한다. `warp_specialize=True`와 `num_ctas>1`은 쓰지 않는 편이 안전하다.

---

## 2. Triton Gluon (`triton.experimental.gluon`)

| 항목 | 값 | 근거 |
|---|---|---|
| F1 | **△** (experimental). Triton 3.8.0 동봉으로, 백엔드와 타깃(`sm_120a`)은 Triton과 같다. NVIDIA 모듈은 `nvidia.ampere`(mma_v2, async_copy, mbarrier), `nvidia.hopper`(tma, mbarrier, cluster, wgmma), `nvidia.blackwell`(tcgen05, TMEM, clc, tma), `nvidia.rubin`이다. sm_120에서 쓸 수 있는 부분은 **ampere + hopper.tma/mbarrier (+ blackwell.tma 일부)**다. 전용 "sm120" 모듈은 없다 | [gluon/language/nvidia/](https://github.com/triton-lang/triton/tree/66aa2f8a62912e821dd974ace5697fe19ac90968/python/triton/experimental/gluon/language/nvidia), blackwell/`__init__.py` L10-13 (gluon-nvidia-blackwell-init.py) |
| F2 | **O**. `ttgl.nvidia.ampere.mma_v2`(blackwell 모듈에서도 재노출) | ampere/`__init__.py` L109 (gluon-nvidia-ampere-init.py) |
| F3 | **(미확인)**. `mma_v2`가 FP8 operand를 받는지 문서·테스트 근거가 없다. 백엔드 MMAv2 lowering은 sm_120 FP8을 지원한다 | triton-MMAv2.cpp |
| F4 | **X**. Gluon의 NVIDIA scaled MMA API는 `tcgen05_mma_scaled`(TMEM 필요, sm_100 계열)뿐이다. `mma_v2` block-scale API는 없다 | blackwell/`__init__.py` L604 |
| F5 | **O** (scatter·multicast 제외). `hopper.tma` async copy/store. 튜토리얼 04-tma는 cc ≥ 9에서 동작한다 | [tutorials/gluon/04-tma.py](https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/python/tutorials/gluon/04-tma.py) L39 |
| F6 | **O**. mbarrier, async_copy, 수동 multi-buffering | tutorials 03/04 |
| F7 | **O** (API). `ttgl.warp_specialize(functions_and_args, worker_num_warps, worker_num_regs)`. 튜토리얼 08의 일부 테스트는 `is_hopper_or_newer`(cc ≥ 9)로 12.x도 포함하고, 나머지는 cc == 10 전용이다. sm_120 CI 실행 여부는 (미확인) | `_core.py` L738, tutorials/gluon/08 L56-61 |
| F8 | **△**. `worker_num_regs` → 백엔드 setmaxnreg. sm_120 검증 근거는 없다 | `_core.py` L738-757 |
| F9 | **X**. `supportClusterOps()`가 12.x를 제외한다. 이슈 #10974는 Gluon `num_ctas=2` + `async_store`가 sm120에서 실패하는 사례다 | triton-TargetFeatures.h, [#10974](https://github.com/triton-lang/triton/pull/10974) (2차) |
| F10 | **△**. `blackwell.clc` 모듈(3.8.0 RN: "Blackwell Gluon kernels can use CLC"). sm_120 사용 근거는 없다 | [3.8.0 RN](https://github.com/triton-lang/triton/releases/tag/v3.8.0) (triton-3.8.0-release-notes.md) |
| F11 | (미확인) | – |
| F12 | X | – |
| F13 | **O**. layout을 명시한다(`BlockedLayout`, `NVMMASharedLayout` swizzle, `DotOperandLayout`, `NVMMADistributedLayout`). 이것이 Gluon을 쓰는 핵심 이유다 | tutorials/gluon/02-layouts.py |
| F14 | **O**. `ttgl.inline_asm_elementwise` | `_core.py` L131 |
| F15 | **O** (Triton과 같다) | – |
| F16 | **(미확인)** (`@gluon.jit`와 `triton.autotune` 조합 근거 없음) | – |
| F17 | sm_120 전용 예제는 없다. 튜토리얼 03(async copy), 04(TMA), 07·08 일부만 cc ≥ 9 게이트다(05 wgmma = cc 9, 06/10/11 tcgen05 = cc 10) | tutorials/gluon/*.py skipif |
| F18 | API 변경이 잦다(3.8.0 breaking changes: TMEM load layout 자동 추론, `blackwell.float2` 제거 등). Hopper·Blackwell(sm_100) 중심 설계라 sm_120은 "Ampere 스타일 + TMA" 조합으로 직접 짜야 한다 | 3.8.0 RN "Breaking Changes" |

**sm_120에서 Gluon을 쓸 때 요점.** Gluon은 Triton 컴파일러의 layout·스케줄 결정을 사람이 가져오는 도구다. sm_120에서는 `mma_v2` + `hopper.tma` + mbarrier + `warp_specialize`로 Ampere/Ada식 파이프라인을 명시적으로 짜는 데 쓸 수 있다. 하지만 **block-scaled FP4/FP8 MMA API가 없어** sm_120의 FP4 Tensor Core를 Gluon에서 직접 쓸 수 없다(이 경우 `tl.dot_scaled`나 CUTLASS/CuTe DSL을 쓴다).

---

## 3. Meta TLX (Triton Low-level Extensions, fbtriton / triton-utlx)

| 항목 | 값 | 근거 |
|---|---|---|
| F1 | **X / (미확인)**. 문서의 hardware tag는 `sm90`(Hopper), `sm100`(Blackwell), `sm90+`와 AMD뿐이다. AutoWS도 "Hopper (sm90) and Blackwell (sm100)"이다. sm_120 언급은 전혀 없다. 배포는 `pip install fbtriton`(fork) 또는 `triton-utlx`(upstream Triton 플러그인, `TRITON_EXT_ENABLED`) | [fbtriton README](https://github.com/facebookexperimental/triton/blob/fd4d8ce1b7cc2597d39697032910356e5ff613fd/README.md) (fbtriton-README.md), [tlx.md](https://github.com/facebookexperimental/triton/blob/fd4d8ce1b7cc2597d39697032910356e5ff613fd/website/content/tlx.md) (fbtriton-tlx.md), [compiler.md](https://github.com/facebookexperimental/triton/blob/fd4d8ce1b7cc2597d39697032910356e5ff613fd/website/content/compiler.md) L56 (fbtriton-compiler.md) |
| F2~F4 | sm_120: (미확인). 커널·튜토리얼은 wgmma(Hopper)·tcgen05/TMEM(sm100)용이다 | website/content/kernels.md |
| F5~F7 | TMA, barrier, 명시적 WS: sm90/sm100 태그만 있다 | tlx.md |
| F8 | (미확인) | – |
| F9·F10 | cluster(`ctas_per_cga`, SM90+), CLC는 **[sm100]** 태그 | [clusters.md](https://github.com/facebookexperimental/triton/blob/fd4d8ce1b7cc2597d39697032910356e5ff613fd/website/content/clusters.md) L5, L59 (fbtriton-tlx-clusters.md) |
| F11~F14 | (미확인) / F12 X | – |
| F15·F16 | TorchTLX: Inductor 템플릿(Blackwell GEMM WS 템플릿). Triton autotune | torchtlx.md |
| F17 | sm_120 예제 없음 | kernels.md |
| F18 | sm_120용으로 쓰지 않는 것이 좋다. upstream Triton WS/Gluon 버그는 upstream에서 다룬다 | README |

**요점.** TLX는 공개 코드가 있지만 **sm_120 지원 근거가 없다**. 공개 커널은 Hopper wgmma와 B200 tcgen05·TMEM·2-CTA·CLC 기반이다. sm_120 커널 언어 후보에서 제외한다.

---

## 4. TileLang (`tile-ai/tilelang`)

| 항목 | 값 | 근거 |
|---|---|---|
| F1 | **O**. README 지원 표에 "code paths from SM70 through SM120"이 있다. `TargetIsSM120()` = 120 ≤ arch < 130(sm_121 포함). `sm_120f` 문자열 파싱 테스트가 있다. block-scaled 경로는 **`sm_120a` + CUDA 12.8 이상**을 요구한다(`CUTLASS_ARCH_MMA_SM120A_ENABLED` 가드). 현행 v0.1.14 | [README L121](https://github.com/tile-ai/tilelang/blob/85fd8fc2d31ff105c857cc2c1785ede155f5cbf7/README.md) (tilelang-README.md), [target_utils.cc L78-83](https://github.com/tile-ai/tilelang/blob/85fd8fc2d31ff105c857cc2c1785ede155f5cbf7/src/cuda/target_utils.cc) (tilelang-target_utils.cc), [mma_block_scale.h L34-60](https://github.com/tile-ai/tilelang/blob/85fd8fc2d31ff105c857cc2c1785ede155f5cbf7/src/tl_templates/cuda/instruction/mma_block_scale.h) (tilelang-mma_block_scale.h) |
| F2 | **O**. `T.gemm` → MMA(`mma_macro_generator`). WGMMA는 `TargetIsHopper`(90 ≤ arch < 100), TCGEN05는 `TargetIsSm100`(100~110)만이라 sm_120은 MMA 경로다 | [gemm.cc](https://github.com/tile-ai/tilelang/blob/85fd8fc2d31ff105c857cc2c1785ede155f5cbf7/src/cuda/op/gemm.cc) L95, L340-380 (tilelang-gemm.cc) |
| F3 | **O** (코드상). MMA generator에 `float8_e4m3`/`e5m2` 매핑이 있다. gemm_fp8 README: "we only support fp8 with mma instructions". sm_120 전용 FP8 테스트는 (미확인)이다. FP8+FP32 누산 절반 속도는 HW 특성이다 | mma_macro_generator.py L65-69, [gemm_fp8/README.md](https://github.com/tile-ai/tilelang/blob/85fd8fc2d31ff105c857cc2c1785ede155f5cbf7/examples/gemm_fp8/README.md) (tilelang-gemm_fp8-README.md) |
| F4 | **△**. **NVFP4만** native다: `T.mma_gemm_blockscaled` → `mma.sync.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X ... ue4m3`(`static_assert`로 kMxf4nvf4 + UE4M3만 허용). `TargetIsSM120`이 아니면 FATAL. PR #2364(SM120 NVF4, README 2026-07-30 공지)와 #2324(NVFP4 GEMM)로 들어왔다. **MXFP8 `kind::mxf8f6f4`는 PR #3099(open)**, **MXFP4 ue8m0 2X/4X는 PR #3081(open)** | mma_block_scale.h L12-93, gemm.cc L360-372, [#2364](https://github.com/tile-ai/tilelang/pull/2364), [#3099](https://github.com/tile-ai/tilelang/pull/3099), [#3081](https://github.com/tile-ai/tilelang/pull/3081) (2차) |
| F5 | **O**. `T.copy`가 조건을 만족하면 TMA bulk load/store로 자동 lowering된다(`TargetHasBulkCopy` = arch ≥ 90, 12.x 포함). im2col TMA는 Hopper 전용(`pipeline_planning`) | copy_analysis.cc L195-240, target_utils.cc L117-122 |
| F6 | **O**. `T.Pipelined(num_stages=…)`, mbarrier | README, examples |
| F7 | **O** (코드상, sm_120 HW 검증 (미확인)). 자동 producer-consumer WS 패스는 `TargetHasBulkCopy`(≥ 90)와 파이프라인 루프 안의 TMA copy만 요구한다. 수동 WS는 `WSSchedule`/`WSRole`(`T.annotate_ws_schedule`) | [producer_consumer_ws.cc L2545-2560](https://github.com/tile-ai/tilelang/blob/85fd8fc2d31ff105c857cc2c1785ede155f5cbf7/src/cuda/transform/producer_consumer_ws.cc) (tilelang-producer_consumer_ws.cc), tilelang/language/warp_specialize.py |
| F8 | **O**. `WSRole.max_nreg` → `setmaxnreg.inc/dec`(num_warps는 4의 배수). `T.no_set_max_nreg` 옵션이 있다 | warp_specialize.py L61·L191, tl_templates/cuda/intrin.h L171 |
| F9 | **O / △**. `T.ClusterKernel(cluster_dims=…)`, `T.cluster_sync`, SM-to-SM copy가 있다. 문서 요구사항은 "CC ≥ 9.0 (Hopper / Blackwell / RTX 5090)"이다. ⚠ 같은 문서의 TMA multicast(`cluster_mask`)는 CUTLASS의 "GeForce에는 multicast 없음"과 충돌한다(06 문서 §3.2) | [cluster_tma.md](https://github.com/tile-ai/tilelang/blob/85fd8fc2d31ff105c857cc2c1785ede155f5cbf7/docs/programming_guides/cluster_tma.md) L1-10 (tilelang-cluster_tma.md) |
| F10 | **△**. persistent/stream-K 예제가 있다(`examples/gemm_streamk`). CLC(`clc_try_cancel`)는 **CuTeDSL 백엔드(contrib)에만** 있고, 기본 CUDA 백엔드 근거는 없다 | tilelang/contrib/cutedsl/cluster.py L13-66 |
| F11 | **O**. `pdl_trigger`/`pdl_sync` → `griddepcontrol`(`lower_pdl.cc`) | src/cuda/transform/lower_pdl.cc |
| F12 | **△**. `T.gemm_sp`: WGMMA sparse는 Hopper 전용이고, 그 외는 `mma_sp_macro_generator`(`mma.sp`) 경로다. sm_120 동작은 (미확인) | [gemm_sp.cc L80-95](https://github.com/tile-ai/tilelang/blob/85fd8fc2d31ff105c857cc2c1785ede155f5cbf7/src/cuda/op/gemm_sp.cc) |
| F13 | **O**. `T.annotate_layout`, `make_mma_swizzle_layout`, ldmatrix(≥ 75)/stmatrix(≥ 90, m16n8 ≥ 100) | target_utils.cc L97-109 |
| F14 | **O**. `T.call_extern`, `T.import_source`, `T.ptx_*` intrinsic | tilelang/language/tir/op.py L230, language/common.py L180 |
| F15 | **O**. `@tilelang.jit` 커널이 torch Tensor를 받는다(DLPack) | README |
| F16 | **O**. `tilelang.autotune` | tilelang/autotuner/tuner.py L1318 |
| F17 | [examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py](https://github.com/tile-ai/tilelang/blob/85fd8fc2d31ff105c857cc2c1785ede155f5cbf7/examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py) (tilelang-sm120_nvfp4_blockscaled_gemm.py). `maint/gemm/gemm_sm120/`에 CUTLASS SM120 NVF4(128×128×256) 대비 벤치와 정확도 비교 스크립트가 있다. 공개 수치와 % of peak는 (미확인) | README L209, maint/gemm/gemm_sm120/*.py |
| F18 | ① sm_120에서 JIT 컴파일이 무한 대기하는 이슈 #2328(open, RTX PRO 6000, DeepSeek-V4 코드. 이슈 본문에는 "Unsupported target for gemm"과 SMEM overflow 같은 이전 sm_120 오류도 언급). ② FP4는 NVFP4 한 포맷만. ③ 0.1.x로 API가 자주 바뀐다 | [#2328](https://github.com/tile-ai/tilelang/issues/2328) (2차) |

**sm_120에서 TileLang을 쓸 때 요점.** 커뮤니티 DSL 중 **sm_120 전용 코드 경로가 가장 명시적이다**(`TargetIsSM120`, `TensorCoreIntrinEmitterSM120`, `gemm_sm120` 예제). TMA 자동 lowering, 자동·수동 WS + setmaxnreg, PDL, cluster까지 코드 게이트가 sm_120을 포함한다. 다만 FP4는 NVFP4(ue4m3, 4X)만 있고, MXFP8/MXFP4 block-scale은 PR 대기 중이다. sm_120 HW에서의 WS/cluster 검증 기록은 공개된 것이 없다.

---

## 5. ThunderKittens (HazyResearch)

| 항목 | 값 | 근거 |
|---|---|---|
| F1 | **△** (primitive 수준). `KITTENS_SM120` 매크로가 있고, tests Makefile은 `-gencode arch=compute_120a,code=sm_120a -DKITTENS_SM120`(**sm_120a**)이다. `MAX_SHARED_MEMORY = 99 * 1024`. 하지만 **`kernels/`에 SM120 커널이 하나도 없다**(ARCH=SM120 선택지만 존재). TK 2.0(2026-01-11)의 "full support for Blackwell"은 B200/B300(tcgen05)이다. SM120/121 커널 포트 PR #203은 **머지되지 않고 닫혔고**, SM121 WGMMA shim PR #204는 open | [kittens.cuh L18-20](https://github.com/HazyResearch/ThunderKittens/blob/cb21f34cdd99996e0aade8a4de45cb5b418fc7f8/include/kittens.cuh), [util.cuh L77-84](https://github.com/HazyResearch/ThunderKittens/blob/cb21f34cdd99996e0aade8a4de45cb5b418fc7f8/include/common/util.cuh) (thunderkittens-util.cuh), [tests/Makefile L45-46](https://github.com/HazyResearch/ThunderKittens/blob/cb21f34cdd99996e0aade8a4de45cb5b418fc7f8/tests/Makefile) (thunderkittens-tests-Makefile), [README](https://github.com/HazyResearch/ThunderKittens/blob/cb21f34cdd99996e0aade8a4de45cb5b418fc7f8/README.md) (thunderkittens-README.md), [#203](https://github.com/HazyResearch/ThunderKittens/pull/203), [#204](https://github.com/HazyResearch/ThunderKittens/pull/204) (2차) |
| F2 | **O**. `warp::` HMMA `mma.sync.m16n8k16` bf16/fp16. `warpgroup::`(wgmma)는 `KITTENS_SM90`에서만, tcgen05는 `KITTENS_SM10X`에서만 include된다 | [mma/warp.cuh](https://github.com/HazyResearch/ThunderKittens/blob/cb21f34cdd99996e0aade8a4de45cb5b418fc7f8/include/ops/group/mma/warp.cuh) (thunderkittens-mma-warp.cuh), [mma/mma.cuh L10-18](https://github.com/HazyResearch/ThunderKittens/blob/cb21f34cdd99996e0aade8a4de45cb5b418fc7f8/include/ops/group/mma/mma.cuh) (thunderkittens-mma.cuh) |
| F3 | **O** (e4m3). `mma.sync.m16n8k32.f32.e4m3.e4m3.f32`가 `SM90‖SM10X‖SM120` 가드 안에 있다. e5m2 warp MMA는 (미확인) | warp.cuh L140-190 |
| F4 | **X**. `fp4e2m1`·`fp8e8m0` 타입은 SM120에서도 정의되지만, block-scaled MMA는 `tcgen05.cuh`(SM10X)에만 있다 | base_types.cuh L64-120, ops/thread/mma/tcgen05.cuh |
| F5 | **O**. TMA tile load/store(테스트 `tma.cu`에 SM120 가드) | tests/thread/memory/tile/tma.cu |
| F6 | **O**. mbarrier/semaphore(`sync.cuh`에 SM120 가드) | include/ops/thread/util/sync.cuh |
| F7 | **△**. C++로 직접 producer/consumer를 짤 수 있다. `prototype` LCF/LCSF 템플릿은 H100/B200 커널 기준이다 | README, prototype/ |
| F8 | **O**. `warpgroup::increase_registers/decrease_registers` → `setmaxnreg.{inc,dec}`(`group.cuh`, `SM90‖SM10X‖SM120` 가드) | [group.cuh L50-58](https://github.com/HazyResearch/ThunderKittens/blob/cb21f34cdd99996e0aade8a4de45cb5b418fc7f8/include/ops/group/group.cuh) |
| F9 | **O** (primitive). `barrier.cluster.*`, DSMEM(같은 가드). multicast 성능은 GeForce 주의(06 문서) | group.cuh L94-103 |
| F10 | **O** (primitive). `clusterlaunchcontrol.try_cancel … .multicast::cluster::all`이 `SM10X‖SM120` 가드 안에 있다 | ops/thread/util/util.cuh L235-280 |
| F11 | **O**. `griddepcontrol.launch_dependents/wait` | ops/thread/util/util.cuh L294-321 |
| F12 | X | – |
| F13 | **O**. shared tile swizzle, ldmatrix 기반 register tile 로드 | README "bank conflict", types/shared/st.cuh |
| F14 | **O**. CUDA C++ 그대로이고 inline PTX를 쓸 수 있다 | README "embedded into CUDA" |
| F15 | **O**. `pyutils/torchutils.cuh`, `CONFIG=pytorch` 빌드 | kernels/common.mk |
| F16 | X (내장 autotuner 없음) | – |
| F17 | **X**. repo 안에 sm_120 커널이나 벤치가 없다(PR #203의 GB10 포트는 머지되지 않음) | kernels/ 디렉터리 |
| F18 | primitive에 sm_120 가드만 있고 커널·튜닝이 없다. 커널마다 nvcc로 따로 컴파일해야 한다(2.0부터 Python 패키지 아님) | README "Recent Updates" |

**sm_120에서 ThunderKittens를 쓸 때 요점.** 헤더 라이브러리 수준에서는 sm_120a 빌드와 TMA, mbarrier, setmaxnreg, cluster, CLC, PDL, warp-level HMMA/FP8 MMA가 가드되어 있다. 그러나 **공식 커널, 벤치, NVFP4/MXFP block-scaled MMA가 없다**. B200용 TK 2.0 커널(tcgen05)은 sm_120에서 컴파일되지 않는다. sm_120용으로는 직접 커널을 짜야 하는 C++ 템플릿 도구로 보는 것이 맞다.

---

## 6. Helion (`pytorch/helion`)

| 항목 | 값 | 근거 |
|---|---|---|
| F1 | **O** (Triton을 상속한다). 기본 백엔드는 Triton(PyTorch 2.14 → Triton 3.8.0)이다. 추가 백엔드는 두 가지다. **Triton-TileIR 백엔드**는 "compute capability 10.x/12.x"(Blackwell)용이고, CuTe DSL 백엔드는 CUDA 13+ 실험 단계다. AOT heuristic 조회는 "on `sm120`, Helion tries `sm120`, `sm100`, `sm90`, … in order" | [README L288-310](https://github.com/pytorch/helion/blob/cfb135ef8da2d87112e4bc9221f10b87c8cc9cad/README.md) (helion-README.md), [_compat.py L296-310](https://github.com/pytorch/helion/blob/cfb135ef8da2d87112e4bc9221f10b87c8cc9cad/helion/_compat.py) (helion-_compat.py), [deployment_autotuning.md L815-822](https://github.com/pytorch/helion/blob/cfb135ef8da2d87112e4bc9221f10b87c8cc9cad/docs/deployment_autotuning.md) (helion-deployment_autotuning.md), [tileir_backend.md](https://github.com/pytorch/helion/blob/cfb135ef8da2d87112e4bc9221f10b87c8cc9cad/docs/tileir_backend.md) (helion-tileir_backend.md) |
| F2·F3 | **O** (Triton `tl.dot` → MMAv2. FP8 주의사항은 Triton과 같다) | – |
| F4 | **△**. `hl.dot_scaled`가 있다 → Triton `tl.dot_scaled`. sm_120 제약(동일 포맷, K-packed)을 그대로 상속한다 | helion/language/matmul_ops.py L837 |
| F5 | **O**. `indexing="tensor_descriptor"`는 cc major ≥ 9에서 켜진다(12.x 포함) | _compat.py L207-230 |
| F6 | **O**. `num_stages`가 autotune 파라미터다 | runtime/config.py |
| F7 | **△**. `range_warp_specializes` 설정 → Triton WS(sm_120 미검증·#10284) | runtime/config.py L43·L85 |
| F8·F9 | X (노출 안 됨 / Triton이 cluster 미지원) | – |
| F10 | △ (persistent `pid_type` 설정. CLC (미확인)) | – |
| F11 | (미확인) | – |
| F12 | X | – |
| F13 | X (컴파일러가 관리) | – |
| F14 | **O**. `hl.inline_asm_elementwise` | helion/language/inline_asm_ops.py |
| F15 | **O**. PyTorch 네이티브(커널 안에서 torch 연산을 쓴다) | README |
| F16 | **O** (핵심 기능, 대규모 autotune + AOT heuristic) | deployment_autotuning.md |
| F17 | sm_120 전용 예제·수치는 (미확인). sm_100 전용 tcgen05 우회 코드(`capability[0] == 10`)는 sm_120에 적용되지 않는다 | _compat.py L630-640 |
| F18 | Triton의 sm_120 한계(99 KB SMEM, WS, cluster)를 그대로 상속한다 | – |

**요점.** Helion은 "PyTorch 문법 → Triton(또는 TileIR/CuTe)" 생성기라 sm_120 기능 범위는 **Triton 3.8.0 범위와 같다**. autotune이 99 KB SMEM 초과 config를 자동으로 걸러주는 장점이 있다. TileIR 백엔드가 12.x를 명시 지원한다는 점이 차별점이지만, TileIR 자체는 NVIDIA 스택(cuTile 계열)이라 다른 조사 범위다.

---

## 7. torch.compile / Inductor (+ torchao)

| 항목 | 값 | 근거 |
|---|---|---|
| F1 | **O**. Triton codegen은 PyTorch 2.7(cu128)부터 Blackwell을 지원하고, 2.14.0은 Triton 3.8.0이다. sm12x 전용 FlexAttention 기본 config가 있다(`sm_120_default_flex_config`, `capability_class = "sm12x"`) | [heuristics/template/triton.py L1455-1590](https://github.com/pytorch/pytorch/blob/31527a43dbf6adf4df2e6eb5c7b38094fec6b6f6/torch/_inductor/heuristics/template/triton.py) (pytorch-inductor-heuristics-template-triton.py), [PyTorch 2.14 RN](https://github.com/pytorch/pytorch/releases/tag/v2.14.0) (pytorch-2.14.0-release-notes.md) |
| F2 | **O** (Triton mm 템플릿 + ATen cuBLAS를 max-autotune에서 비교) | – |
| F3 | **O/△**. `torch._scaled_mm`(ATen/cuBLASLt)과 Inductor Triton scaled-mm 템플릿 | – |
| F4 | **△**. torchao MX/NVFP4는 `is_sm_at_least_100()`(= capability ≥ (10,0)이라 **12.x도 통과**)으로 게이트된다. 실제 GEMM은 `_scaled_mm` → cuBLASLt. sm_120에서의 동작·성능은 (미확인). 2.14 NVGEMM(CuTeDSL, NVFP4 포함)의 sm_120 적용 여부도 (미확인) | [torchao/utils.py L1158-1163](https://github.com/pytorch/ao/blob/3005bc1d2407646103ff759f38beb5c59a453728/torchao/utils.py) (torchao-utils.py), mx_formats/inference_workflow.py L305 |
| F5 | **△**. `has_triton_tma_device()` = capability ≥ (9,0)이라 TMA/persistent 템플릿은 쓸 수 있다. **Blackwell 전용 Triton 템플릿과 CuTeDSL 템플릿은 `is_datacenter_blackwell_arch()`(100 ≤ arch < 110)라 sm_120은 제외**다 | [utils/_triton.py L142-160](https://github.com/pytorch/pytorch/blob/31527a43dbf6adf4df2e6eb5c7b38094fec6b6f6/torch/utils/_triton.py) (pytorch-utils-_triton.py), [cuda_env.py L31-36](https://github.com/pytorch/pytorch/blob/31527a43dbf6adf4df2e6eb5c7b38094fec6b6f6/torch/_inductor/codegen/cuda/cuda_env.py) (pytorch-inductor-cuda_env.py), [_inductor/utils.py L2736-2890](https://github.com/pytorch/pytorch/blob/31527a43dbf6adf4df2e6eb5c7b38094fec6b6f6/torch/_inductor/utils.py) |
| F6 | **O** (autotune config의 `num_stages`) | – |
| F7 | **X** (sm_120). WS를 쓰는 Blackwell 템플릿이 datacenter Blackwell 전용이다 | 위와 같음 |
| F8~F12 | X (Inductor가 사용자에게 노출하지 않음) | – |
| F13·F14 | X (사용자 제어 없음. custom Triton op로 우회) | – |
| F15 | **O** (네이티브) | – |
| F16 | **O** (`mode="max-autotune"`) | – |
| F17 | 공개 sm_120 수치는 (미확인). 2.14 RN: "Disable cuDNN convolution engines 58 and 63 on `sm120`"(#190112), "Update the cuDNN errata filter for `sm120`"(#191701) | PyTorch 2.14 RN L1236-1238 |
| F18 | Inductor의 "Blackwell" 최적화(TMA WS 템플릿, CuTeDSL/NVGEMM 일부)는 sm_100 전용이다. sm_120은 일반 Triton 템플릿으로 떨어진다 | cuda_env.py |

**요점.** torch.compile은 sm_120에서 **일반 Triton 코드 생성기**로 동작한다(sm12x FlexAttention config 정도만 전용). Blackwell 고성능 GEMM 템플릿(TMA + WS, CuTeDSL)은 `is_datacenter_blackwell_arch()` 게이트로 빠진다. 저정밀도는 torchao → `_scaled_mm`(cuBLASLt) 경로에 의존한다.

---

## 8. JAX Pallas (Mosaic GPU / Triton 백엔드)

| 항목 | 값 | 근거 |
|---|---|---|
| F1 | **△**. Mosaic GPU는 드라이버 CC로 `sm_XY`를 만든 뒤 LLVM에 `a` 변형이 있으면 그것을 고른다(→ `sm_120a`). 문서와 예제는 H100(wgmma)/B200(tcgen05) 대상이다. `tcgen05` 모듈은 `assert arch.major in {10, 11}`. sm_120 제외 테스트 skip PR(#38037, 2026-06 머지)이 있다 | [target.cc L41-78](https://github.com/jax-ml/jax/blob/5570f3d6fef9e44f44f3b06096674298a7e89ed3/jaxlib/mosaic/gpu/target.cc) (jax-mosaic-gpu-target.cc), [tcgen05.py L119-121](https://github.com/jax-ml/jax/blob/5570f3d6fef9e44f44f3b06096674298a7e89ed3/jax/experimental/mosaic/gpu/tcgen05.py) (jax-mosaic-gpu-tcgen05.py), [quickstart](https://github.com/jax-ml/jax/blob/5570f3d6fef9e44f44f3b06096674298a7e89ed3/docs/pallas/gpu/quickstart.md) (jax-pallas-gpu-quickstart.md), [#38037](https://github.com/jax-ml/jax/pull/38037) (2차) |
| F2 | **△**. `plgpu.mma` → `mma.sync.aligned.m16n8k{k}`(`mosaic/gpu/mma.py`에 arch 게이트 없음). sm_120 검증은 (미확인) | [mma.py L142](https://github.com/jax-ml/jax/blob/5570f3d6fef9e44f44f3b06096674298a7e89ed3/jax/experimental/mosaic/gpu/mma.py) (jax-mosaic-gpu-mma.py), pallas/mosaic_gpu/primitives.py L2272 |
| F3 | **△** (`SUPPORTED_F8_TYPES = (E4M3FN, E5M2)`, (미확인)) | mma.py |
| F4 | **X**. block-scale은 tcgen05 경로에만 있다. `mma.py`에 block_scale이 없다 | mma.py, tcgen05.py |
| F5 | **O** (코드상). TMA 경로는 `get_arch().major >= 9` 게이트 | pallas/mosaic_gpu/primitives.py L996·L1345 |
| F6 | **O**. barrier, `emit_pipeline` | docs/pallas/gpu/pipelining.md |
| F7 | **△** (warpgroup specialization은 "Hopper+" 문서. sm_120 (미확인)) | pipelining.md L179 |
| F8~F11 | (미확인). CLC `try_cancel` helper가 있으나(pallas/mosaic_gpu/helpers.py L389-430) sm_120 사용 근거는 없다 | – |
| F12 | X | – |
| F13 | **O** (Tiling/Swizzle transform) | pipelining.md L95, jax-pallas-gpu-reference.md |
| F14 | △ (Mosaic GPU 저수준 API. (미확인)) | – |
| F15 | X (JAX 배열. DLPack으로 상호 운용) | – |
| F16 | X (내장 autotuner 없음) | – |
| F17 | 없음 | – |
| F18 | Pallas **Triton 백엔드**는 XLA 동봉 Triton을 쓰고, sm_120 상태는 (미확인)이다(`float8_e4m3fn` cast는 cc ≥ 89) | pallas/triton/lowering.py L1663-1664 |

**요점.** JAX Pallas(Mosaic GPU)는 H100 wgmma와 B200 tcgen05를 1급 타깃으로 설계되어 있다. sm_120에서는 `mma.sync` + TMA 같은 일반 경로만 이론상 쓸 수 있고, FP4 block-scale이나 공식 예제는 없다. sm_120 커널 언어로 권하기 어렵다.

---

## 9. Mojo / MAX (Modular)

| 항목 | 값 | 근거 |
|---|---|---|
| F1 | **△**. Mojo stdlib가 `sm_120`/`sm_120a`, `sm_121`/`sm_121a`를 인식한다(`_is_sm_120x`). MAX 패키지 페이지는 "RTX 50XX series"를 "Known compatible for development"로 두고, 서빙 검증은 B200뿐이다. 일부 SIMD 경로는 `not _is_sm_120x_or_newer()`로 sm_120을 제외한다 | [info.mojo L867-952](https://github.com/modular/modular/blob/da4f8b76ac0f51091f927e4a9b6f0b17f40b11e6/Mojo/stdlib/std/sys/info.mojo) (mojo-std-sys-info.mojo), [max.modular.com/packages](https://max.modular.com/packages) (2026-09-11 확인, 링크만), simd.mojo L3829-3893 |
| F2·F3 | (미확인). `mma_nvidia.mojo`(`mma.sync`)는 범용이다 | max/mojo/max/gpu/compute/arch/mma_nvidia.mojo, max-gpu-compute-mma.mojo |
| F4 | **△**. NVFP4 양자화 커널은 "only supported on SM100 or SM120"이다. **block-scaled matmul은 sm_120에 Mojo 커널이 없어 cuBLASLt(vendor)로 fallback**한다("consumer Blackwell (sm_120 / sm_121) has no SM100 kernel"). Mojo의 block-scale MMA는 `mma_nvidia_sm100.mojo`(tcgen05)에만 있다 | [block_scaled_quantization.mojo L2388-2390](https://github.com/modular/modular/blob/da4f8b76ac0f51091f927e4a9b6f0b17f40b11e6/max/kernels/src/linalg/block_scaled_quantization.mojo) (max-block_scaled_quantization.mojo), matmul/vendor/blas.mojo L1146-1149 |
| F5~F18 | (미확인). sm_120 전용 커널·예제 근거가 없다 | – |

**요점.** Mojo는 sm_120 타깃을 인식하지만, MAX의 Blackwell 고성능 커널은 B200(tcgen05) 전용이다. sm_120의 FP4 GEMM은 cuBLASLt로 넘긴다. "RTX 50 개발 호환" 수준이다.

---

## 10. 기타 (커널 작성 도구가 아니라 라이브러리라서 짧게만)

- **QuACK (Dao-AILab)**, **FlashInfer**: #11320 본문에 따르면 SM120에서 FP8을 `mxf8f6f4` + unit scale(E8M0 127)로 돌린다(2차, 본문도 "AI output, unchecked"라고 적음). QuACK은 CuTe DSL 기반이라 NVIDIA 스택 조사 범위다.
- **triton_kernels**(Triton 저장소 안의 MoE 커널 라이브러리): sm120/121을 sm80 fallback으로 지원하는 PR #8484는 머지되지 않았다. gpt-oss가 5090에서 SMEM OOR로 실패한 이슈 #8182가 있다(2차).
- **Triton-to-TileIR 백엔드**: Helion README가 "compute capability 10.x/12.x"를 명시한다. NVIDIA TileIR 계열이라 공식 스택 조사와 합칠 대상이다.

---

## 요약 매트릭스 (F1~F12)

| 도구 | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Triton 3.8 | O | O | O¹ | △² | O³ | O | △ | △ | X | △ | O | X |
| Gluon | △ | O | (미확인) | X | O³ | O | O | △ | X | △ | (미확인) | X |
| TLX | X | (미확인) | (미확인) | (미확인) | (미확인) | (미확인) | (미확인) | (미확인) | (미확인) | X(sm100) | (미확인) | X |
| TileLang 0.1.14 | O | O | O | △⁴ | O | O | O | O | O/△⁵ | △ | O | △ |
| ThunderKittens | △ | O | O | X | O | O | △ | O | O | O | O | X |
| Helion 1.4 | O | O | O¹ | △² | O | O | △ | X | X | △ | (미확인) | X |
| Inductor 2.14 | O | O | O/△ | △ | △ | O | X | X | X | △ | X | X |
| JAX Pallas MGPU | △ | △ | △ | X | O | O | △ | (미확인) | (미확인) | (미확인) | (미확인) | X |
| Mojo/MAX | △ | (미확인) | (미확인) | △(cuBLASLt) | (미확인) | (미확인) | (미확인) | (미확인) | (미확인) | (미확인) | (미확인) | (미확인) |

¹ FP8 `mma.sync` + FP32 누산은 RTX 50에서 절반 속도다(#11320). ² MXFP8, MXFP4, NVFP4는 같은 포맷끼리만 native이고 mixed는 decompose. ³ scatter는 X. ⁴ NVFP4만. ⁵ multicast는 GeForce에서 권장되지 않음.
