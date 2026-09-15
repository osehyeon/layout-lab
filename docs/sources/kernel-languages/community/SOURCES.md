# SOURCES — kernel-languages/community

모든 파일은 2026-09-11에 가져왔다. GitHub 소스는 해당 커밋의 shallow clone에서 복사했다. 릴리스 노트는 GitHub Releases API의 `body`(Markdown)를 저장했다. 모든 파일을 `file`로 텍스트인지 확인했다.

커밋: triton `66aa2f8a62912e821dd974ace5697fe19ac90968`, fbtriton `fd4d8ce1b7cc2597d39697032910356e5ff613fd`, tilelang `85fd8fc2d31ff105c857cc2c1785ede155f5cbf7`, ThunderKittens `cb21f34cdd99996e0aade8a4de45cb5b418fc7f8`, helion `cfb135ef8da2d87112e4bc9221f10b87c8cc9cad`, pytorch `31527a43dbf6adf4df2e6eb5c7b38094fec6b6f6`, torchao `3005bc1d2407646103ff759f38beb5c59a453728`, jax `5570f3d6fef9e44f44f3b06096674298a7e89ed3`, modular `da4f8b76ac0f51091f927e4a9b6f0b17f40b11e6`.

| 파일 | 원본 URL | 가져온 날짜 | 설명 |
|---|---|---|---|
| findings.md | — (이 폴더의 조사 메모) | 2026-09-11 | 도구별 F1~F18 근거 표. 07-sm120-kernel-languages.md의 커뮤니티 도구 셀 근거 |
| triton-TargetFeatures.h | https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/include/triton/Dialect/TritonNvidiaGPU/IR/TargetFeatures.h | 2026-09-11 | `supportClusterOps()`의 12.x 제외, ldmatrix/stmatrix 게이트 |
| triton-AccelerateMatmul.cpp | https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp | 2026-09-11 | sm120 MMA 버전 선택(v2), `ScaledBlockedToMMA`(sm120 dot_scaled), fp8 MMAv2 |
| triton-MMAv2.cpp | https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/MMAv2.cpp | 2026-09-11 | FP8 `mma.sync` 및 `kind::mxf8f6f4`/`mxf4nvf4 .block_scale` PTX 문자열 |
| triton-nvidia-backend-compiler.py | https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/third_party/nvidia/backend/compiler.py | 2026-09-11 | `sm_120a` 타깃, 패스 파이프라인(WS/TMA/CLC 분기) |
| triton-_internal_testing.py | https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/python/triton/_internal_testing.py | 2026-09-11 | `is_sm12x`, `is_blackwell`(10,11), `supports_clc` |
| triton-test_tensor_descriptor.py | https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/python/test/unit/language/test_tensor_descriptor.py | 2026-09-11 | sm12x TMA scatter skip, SMEM 부족 주석 |
| triton-test_warp_specialization.py | https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/python/test/unit/language/test_warp_specialization.py | 2026-09-11 | WS 테스트 게이트(Hopper/Blackwell 10·11만) |
| triton-tutorial-09-persistent-matmul.py | https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/python/tutorials/09-persistent-matmul.py | 2026-09-11 | TMA/WS/CLC 게이트(cc ≥ 9 / ≥ 10) |
| triton-tutorial-10-block-scaled-matmul.py | https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/python/tutorials/10-block-scaled-matmul.py | 2026-09-11 | block-scaled 튜토리얼(cc 10/11 전용) |
| gluon-tutorial-01-intro.py | https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/python/tutorials/gluon/01-intro.py | 2026-09-11 | Gluon 정의("same compiler stack as Triton … lower-level") |
| gluon-tutorial-02-layouts.py | https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/python/tutorials/gluon/02-layouts.py | 2026-09-11 | Gluon layout(`BlockedLayout` 등) |
| gluon-tutorial-08-warp-specialization.py | https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/python/tutorials/gluon/08-warp-specialization.py | 2026-09-11 | Gluon warp specialization("Hopper and newer", tcgen05 예제) |
| gluon-nvidia-ampere-init.py | https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/python/triton/experimental/gluon/language/nvidia/ampere/__init__.py | 2026-09-11 | Gluon `mma_v2` |
| gluon-nvidia-blackwell-init.py | https://github.com/triton-lang/triton/blob/66aa2f8a62912e821dd974ace5697fe19ac90968/python/triton/experimental/gluon/language/nvidia/blackwell/__init__.py | 2026-09-11 | Gluon tcgen05 / `tcgen05_mma_scaled` / clc |
| triton-3.5.0-release-notes.md | https://github.com/triton-lang/triton/releases/tag/v3.5.0 | 2026-09-11 | "#7409 Don't promote fp8 MMAv2 dot inputs for sm120" |
| triton-3.6.0-release-notes.md | https://github.com/triton-lang/triton/releases/tag/v3.6.0 | 2026-09-11 | "SM120 Features": native FP4/MXFP8 scaled dot, TMA gather4 |
| triton-3.7.0-release-notes.md | https://github.com/triton-lang/triton/releases/tag/v3.7.0 | 2026-09-11 | 3.7.0 릴리스 노트 |
| triton-3.8.0-release-notes.md | https://github.com/triton-lang/triton/releases/tag/v3.8.0 | 2026-09-11 | 현행 3.8.0 (Gluon CLC, Rubin 등) |
| pytorch-2.14.0-release-notes.md | https://github.com/pytorch/pytorch/releases/tag/v2.14.0 | 2026-09-11 | NVGEMM, sm120 cuDNN 엔진 비활성화 등 |
| fbtriton-README.md | https://github.com/facebookexperimental/triton/blob/fd4d8ce1b7cc2597d39697032910356e5ff613fd/README.md | 2026-09-11 | fbtriton / TLX / uTLX 개요 |
| fbtriton-tlx.md | https://github.com/facebookexperimental/triton/blob/fd4d8ce1b7cc2597d39697032910356e5ff613fd/website/content/tlx.md | 2026-09-11 | TLX hardware tag(sm90/sm100) |
| fbtriton-tlx-clusters.md | https://github.com/facebookexperimental/triton/blob/fd4d8ce1b7cc2597d39697032910356e5ff613fd/website/content/clusters.md | 2026-09-11 | TLX cluster / CLC [sm100] |
| fbtriton-compiler.md | https://github.com/facebookexperimental/triton/blob/fd4d8ce1b7cc2597d39697032910356e5ff613fd/website/content/compiler.md | 2026-09-11 | AutoWS 대상(sm90/sm100) |
| tilelang-README.md | https://github.com/tile-ai/tilelang/blob/85fd8fc2d31ff105c857cc2c1785ede155f5cbf7/README.md | 2026-09-11 | "SM70 through SM120", SM120 NVF4 공지 |
| tilelang-target_utils.cc | https://github.com/tile-ai/tilelang/blob/85fd8fc2d31ff105c857cc2c1785ede155f5cbf7/src/cuda/target_utils.cc | 2026-09-11 | `TargetIsSM120`, `TargetHasBulkCopy` 등 게이트 |
| tilelang-gemm.cc | https://github.com/tile-ai/tilelang/blob/85fd8fc2d31ff105c857cc2c1785ede155f5cbf7/src/cuda/op/gemm.cc | 2026-09-11 | GEMM lowering 선택(WGMMA/TCGEN05/MMA/MMABlockScaled) |
| tilelang-mma_block_scale.h | https://github.com/tile-ai/tilelang/blob/85fd8fc2d31ff105c857cc2c1785ede155f5cbf7/src/tl_templates/cuda/instruction/mma_block_scale.h | 2026-09-11 | SM120a NVF4 `mxf4nvf4 4X ue4m3` |
| tilelang-producer_consumer_ws.cc | https://github.com/tile-ai/tilelang/blob/85fd8fc2d31ff105c857cc2c1785ede155f5cbf7/src/cuda/transform/producer_consumer_ws.cc | 2026-09-11 | 자동 WS 패스 게이트 |
| tilelang-cluster_tma.md | https://github.com/tile-ai/tilelang/blob/85fd8fc2d31ff105c857cc2c1785ede155f5cbf7/docs/programming_guides/cluster_tma.md | 2026-09-11 | cluster/TMA multicast (CC ≥ 9.0, RTX 5090 표기) |
| tilelang-sm120_nvfp4_blockscaled_gemm.py | https://github.com/tile-ai/tilelang/blob/85fd8fc2d31ff105c857cc2c1785ede155f5cbf7/examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py | 2026-09-11 | SM120 NVFP4 GEMM 예제 |
| tilelang-gemm_fp8-README.md | https://github.com/tile-ai/tilelang/blob/85fd8fc2d31ff105c857cc2c1785ede155f5cbf7/examples/gemm_fp8/README.md | 2026-09-11 | FP8은 mma 경로로 지원한다는 노트 |
| thunderkittens-README.md | https://github.com/HazyResearch/ThunderKittens/blob/cb21f34cdd99996e0aade8a4de45cb5b418fc7f8/README.md | 2026-09-11 | TK 2.0 공지, ARCH=SM120 옵션 |
| thunderkittens-util.cuh | https://github.com/HazyResearch/ThunderKittens/blob/cb21f34cdd99996e0aade8a4de45cb5b418fc7f8/include/common/util.cuh | 2026-09-11 | SM120 `MAX_SHARED_MEMORY = 99 KB` |
| thunderkittens-mma-warp.cuh | https://github.com/HazyResearch/ThunderKittens/blob/cb21f34cdd99996e0aade8a4de45cb5b418fc7f8/include/ops/group/mma/warp.cuh | 2026-09-11 | warp HMMA / FP8 e4m3 `mma.sync` (SM120 가드) |
| thunderkittens-mma.cuh | https://github.com/HazyResearch/ThunderKittens/blob/cb21f34cdd99996e0aade8a4de45cb5b418fc7f8/include/ops/group/mma/mma.cuh | 2026-09-11 | warpgroup(SM90) / tcgen05(SM10X) include 분기 |
| thunderkittens-tests-Makefile | https://github.com/HazyResearch/ThunderKittens/blob/cb21f34cdd99996e0aade8a4de45cb5b418fc7f8/tests/Makefile | 2026-09-11 | `compute_120a` 빌드 플래그 |
| helion-README.md | https://github.com/pytorch/helion/blob/cfb135ef8da2d87112e4bc9221f10b87c8cc9cad/README.md | 2026-09-11 | TileIR(10.x/12.x), CuTe 백엔드 |
| helion-_compat.py | https://github.com/pytorch/helion/blob/cfb135ef8da2d87112e4bc9221f10b87c8cc9cad/helion/_compat.py | 2026-09-11 | tensor descriptor 게이트(cc ≥ 9), TileIR 게이트(10·12) |
| helion-tileir_backend.md | https://github.com/pytorch/helion/blob/cfb135ef8da2d87112e4bc9221f10b87c8cc9cad/docs/tileir_backend.md | 2026-09-11 | TileIR 백엔드 하드웨어 요구사항 |
| helion-deployment_autotuning.md | https://github.com/pytorch/helion/blob/cfb135ef8da2d87112e4bc9221f10b87c8cc9cad/docs/deployment_autotuning.md | 2026-09-11 | sm120 → sm100 → sm90 heuristic fallback |
| pytorch-inductor-cuda_env.py | https://github.com/pytorch/pytorch/blob/31527a43dbf6adf4df2e6eb5c7b38094fec6b6f6/torch/_inductor/codegen/cuda/cuda_env.py | 2026-09-11 | `is_datacenter_blackwell_arch()` (100~109) |
| pytorch-utils-_triton.py | https://github.com/pytorch/pytorch/blob/31527a43dbf6adf4df2e6eb5c7b38094fec6b6f6/torch/utils/_triton.py | 2026-09-11 | `has_triton_tma_device()` (cc ≥ 9.0) |
| pytorch-inductor-heuristics-template-triton.py | https://github.com/pytorch/pytorch/blob/31527a43dbf6adf4df2e6eb5c7b38094fec6b6f6/torch/_inductor/heuristics/template/triton.py | 2026-09-11 | `sm_120_default_flex_config`, `sm12x` 클래스 |
| torchao-mx_formats-README.md | https://github.com/pytorch/ao/blob/3005bc1d2407646103ff759f38beb5c59a453728/torchao/prototype/mx_formats/README.md | 2026-09-11 | MX/NVFP4 워크플로 문서 |
| torchao-utils.py | https://github.com/pytorch/ao/blob/3005bc1d2407646103ff759f38beb5c59a453728/torchao/utils.py | 2026-09-11 | `is_sm_at_least_100()` (≥ (10,0), 12.x 포함) |
| jax-mosaic-gpu-target.cc | https://github.com/jax-ml/jax/blob/5570f3d6fef9e44f44f3b06096674298a7e89ed3/jaxlib/mosaic/gpu/target.cc | 2026-09-11 | `sm_XYa` 자동 선택 |
| jax-mosaic-gpu-tcgen05.py | https://github.com/jax-ml/jax/blob/5570f3d6fef9e44f44f3b06096674298a7e89ed3/jax/experimental/mosaic/gpu/tcgen05.py | 2026-09-11 | `assert arch.major in {10, 11}` |
| jax-mosaic-gpu-mma.py | https://github.com/jax-ml/jax/blob/5570f3d6fef9e44f44f3b06096674298a7e89ed3/jax/experimental/mosaic/gpu/mma.py | 2026-09-11 | `mma.sync` 경로(FP8 타입 포함) |
| jax-pallas-gpu-reference.md | https://github.com/jax-ml/jax/blob/5570f3d6fef9e44f44f3b06096674298a7e89ed3/docs/pallas/gpu/reference.md | 2026-09-11 | Pallas:MGPU 레퍼런스(Hopper wgmma / Blackwell tcgen05) |
| jax-pallas-gpu-quickstart.md | https://github.com/jax-ml/jax/blob/5570f3d6fef9e44f44f3b06096674298a7e89ed3/docs/pallas/gpu/quickstart.md | 2026-09-11 | "examples target Hopper (H100)" |
| mojo-std-sys-info.mojo | https://github.com/modular/modular/blob/da4f8b76ac0f51091f927e4a9b6f0b17f40b11e6/Mojo/stdlib/std/sys/info.mojo | 2026-09-11 | `_SM_120X_ARCHS = ["sm_120", "sm_120a"]` 등 |
| max-block_scaled_quantization.mojo | https://github.com/modular/modular/blob/da4f8b76ac0f51091f927e4a9b6f0b17f40b11e6/max/kernels/src/linalg/block_scaled_quantization.mojo | 2026-09-11 | sm_120: block-scaled matmul이 cuBLASLt로 fallback |
| max-gpu-compute-mma.mojo | https://github.com/modular/modular/blob/da4f8b76ac0f51091f927e4a9b6f0b17f40b11e6/max/mojo/max/gpu/compute/mma.mojo | 2026-09-11 | Mojo GPU MMA 공통 인터페이스 |
| (링크만) MAX 패키지 / GPU 호환성 | https://max.modular.com/packages | 2026-09-11 | "RTX 50XX series" = Known compatible for development, B200 = tested for serving (WebFetch 요약) |
| (링크만) PyTorch release/2.14 Triton pin | https://github.com/pytorch/pytorch/blob/release/2.14/.ci/docker/triton_version.txt | 2026-09-11 | `3.8.0` |
| (링크만, 2차) Triton issues/PRs | https://github.com/triton-lang/triton/issues/11320 , /issues/11344 , /pull/11363 , /issues/10973 , /pull/10974 , /issues/10963 , /issues/10284 , /issues/8182 , /issues/7188 , /issues/7550 , /pull/7918 , /pull/8494 , /pull/8498 , /pull/10010 , /pull/10726 , /pull/11386 , /pull/8484 , /issues/11494 | 2026-09-11 | sm_120 관련 버그·기능 PR (GitHub API로 확인) |
| (링크만, 2차) TileLang | https://github.com/tile-ai/tilelang/issues/2328 , /pull/2364 , /pull/2324 , /pull/3099 , /pull/3081 | 2026-09-11 | sm_120 JIT hang, NVFP4, MXFP8/MXFP4 PR |
| (링크만, 2차) ThunderKittens | https://github.com/HazyResearch/ThunderKittens/pull/203 , /pull/204 | 2026-09-11 | SM120/121 포트 PR (머지되지 않음 / open) |
| (링크만, 2차) JAX | https://github.com/jax-ml/jax/pull/38037 | 2026-09-11 | sm_120 테스트 skip |

재사용한 기존 자료(다시 받지 않음): `docs/sources/architecture/pytorch-2.7-release-blog.html`(Triton 3.3 Blackwell 지원), `docs/sources/architecture/ptx-isa.txt`, `docs/sources/architecture/cutlass-blackwell_functionality.md`(GeForce multicast 없음).

실패·미수집: TLX 공식 사이트(facebookexperimental.github.io/triton)는 repo 안의 `website/content/*.md`로 대신했다. PyTorch main의 `.ci/docker/triton_version.txt`는 sparse-checkout 문제로 한 번 실패했다가 다시 받았다(pin `3.8.0`, 커밋 `c01b677…`).
