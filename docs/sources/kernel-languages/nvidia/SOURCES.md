# SOURCES — kernel-languages/nvidia (NVIDIA 공식 커널 작성 스택)

모든 파일 수집일: **2026-09-11** (`curl -L`). `.html`은 옆에 텍스트 추출본 `.txt`가 있다. URL은 2026-09-11에 HTTP 200으로 열리는지 확인했다(GitHub 파일은 `main` 브랜치 기준이라 이후 내용이 바뀔 수 있음).

## CUDA C++ / CCCL (`cuda::ptx`)

| 파일 | 원본 URL | 설명 |
|---|---|---|
| `cccl-ptx-api.html` (+`.txt`) | https://NVIDIA.github.io/cccl/libcudacxx/ptx_api.html | `cuda::ptx` 네임스페이스 개요 |
| `cccl-ptx-mma-less-list.html` (+`.txt`) | https://NVIDIA.github.io/cccl/libcudacxx/ptx/instructions.html | `cuda::ptx`가 감싸는 PTX 명령 목록. `mma`/`wgmma`/`tcgen05.mma`는 wrapper가 없음 |
| `cccl-ptx-setmaxnreg.h` | https://raw.githubusercontent.com/NVIDIA/cccl/main/libcudacxx/include/cuda/__ptx/instructions/generated/setmaxnreg.h | `setmaxnreg` wrapper. 대상에 SM_120a/SM_120f 포함 |
| `cccl-ptx-clusterlaunchcontrol.h` | https://raw.githubusercontent.com/NVIDIA/cccl/main/libcudacxx/include/cuda/__ptx/instructions/generated/clusterlaunchcontrol.h | CLC wrapper. `try_cancel`은 base SM_100 → sm_120에서도 사용 가능 |
| `cccl-ptx-cp_async_bulk_tensor.h` | https://raw.githubusercontent.com/NVIDIA/cccl/main/libcudacxx/include/cuda/__ptx/instructions/generated/cp_async_bulk_tensor.h | TMA(`cp.async.bulk.tensor`) wrapper. base SM_90 |
| `cccl-make_tma_descriptor.rst` | https://raw.githubusercontent.com/NVIDIA/cccl/main/docs/libcudacxx/extended_api/tma/make_tma_descriptor.rst | `cuda::make_tma_descriptor` (DLTensor → CUtensorMap). CC 9.0 이상 |
| `pytorch-cpp_extension.py` | https://raw.githubusercontent.com/pytorch/pytorch/main/torch/utils/cpp_extension.py | PyTorch C++/CUDA 확장 빌드(`TORCH_CUDA_ARCH_LIST` 처리) |

## CUTLASS C++ / CuTe

| 파일 | 원본 URL | 설명 |
|---|---|---|
| `cutlass-blackwell_functionality-20260908.md` | https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/blackwell_functionality.md | Blackwell 기능 문서(2026-09-08 스냅샷). "Blackwell SM120 GEMMs" 절 |
| `cutlass-79a_blackwell_geforce_nvfp4_bf16_gemm.cu` | https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu | SM120 NVFP4×NVFP4→BF16 GEMM 예제 |
| `cutlass-cute-arch-mma_sm120.hpp` | https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/arch/mma_sm120.hpp | SM120 MMA atom (`SM120_16x8x32_TN<f4/f6/f8 조합>`) |
| `cutlass-sm120_mma_builder.inl` | https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/collective/builders/sm120_mma_builder.inl | SM120 collective builder. F8F6F4 전용, TN 전용, cluster 1 강제 |
| `cutlass-arch-reg_reconfig.h` | https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/arch/reg_reconfig.h | `setmaxnreg` 활성 조건(`__CUDA_ARCH__ == 1200` 포함) |
| `cutlass-arch-grid_dependency_control.h` | https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/arch/grid_dependency_control.h | PDL(GDC) 활성 조건(1200/1210 포함) |
| `cutlass-sm90_gemm_tma_warpspecialized_pingpong.hpp` | https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp | Hopper pingpong 커널(SM120 스케줄의 원형) |
| (링크만) | https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm120_sparse.hpp | SM120 sparse MMA atom |
| (링크만) | https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/builders/sm120_sparse_mma_builder.inl | SM120 sparse builder (blockscaled sparse / array / blockwise builder도 같은 폴더) |
| (링크만) | https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm120_sparse_tensorop_gemm | SM120 sparse GEMM 단위 테스트 |

## CuTe DSL (Python)

| 파일 | 원본 URL | 설명 |
|---|---|---|
| `cute-dsl-overview.html` (+`.txt`) | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html | CuTe DSL 개요 |
| `cute-dsl-limitations.html` (+`.txt`) | https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/limitations.html | 제약 사항(32-bit layout, Windows 미지원, 디버깅 제한 등) |
| `cutedsl-limitations.rst` | https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/pythonDSL/limitations.rst | 위 문서의 원본 소스 |
| `cutedsl-faqs.rst` | https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/pythonDSL/faqs.rst | FAQ |
| `cutedsl-tvm_ffi_compilation.rst` | 원본 경로 (미확인) — CUTLASS 저장소 `media/docs/pythonDSL/` 하위 문서 | TVM-FFI를 통한 컴파일·프레임워크 연동 |
| `cutedsl-nvgpu-warp-mma.py` | https://raw.githubusercontent.com/NVIDIA/cutlass/main/python/CuTeDSL/cutlass/cute/nvgpu/warp/mma.py | warp MMA op: F16/BF16(sm_80+), FP8(sm_89+), F16 sparse(sm_80+), `MmaSM120BlockScaledOp`(sm_120a/120f/121a/121f) |
| `cutedsl-nvgpu-cpasync-copy.py` | https://raw.githubusercontent.com/NVIDIA/cutlass/main/python/CuTeDSL/cutlass/cute/nvgpu/cpasync/copy.py | TMA copy op(sm_90+), gather4(sm_100+), multicast op |
| `cutedsl-base_dsl-runtime-cuda.py` | https://raw.githubusercontent.com/NVIDIA/cutlass/main/python/CuTeDSL/cutlass/base_dsl/runtime/cuda.py | DSL 런타임(아키텍처 감지) |
| `cutedsl-geforce-dense_gemm.py` | https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py | `Sm120GemmKernel`: FP16/BF16, TMA, multi-stage, `setmaxregister_increase` |
| `cutedsl-geforce-dense_blockscaled_gemm_persistent_pingpong.py` | https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_pingpong.py | `Sm120BlockScaledGemmKernel`: NVFP4/MXFP4, TMA, warp specialization, persistent pingpong |
| (링크만) | https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute_ext/blackwell/dense_gemm/sm120_dense_block_scaled_gemm_persistent_pingpong.py | cute_ext 버전 SM120 block-scaled 예제 |

## cuTile / CUDA Tile

| 파일 | 원본 URL | 설명 |
|---|---|---|
| `cuda-13.1-release-notes.html` (+`.txt`) | https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html | CUDA Tile(Tile IR, cuTile) 최초 도입. 초기에는 Blackwell급만 지원 |
| `tile-ir-docs.html` (+`.txt`) | https://docs.nvidia.com/cuda/tile-ir/latest/ | Tile IR 문서 |
| `cuda-tile-AttrDefs.td` | https://raw.githubusercontent.com/NVIDIA/cuda-tile/main/include/cuda_tile/Dialect/CudaTile/IR/AttrDefs.td | Tile IR 지원 아키텍처 목록(SM_120은 13.1부터) |
| `cutile-python-README.md` | https://raw.githubusercontent.com/NVIDIA/cutile-python/main/README.md | 요구 사항(driver r580+, tileiras 버전별 지원 GPU) |
| `cutile-python-CHANGELOG.md` | https://raw.githubusercontent.com/NVIDIA/cutile-python/main/CHANGELOG.md | 1.6.0(2026-09-08)까지 릴리스 노트. `mma_scaled`(1.4.0/CTK 13.3), PDL(1.6.0/CTK 13.4) |
| `cutile-python-quickstart.html` (+`.txt`) | https://docs.nvidia.com/cuda/cutile-python/quickstart.html | 요구 사항: CC 8.x/9.x/10.x/11.x/12.x |
| `cutile-python-quickstart.rst` | 원본 경로 (미확인) — cutile-python 저장소 docs 소스 | 위 페이지의 원본 소스 |
| `cutile-python-performance.html` (+`.txt`) | https://docs.nvidia.com/cuda/cutile-python/performance.html | 성능 가이드 |

## 라이브러리 호출 경로 (커널 작성 X)

| 파일 | 원본 URL | 설명 |
|---|---|---|
| `cublas-docs.html` (+`.txt`) | https://docs.nvidia.com/cuda/cublas/index.html | cuBLASLt block scaling(VEC16_UE4M3, VEC32_UE8M0), FP8 TN 제약(CC 12.x 포함), FP64 에뮬레이션 지원 CC |
| `cudnn-frontend-README.md` | https://raw.githubusercontent.com/NVIDIA/cudnn-frontend/main/README.md | cuDNN frontend. 대상 Hopper, Blackwell(B200/GB200/GB300) |

## Numba-CUDA / Warp

| 파일 | 원본 URL | 설명 |
|---|---|---|
| `numba-cuda-docs-index.html` (+`.txt`) | https://nvidia.github.io/numba-cuda/ | 유지보수 모드 공지(Numba-CUDA-MLIR로 이전 권장) |
| `numba-cuda-installation.rst` | https://raw.githubusercontent.com/NVIDIA/numba-cuda/main/docs/source/user/installation.rst | 지원 CC(CUDA 13: 7.5~12.1) |
| `numba-cuda-cuda_ffi.rst` | https://raw.githubusercontent.com/NVIDIA/numba-cuda/main/docs/source/user/cuda_ffi.rst | CUDA C++ 디바이스 함수 링크(FFI) |
| `warp-docs-index.html` (+`.txt`) | https://nvidia.github.io/warp/ | Warp 문서 첫 페이지 |
| `warp-faq.rst` | https://raw.githubusercontent.com/NVIDIA/warp/main/docs/user_guide/faq.rst | Warp tile vs cuTile 비교. Warp tile은 TMA를 쓰지 않음 |
| `warp-build_dll.py` | https://raw.githubusercontent.com/NVIDIA/warp/main/warp/_src/build_dll.py | Warp 네이티브 빌드 스크립트 |
