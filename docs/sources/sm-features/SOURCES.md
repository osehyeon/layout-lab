# SM 타깃별 기능 표(06-sm-features.md)용 1차 출처 목록

- 수집일(retrieval date): **2026-09-11** (모든 항목 공통)
- 수집 방법: `curl -L` (브라우저 User-Agent), 다운로드 후 `file`로 PDF/HTML 여부 확인, 오류/HTML 뷰어 응답은 삭제.
- 텍스트 추출: PDF는 `pdftotext -layout`, HTML은 `textutil -convert txt` → 같은 이름의 `.txt`(grep용).
- 이미 `../architecture/`에 있는 자료(PTX ISA, CUDA Programming Guide 13.4 부록, CUDA 13.4 릴리스 노트, CUDA Features Archive, Blackwell tuning/compatibility guide, 12.9 family-specific 블로그, CUTLASS 문서, RTX/RTX PRO Blackwell 백서, Blackwell Ultra 블로그, CUDA GPUs 목록)는 중복 다운로드하지 않고 재사용했다 → [../architecture/SOURCES.md](../architecture/SOURCES.md).

## 다운로드된 파일

| 파일 | 원본 URL | 설명 |
|---|---|---|
| `volta-tuning-guide.html` (+`.txt`) | https://docs.nvidia.com/cuda/volta-tuning-guide/index.html | Volta Tuning Guide: SM당 FP32 64/FP64 32/Tensor Core 8, 64 warps·32 blocks/SM, SMEM 96 KB, 통합 L1/SMEM 128 KB, independent thread scheduling 주의점 |
| `turing-tuning-guide.html` (+`.txt`) | https://docs.nvidia.com/cuda/turing-tuning-guide/index.html | Turing Tuning Guide: 32 warps·16 blocks/SM, SMEM 64 KB, 통합 96 KB, INT8/INT4/INT1 Tensor Core, Volta TC 바이너리는 Turing 피크의 절반 |
| `ampere-tuning-guide.html` (+`.txt`) | https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html | Ampere Tuning Guide: CC 8.0/8.6 occupancy·SMEM(164/100 KB), async copy, split arrive/wait barrier, 3세대 TC(FP64 DMMA/BF16/TF32), 8.6의 FP32 2x |
| `ada-tuning-guide.html` (+`.txt`) | https://docs.nvidia.com/cuda/ada-tuning-guide/index.html | Ada Tuning Guide: 48 warps·24 blocks/SM, SMEM 100 KB, 4세대 TC + FP8 Transformer Engine, AD102 L2 96 MB |
| `hopper-tuning-guide.html` (+`.txt`) | https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html | Hopper Tuning Guide: SMEM 228/227 KB, 통합 256 KB, TMA, Thread Block Cluster(portable 8 / H100 non-portable 16), DSMEM, DPX |
| `hopper-compatibility-guide.html` (+`.txt`) | https://docs.nvidia.com/cuda/hopper-compatibility-guide/index.html | Hopper Compatibility Guide: cubin/PTX 호환 규칙, `sm_90a`/`compute_90a` 비호환성 |
| `cuda-13.0-release-notes.html` (+`.txt`) | https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html | CUDA 13.0 Release Notes: "SM101 has been renumbered as SM110", Maxwell/Pascal/Volta 오프라인 컴파일·라이브러리 지원 제거, Thor cuDLA 미지원 |
| `cuda-12.9-c-programming-guide.html` (+`.txt`) | https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html | CUDA C++ Programming Guide 12.9 (구판, ~4 MB): CC 5.0~12.0 전체 Technical Specifications 표(13.x 판에서 빠진 7.0/7.2 값), CC별 SM 구성 절, 산술 처리량 표 |
| `volta-architecture-whitepaper.pdf` (+`.txt`) | https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf | NVIDIA Tesla V100 (GV100) 백서: 1세대 Tensor Core 8/SM, 통합 L1/SMEM, independent thread scheduling, HW MPS, Cooperative Groups |
| `nvidia-turing-gpu-architecture-whitepaper.pdf` (+`.txt`) | https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf | NVIDIA Turing (TU102) 백서: INT8/INT4 Tensor Core, RT Core, SM당 TC 8·L1/SMEM 96 KB, FP64 = FP32의 1/32 |
| `nvidia-ampere-architecture-whitepaper.pdf` (+`.txt`) | https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf | NVIDIA A100 백서: 3세대 TC 4/SM, TF32/BF16/FP64 TC, 2:4 sparsity, async copy/barrier, L2 residency control, MIG, compute data compression |
| `nvidia-ampere-ga102-gpu-architecture-whitepaper.pdf` (+`.txt`) | https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf | NVIDIA GA102 백서(v2): GA10x SM(TC 4/SM, FP32 2x), FP64 = FP32의 1/64, sparsity |
| `nvidia-ada-gpu-architecture.pdf` (+`.txt`) | https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf | NVIDIA Ada (AD102) 백서: 4세대 TC 4/SM + FP8, SER, FP64 1/64("FP64 Tensor Core code 정확성용"), L2 96 MB |
| `nvidia-h100-tensor-core-hopper-whitepaper.pdf` (+`.txt`) | 원본: https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper (HTML 뷰어만 반환되어 실패) → 실제 취득: https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf (동일 NVIDIA PDF의 제3자 호스팅 사본) | NVIDIA H100 Tensor Core GPU Architecture 백서(GTC22): 4세대 TC 4/SM, FP8, DPX, Thread Block Cluster, DSMEM, TMA, asynchronous transaction barrier, Confidential Computing |

## 링크만 (다운로드 실패 또는 미보관)

| 대상 | URL | 사유 |
|---|---|---|
| H100 백서 (NVIDIA 원본 호스트) | https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper , https://nvdam.widen.net/s/9bz6dw7dqr/gtc22-whitepaper-hopper | 링크만 — HTML 뷰어 반환. 동일 PDF를 제3자 사본으로 취득(위 표) |
| Jetson AGX Xavier(CC 7.2) / Jetson AGX Orin(CC 8.7) / Jetson Thor(CC 11.0) 모듈 데이터시트 | https://developer.nvidia.com/embedded/jetson-modules | 링크만 — 미수집. SM 한도는 CUDA Programming Guide 표로 대체 |
