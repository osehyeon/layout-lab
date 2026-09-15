# Architecture / Software-stack 1차 출처 목록

- 수집일(retrieval date): **2026-09-11** (모든 항목 공통)
- 수집 방법: `curl -L` (브라우저 User-Agent), 다운로드 후 `file`로 PDF/HTML 여부 확인, 오류 페이지는 삭제.
- 텍스트 추출: PDF는 `pdftotext -layout`, HTML은 `textutil`/HTML 태그 제거 → 같은 이름의 `.txt`로 저장(grep용).
- 예외: `nvidia-blackwell-architecture-technical-brief.pdf`는 글꼴이 없는 이미지형 PDF(Skia 인쇄본)라 `pdftotext` 결과가 비어 있어 `pdftoppm`+`tesseract` OCR로 `.txt`를 만들었음 → OCR 오류 가능성 있으니 수치 인용 시 PDF 원본 확인 권장.

## 다운로드된 파일

| 파일 | 원본 URL | 설명 |
|---|---|---|
| `nvidia-blackwell-architecture-technical-brief.pdf` (+`.txt` OCR) | 원본: https://nvdam.widen.net/s/xqt56dflgh/nvidia-blackwell-architecture-technical-brief.pdf (HTML 뷰어만 반환되어 실패) → 실제 취득: https://cdn.prod.website-files.com/61dda201f29b7efc52c5fbaf/6602ea9d0ce8cb73fb6de87f_nvidia-blackwell-architecture-technical-brief.pdf (동일 NVIDIA PDF의 제3자 호스팅 사본, 22쪽) | NVIDIA Blackwell Architecture Technical Brief (2024-03): 208B transistors, NV-HBI 10 TB/s, 2세대 Transformer Engine, Decompression Engine, RAS, TEE-I/O, NVLink 5, GB200 사양표 |
| `nvidia-rtx-blackwell-gpu-architecture.pdf` (+`.txt`) | https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf | NVIDIA RTX Blackwell GPU Architecture whitepaper v1.1 (GeForce; GB202/GB203/GB205, RTX 5090 사양표, DLSS 4, Neural Shaders, NVENC/NVDEC) |
| `NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf` (+`.txt`) | https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf | NVIDIA RTX PRO Blackwell GPU Architecture whitepaper v1.1 (GB202 full chip 128 MB L2, RTX PRO 6000, MIG 구성) |
| `cuda-programming-guide-compute-capabilities.html` (+`.txt`) | https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html | CUDA Programming Guide 부록: family-specific target 호환표(Table 28), CC별 기능/SMEM/Tensor Core 입력 타입 표 |
| `cuda-c-programming-guide.html` (+`.txt`) | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html | CUDA Programming Guide 목차 페이지(13.x 개편판; 세부는 위 부록 파일 참고) |
| `ptx-isa.html` (+`.txt`) | https://docs.nvidia.com/cuda/parallel-thread-execution/index.html | PTX ISA 전체(단일 HTML, ~3.9 MB): tcgen05.*, Tensor Memory, mma.sync block_scale, cvt.rs 등 target ISA notes |
| `cuda-toolkit-release-notes.html` (+`.txt`) | https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html | CUDA Toolkit 13.4 Release Notes (현행): 드라이버 요구사항, sm_107(Rubin) dev preview, cuBLAS/cuDNN 등 라이브러리 노트 |
| `cuda-12.8-release-notes.html` (+`.txt`) | https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html | CUDA 12.8 Release Notes: SM_100/SM_101/SM_120 최초 컴파일러 지원, 드라이버 570.26+, cuBLASLt block-scaled FP4/FP8 |
| `CUDA_Features_Archive.pdf` (+`.txt`) | https://docs.nvidia.com/cuda/pdf/CUDA_Features_Archive.pdf | CUDA Features Archive (13.3판): 12.8/12.9/13.0 등 과거 릴리스 기능 모음(sm_103/sm_121/family-specific 도입 등) |
| `blackwell-tuning-guide.html` (+`.txt`) | https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html | Blackwell Tuning Guide 13.4: occupancy(warps/SM), SMEM 용량, cluster size 8/16, GB200 L2 126 MB, NVLink 5 |
| `blackwell-compatibility-guide.html` (+`.txt`) | https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html | Blackwell Compatibility Guide: cubin/PTX 호환 규칙, `a` 타깃 비호환성 |
| `nvidia-blog-family-specific-arch-cuda12.9.html` (+`.txt`) | https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/ | NVIDIA 기술 블로그(2025-05-01): `a` vs `f` suffix, `__CUDA_ARCH_FAMILY_SPECIFIC__` |
| `nvidia-blog-introducing-nvfp4.html` (+`.txt`) | https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ | NVFP4 소개 블로그: E2M1 + E4M3 scale/16 + FP32 per-tensor, MXFP4 비교 |
| `nvidia-blog-blackwell-ultra.html` (+`.txt`) | https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/ | Blackwell Ultra 심층 블로그: 160 SM, TMEM 256 KB/SM, NVFP4 15 PF dense, SFU 2x, Hopper/Blackwell/Ultra 비교표 |
| `nvidia-cuda-gpus.html` (+`.txt`) | https://developer.nvidia.com/cuda-gpus | CUDA GPU Compute Capability 목록(B200=10.0, B300=10.3, RTX 5090/RTX PRO=12.0, GB10=12.1) |
| `cutlass-blackwell_functionality.md` | https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/blackwell_functionality.md (렌더링: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html) | CUTLASS "Blackwell SM100 GEMMs" + "Blackwell SM120 GEMMs" 문서: tcgen05.mma 종류/처리량, block-scaled 타입, SM120 차이점 |
| `cutlass-CHANGELOG.md` | https://raw.githubusercontent.com/NVIDIA/cutlass/main/CHANGELOG.md | CUTLASS CHANGELOG (3.8.0 SM100 도입 ~ 4.8.0, 2026-08-25) |
| `arxiv-2310.10537-microscaling-formats.pdf` (+`.txt`) | https://arxiv.org/pdf/2310.10537 | "Microscaling Data Formats for Deep Learning" (Rouhani et al., 2023) — OCP MX 스펙 저자들의 논문. OCP 스펙 PDF 대체 자료 |
| `arxiv-2509.25149-nvfp4-pretraining.pdf` (+`.txt`) | https://arxiv.org/pdf/2509.25149 | "Pretraining Large Language Models with NVFP4" (NVIDIA, v2 2026-03): 12B/10T tokens NVFP4 학습 레시피 |
| `arxiv-2507.10789-dissecting-blackwell.pdf` (+`.txt`) | https://arxiv.org/pdf/2507.10789 | "Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks" (참고용 2차 연구) |
| `te-nvfp4.html` (+`.txt`) | https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html | Transformer Engine NVFP4 문서(2.20.0.dev): 2D 16x16 weight scaling, stochastic rounding, RHT |
| `pytorch-2.7-release-blog.html` (+`.txt`) | https://pytorch.org/blog/pytorch-2-7/ | PyTorch 2.7 릴리스 블로그: Blackwell 지원, cu128 wheels, Triton 3.3 |
| `tensorrt-10.8.0-release-notes.html` (+`.txt`) | https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.8.0.html | TensorRT 10.8.0 RN: Blackwell(GeForce 50) 지원, B200/GB200 early access, E2M1 FP4 |
| `nccl-2.25.1-release-notes.html` (+`.txt`) | https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-25-1.html | NCCL 2.25.1 RN: "Added support for Blackwell" |
| `cudnn-support-matrix.html` (+`.txt`) | https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html | cuDNN 9.26.0 Support Matrix: CC 10.0/10.3/10.7/12.0/12.1, 드라이버 요구 |
| `deepgemm-README.md` | https://raw.githubusercontent.com/deepseek-ai/DeepGEMM/main/README.md | DeepGEMM README: SM90/SM100 전용, CUDA 12.9+ |
| `flash-attention-README.md` | https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/README.md | FlashAttention README: FA3=Hopper, FA4(CuTeDSL)=Hopper/Blackwell(B200) |

## 링크만 (다운로드 실패 또는 미보관)

| 대상 | URL | 사유 |
|---|---|---|
| OCP Microscaling Formats (MX) Specification v1.0 | https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf | 링크만 — Cloudflare "Just a moment..." 챌린지 페이지만 반환(`/download` 경로도 동일). 대체로 arXiv 2310.10537 보관 |
| Blackwell Technical Brief (NVIDIA 원본 호스트) | https://nvdam.widen.net/s/xqt56dflgh/nvidia-blackwell-architecture-technical-brief.pdf , https://resources.nvidia.com/en-us-blackwell-architecture | 링크만 — HTML 뷰어 반환. 동일 PDF를 제3자 CDN에서 취득 |
| Transformer Engine NVFP4 문서 (docs.nvidia.com 판) | https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html | 링크만 — curl 시 "Page Not Found" 반환(브라우저에서는 열릴 수 있음). GitHub Pages 판으로 대체 |
| FlashAttention-4 SM120 지원 이슈 #2307 | https://github.com/Dao-AILab/flash-attention/issues/2307 | 링크만 (WebFetch로 확인: 2026-03-06 개설, SM120 미지원 상태) |
| vLLM SM120 NVFP4 관련 이슈 | https://github.com/vllm-project/vllm/issues/31085 , https://github.com/vllm-project/vllm/issues/47749 , https://github.com/vllm-project/vllm/pull/50288 | 링크만 (커뮤니티 이슈, 2차 출처) |
| FlashInfer SM120 NVFP4 GEMM 이슈 | https://github.com/flashinfer-ai/flashinfer/issues/2577 | 링크만 (2차 출처) |
| TensorRT-LLM SM120/SM121 FMHA cubin 이슈 | https://github.com/NVIDIA/TensorRT-LLM/issues/11799 | 링크만 (2차 출처) |
| Colfax: NVFP4 blockscaled GEMM on RTX PRO 6000 (SM120) | https://research.colfax-intl.com/optimizing-an-nvfp4-blockscaled-gemm-on-rtx-pro-6000-blackwell-gpu-sm120/ | 링크만 (참고용 튜토리얼) |
