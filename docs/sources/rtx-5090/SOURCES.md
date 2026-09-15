# RTX 5090 1차 자료 목록 (sources/rtx-5090)

수집일: 2026-09-11 (전 항목 동일). 모든 파일은 `curl -L`(브라우저 User-Agent)로 받았고 `file`로 PDF/HTML 여부를 확인함. Cloudflare 챌린지 페이지나 404 페이지는 삭제하고 아래 "링크만" 항목으로 옮김.

## NVIDIA 공식 자료

| 파일 | URL | 설명 |
|---|---|---|
| `nvidia-rtx-blackwell-gpu-architecture.pdf` / `.txt` | https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf | **핵심 자료.** NVIDIA RTX Blackwell GPU Architecture 백서 v1.1 (GB202/GB203/GB205). Appendix A Table 3에 RTX 5090 전체 스펙과 정밀도별 Tensor TFLOPS(FP16/FP32 accumulate, dense/sparse)가 있음. v1.1에는 ECC와 INT 관련 업데이트가 추가됨. `.txt`는 `pdftotext -layout`으로 추출한 텍스트 |
| `nvidia-geforce-rtx-5090-product-page.html` | https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/ | GeForce RTX 5090 제품/스펙 페이지: AI TOPS 3352, boost/base clock, NVENC/NVDEC, NVLink "No", CUDA Capability 12.0, TGP 575 W, 전원 커넥터 |
| `nvidia-geforce-rtx-50-series-page.html` | https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/ | GeForce RTX 50 시리즈 데스크톱 라인업 페이지 |
| `nvidia-geforce-rtx-50-laptops-page.html` | https://www.nvidia.com/en-us/geforce/laptops/50-series/ | RTX 50 노트북 GPU 스펙 표: RTX 5090 Laptop 1824 AI TOPS, 10496 CUDA cores, 24 GB, 896 GB/s |
| `nvidia-newsroom-rtx-50-ces2025.html` | https://nvidianews.nvidia.com/news/nvidia-blackwell-geforce-rtx-50-series-opens-new-world-of-ai-computer-graphics | CES 2025 보도자료: RTX 5090 1월 30일 출시, $1,999, 3,352 AI TOPS |
| `nvidia-blog-new-ai-sdks-rtx-50.html` | https://developer.nvidia.com/blog/new-ai-sdks-and-tools-released-for-nvidia-blackwell-geforce-rtx-50-series-gpus/ | NVIDIA Technical Blog (2025-01-30): CUDA 12.8, TensorRT 10.8(FP4), PyTorch Windows/Linux nightly 지원 |
| `nvidia-blog-tensorrt-fp4-image-gen-rtx-50.html` | https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus/ | NVIDIA Technical Blog (2025-05-14): RTX 5090에서 FLUX FP4 추론. FC layer 기준 FP8 대비 최대 3.1x, 파이프라인 지연 시간 수치 |
| `nvidia-blog-unsloth-blackwell.html` | https://developer.nvidia.com/blog/train-an-llm-on-an-nvidia-blackwell-desktop-with-unsloth-and-scale-it/ | NVIDIA Technical Blog: Unsloth로 Blackwell(GeForce RTX 50 포함)에서 LLM fine-tuning. 벤더 주장 수치 |
| `nvidia-blog-open-kernel-modules-transition.html` | https://developer.nvidia.com/blog/nvidia-transitions-fully-towards-open-source-gpu-kernel-modules/ | NVIDIA 블로그: Blackwell은 open-source GPU kernel modules 필수이며 proprietary 모듈은 미지원 |
| `nvidia-driver-guide-kernel-modules.html` | https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/kernel-modules.html | NVIDIA Driver Installation Guide, Kernel Modules 절: `nvidia-open` 패키지 설치 방법 |
| `nvidia-open-gpu-kernel-modules-readme.html` | https://github.com/NVIDIA/open-gpu-kernel-modules | open-gpu-kernel-modules README: 지원 GPU 목록(RTX 5090, 5090 D, 5090 D v2, RTX 5090 Laptop 등 device ID) |
| `nvidia-cuda-gpus-compute-capability.html` | https://developer.nvidia.com/cuda-gpus | CUDA GPU Compute Capability 표: RTX 5090 = 12.0, B200 = 10.0 |
| `cuda-12.8-release-notes.html` | https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html | CUDA 12.8 릴리스 노트: SM_100/SM_101/SM_120 컴파일러 지원 추가, 최소 드라이버 버전 (Linux ≥570.26, Windows ≥570.65) |
| `cuda-12.8-c-programming-guide.html` | https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html | CUDA 12.8 C++ Programming Guide: Compute Capability 12.0 절 (SM당 FP32 core 128개, FP64 core 2개, unified L1/shared 128 KB) |
| `nvidia-blackwell-compatibility-guide.html` | https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html | Blackwell 호환성 가이드: cubin/PTX 호환성 규칙 |
| `nvidia-mig-user-guide-supported-gpus.html` | https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html | MIG 지원 GPU 표: B200과 RTX PRO 6000/5000 Blackwell 등이 있고 GeForce는 없음 |
| `nvidia-forum-p2p-two-rtx-5090.html` | https://forums.developer.nvidia.com/t/p2p-issue-using-two-rtx-5090-gpus/326776 | NVIDIA Developer Forums (2025-03): "RTX 50-series ... p2p is not supported for GPU<->GPU P2P" 답변, NCCL fallback 이슈 |
| `cutlass-blackwell-functionality.md` | https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/blackwell_functionality.md | CUTLASS 문서: SM100 `tcgen05.mma` 대비 SM120 `mma.sync` block-scaled 명령. GeForce는 multicast가 없어 cluster shape 1x1x1 고정 |

## 제3자 자료

| 파일 | URL | 설명 |
|---|---|---|
| `puget-rtx-5090-5080-ai-review.html` | https://www.pugetsystems.com/labs/articles/nvidia-geforce-rtx-5090-amp-5080-ai-review/ | Puget Systems RTX 5090/5080 AI 리뷰 (출시 시점, Windows): llama.cpp, MLPerf Client, SD.Next, Resolve, Topaz |
| `phoronix-rtx-5090-linux-compute.html` | https://www.phoronix.com/review/nvidia-geforce-rtx5090-linux | Phoronix RTX 5090 Linux GPU compute 초기 벤치마크 (CUDA 12.8 번들 드라이버 570.86.10) |
| `arxiv-2601.09527-consumer-blackwell-llm.pdf` / `.html` / `.txt` | https://arxiv.org/abs/2601.09527 (PDF: https://arxiv.org/pdf/2601.09527v1) | Knoop & Holtmann (2026-01), "Private LLM Inference on Consumer Blackwell GPUs": vLLM 0.12 기반 RTX 5060 Ti/5070 Ti/5090 비교, NVFP4/MXFP4/W4A16/BF16 |
| `tomshardware-qwen-27b-rtx-5090-benchmark.html` | https://www.tomshardware.com/tech-industry/artificial-intelligence/benchmarking-qwen-3-8-27b-on-rtx-5090-and-beyond-vram-capacity-alone-cant-overcome-severe-software-and-inference-engine-bottlenecks | Tom's Hardware (2026): Qwen 3.8 27B를 RTX 5090 1장/2장에서 llama.cpp, vLLM(NVFP4), SGLang으로 측정 |
| `colfax-nvfp4-blockscaled-gemm-sm12x.html` | https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/ | Colfax Research: SM12x NVFP4 block-scaled GEMM 튜토리얼. SM12x에는 tcgen05/TMEM이 없고 `mma.sync`를 쓰며 TMA/CLC/PDL은 지원 |
| `tomshardware-rtx-5090-cable-150c.html` | https://www.tomshardware.com/pc-components/gpus/rtx-5090-cable-overheats-to-150-degrees-celsius-uneven-current-distribution-likely-the-culprit | Tom's Hardware (2025-02-11): der8auer 측정. 12V 와이어 1가닥에 22 A 이상, 150 °C 이상 |
| `tomshardware-geforce-lacks-p2p.html` | https://www.tomshardware.com/news/nvidia-confirms-geforce-cards-lack-p2p-support | Tom's Hardware (2023-02): NVIDIA가 GeForce(RTX 4090)의 P2P 미지원을 확인 (배경 자료) |
| `tomshardware-rtx-50-laptop-gpus.html` | https://www.tomshardware.com/pc-components/gpus/nvidia-introduces-rtx-5090-rtx-5080-and-rtx-5070-laptop-gpus-rtx-50-blackwell-goes-mobile-with-up-to-24gb-of-gddr7-memory | Tom's Hardware (2025-01-07): RTX 50 노트북 GPU 발표 기사 |
| `pytorch-2.7-release-blog.html` | https://pytorch.org/blog/pytorch-2-7/ | PyTorch 2.7 릴리스 (2025-04-23): Blackwell 지원과 CUDA 12.8 pre-built wheel (Prototype 기능) |

## 링크만 (다운로드 실패 또는 미수집)

| URL | 사유 / 설명 |
|---|---|
| https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216 | 링크만. Cloudflare 챌린지로 받지 못함 (TechPowerUp GPU DB) |
| https://videocardz.com/newz/german-media-outlet-reports-melted-rtx-5090-12v-2x6-cable-in-their-press-test-system | 링크만. Cloudflare 챌린지 (언론사 테스트 시스템의 12V-2x6 용융 보도) |
| https://www.nvidia.com/en-us/geforce/news/rtx-5090-5080-available-now/ | 링크만. 404 (추정 URL) |
| https://forums.developer.nvidia.com/t/rtx-5090-not-working-with-pytorch-and-stable-diffusion-sm-120-unsupported/338015 | 링크만. 구버전 PyTorch의 "sm_120 is not compatible" 오류 사례 |
| https://github.com/NVIDIA/nccl/issues/1637 | 링크만. RTX 5090 2장 NCCL P2P 이슈 (포럼에서 신버전 NCCL로 해결됐다고 언급) |
| https://github.com/NVIDIA/cutlass/issues/3096 | 링크만. SM120 NVFP4 MoE grouped GEMM 오출력 이슈 (sm_120 커널 생태계 성숙도 참고) |
| https://github.com/NVIDIA/TensorRT-LLM/issues/11368 | 링크만. B200 크기 tile config를 SM12x(GB10)에 쓸 때 shared memory overflow |
| https://github.com/flashinfer-ai/flashinfer/issues/3628 | 링크만. SM120a MXFP8 block-scaled prefill attention RFC |
| https://www.notebookcheck.net/Nvidia-GeForce-RTX-5090-Laptop-Benchmarks-and-Specs.934947.0.html | 링크만. RTX 5090 Laptop의 GB203 다이와 TGP 관련 제3자 정보 |
