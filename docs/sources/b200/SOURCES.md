# B200 1차 자료 목록 (sources/b200)

모든 파일 수집일: **2026-09-11** (`curl -L`, 브라우저 User-Agent). 각 파일은 `file`로 형식을 확인했습니다(PDF/HTML). PDF는 `pdftotext -layout`로 추출한 `.txt`를 옆에 두었습니다. 텍스트 레이어가 없는 PDF는 `tesseract`로 OCR한 `.ocr.txt`를 두었습니다.

## 데이터시트 / 제품 페이지 (NVIDIA)

| 파일 | URL | 설명 |
|---|---|---|
| `blackwell-datasheet_nortech.pdf` (+`.txt`) | https://nor-tech.com/wp-content/uploads/2026/03/blackwell-datasheet-B200.pdf | NVIDIA Blackwell Datasheet (NVIDIA 문서번호 4204213, OCT25). GB200 NVL72 / GB200 NVL4 / HGX B200 스펙 표(GPU별 sparse 수치, "Dense is one-half" 각주)가 들어 있습니다. 파트너(Nor-Tech)가 호스팅한 NVIDIA 원본 사본입니다. 같은 문서가 https://productinformation.ocf.co.uk/media/nvldqqds/blackwell-datasheet-4204213.pdf 에도 있습니다(텍스트 동일, 중복이라 삭제). |
| `blackwell-datasheet_openzeka.pdf` (+`.txt`) | https://openzeka.com/wp-content/uploads/2025/02/blackwell-datasheet.pdf | 같은 데이터시트의 이전 판(문서번호 3384703, DEC24, "Preliminary"). 수치 변화를 추적하는 데 씁니다. |
| `blackwell-architecture-technical-brief.pdf` (+`.ocr.txt`) | https://cdn.prod.website-files.com/61dda201f29b7efc52c5fbaf/6602ea9d0ce8cb73fb6de87f_nvidia-blackwell-architecture-technical-brief.pdf | NVIDIA Blackwell Architecture Technical Brief (2024-03, 22쪽, 사전 스펙) 사본입니다. 이미지 전용 PDF라 tesseract로 OCR했습니다. HGX B200 원래 스펙(192 GB, 8 TB/s, PCIe Gen6)이 들어 있습니다. NVIDIA 원본 링크는 https://nvdam.widen.net/s/xqt56dflgh/nvidia-blackwell-architecture-technical-brief.pdf 입니다(curl로 받으면 HTML 뷰어가 와서 PDF를 받지 못함). |
| `dgx-b200_nvidia.html` | https://www.nvidia.com/en-us/data-center/dgx-b200/ | DGX B200 제품 페이지. 시스템 스펙 표(1,440 GB, 64 TB/s, ~14.3 kW, Xeon 8570, CX-7/BF-3)가 있습니다. |
| `dgx-b200-datasheet_pny.pdf` (+`.txt`) | https://www.pny.com/en-eu/File%20Library/Unassigned/dgx-scale-ai-infrastructure-dgx-b200-datasheet-nvidia-pny-webonly.pdf | DGX B200 데이터시트(PNY 재배포판). 스펙 값 대부분이 윤곽선 처리돼 있어 텍스트 추출이 불완전합니다. 수치는 제품 페이지를 기준으로 삼았습니다. |
| `dgx-b200-datasheet_nvidia.html` | https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet | NVIDIA 공식 DGX B200 데이터시트 뷰어 페이지(PathFactory 셸). 본문 PDF는 따로 받지 못했습니다. |
| `hgx_nvidia.html` | https://www.nvidia.com/en-us/data-center/hgx/ | HGX 플랫폼 페이지. HGX B300 / HGX B200 스펙 표("Sparse \| Dense" 각주 포함)가 있습니다. |
| `hgx-b200-pcf-summary.pdf` (+`.txt`) | https://images.nvidia.com/aem-dam/Solutions/documents/HGX-B200-PCF-Summary.pdf | HGX B200 탄소발자국 요약. 베이스보드 32 kg, GPU당 최대 1000 W, B200 180 GB HBM3E, 공랭/수랭 구성이 명시돼 있습니다. |
| `gb200-nvl72_nvidia.html` | https://www.nvidia.com/en-us/data-center/gb200-nvl72/ | GB200 NVL72 제품 페이지와 스펙 표(NVL72 / GB200 Superchip). |
| `blackwell-architecture_nvidia.html` | https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/ | Blackwell 아키텍처 소개 페이지. |
| `lenovo-lp2226-hgx-b200-180gb-1000w.pdf` (+`.txt`) | https://lenovopress.lenovo.com/lp2226.pdf | Lenovo ThinkSystem HGX B200 180GB 1000W 제품 가이드(OEM). SXM6 폼팩터, MIG 7×23 GB가 나옵니다. |

## 개발자 문서

| 파일 | URL | 설명 |
|---|---|---|
| `cuda-gpus_compute-capability.html` | https://developer.nvidia.com/cuda-gpus | CUDA Compute Capability 표(B200/GB200 = 10.0). |
| `mig-user-guide_supported-profiles.html` | https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html | MIG 사용자 가이드의 B200 180GB 프로파일 표(1g.23gb ~ 7g.180gb). |

## NVIDIA Technical Blog / NVIDIA Blog

| 파일 | URL | 설명 |
|---|---|---|
| `blog_nvfp4-inference-intro.html` | https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ | NVFP4 소개(E2M1, 16원소 블록, E4M3 스케일 + FP32 텐서 스케일). |
| `blog_nvfp4-training.html` | https://developer.nvidia.com/blog/using-nvfp4-low-precision-model-training-for-higher-throughput-without-losing-accuracy/ | FP8-CS/MXFP8/NVFP4 학습 비교(B200에서 NeMo Megatron Bridge로 8B 모델 1T 토큰). |
| `blog_mlperf-training-v4.1.html` | https://developer.nvidia.com/blog/nvidia-blackwell-doubles-llm-training-performance-in-mlperf-training-v4-1/ | MLPerf Training v4.1: 첫 B200 제출(preview), GPU당 2x/2.2x. |
| `blog_mlperf-training-v5.0.html` | https://developer.nvidia.com/blog/nvidia-blackwell-delivers-up-to-2-6x-higher-performance-in-mlperf-training-v5-0/ | MLPerf Training v5.0: Hopper 대비 GPU당 최대 2.6x. |
| `blog_mlperf-training-v5.1.html` | https://developer.nvidia.com/blog/nvidia-blackwell-enables-3x-faster-training-and-nearly-2x-training-performance-per-dollar-than-previous-gen-architecture/ | MLPerf Training v5.1: NVFP4 학습, GB200 NVL72가 Hopper FP8 대비 3.2x. |
| `blog_mlperf-training-6.0.html` | https://blogs.nvidia.com/blog/blackwell-mlperf-training-6-0/ | MLPerf Training 6.0 (2026-06): GB200/GB300 NVL72 결과. |
| `blog_mlperf-inference-v4.1_blackwell-debut.html` | https://blogs.nvidia.com/blog/mlperf-inference-benchmark-blackwell/ | MLPerf Inference v4.1: Blackwell 첫 제출, Llama 2 70B에서 H100 대비 최대 4x. |
| `blog_mlperf-inference-v5.0.html` | https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0/ | MLPerf Inference v5.0: DGX B200 vs 8×H200 표(엔트리 5.0-0056/0060). |
| `blog_blackwell-ultra-mlperf-inference.html` | https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/ | Blackwell Ultra의 MLPerf Inference 데뷔(후속 제품 참고). |
| `blog_inside-blackwell-ultra.html` | https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/ | Blackwell Ultra 칩 해설. Hopper/Blackwell/Blackwell Ultra 비교 표(NVFP4 dense\|sparse, SFU, TGP), TMEM 256 KB/SM, 2-CTA MMA 설명이 있습니다. |
| `blog_deepseek-r1-world-record.html` | https://developer.nvidia.com/blog/nvidia-blackwell-delivers-world-record-deepseek-r1-inference-performance/ | DGX B200 DeepSeek-R1 추론(>250 TPS/user, >30k TPS), DGX H200 대비 >3x. |
| `trtllm-blog_deepseek-r1-b200-latency.html` | https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog1_Pushing_Latency_Boundaries_Optimizing_DeepSeek-R1_Performance_on_NVIDIA_B200_GPUs.html | TensorRT-LLM 기술 블로그: 8×B200에서 DeepSeek-R1 min-latency 368 TPS/user. |

## MLPerf 결과 페이지 (MLCommons)

| 파일 | URL | 설명 |
|---|---|---|
| `mlcommons_training.html` | https://mlcommons.org/benchmarks/training/ | MLPerf Training 벤치마크/결과 페이지. |
| `mlcommons_inference-datacenter.html` | https://mlcommons.org/benchmarks/inference-datacenter/ | MLPerf Inference Datacenter 벤치마크/결과 페이지. |

## 제3자 연구

| 파일 | URL | 설명 |
|---|---|---|
| `arxiv-2608.26575_cc-blackwell.pdf` (+`.txt`) | https://arxiv.org/abs/2608.26575 | "Benchmarking Confidential Computing Performance on NVIDIA Blackwell GPUs" (Confidential.ai, 2026). 8×B200 + Intel TDX 환경에서 CC 오버헤드를 측정했습니다. |

## 링크만 (다운로드 실패 또는 미수집)

| URL | 사유 / 설명 |
|---|---|
| https://nvdam.widen.net/s/xqt56dflgh/nvidia-blackwell-architecture-technical-brief.pdf | 링크만. curl로 받으면 HTML 뷰어가 옵니다(PDF 아님, 삭제). 같은 문서의 사본을 위에 보관했습니다. |
| https://www.aspsys.com/wp-content/uploads/2025/05/nvidia-dgx-b200-datasheet.pdf | 링크만. HTTP 403(삭제). |
| https://www.primeline-solutions.com/media/categories/server/nach-gpu/nvidia-hgx-h200/nvidia-blackwell-b200-datasheet.pdf | 링크만. HTTP 403(삭제). |
| https://resources.nvidia.com/en-us-blackwell-architecture | 링크만. NVIDIA 공식 아키텍처 기술 개요 뷰어입니다. 아키텍처 문서(01) 담당 에이전트가 따로 수집합니다. |
| https://docs.nvidia.com/cuda/parallel-thread-execution/ | 링크만. PTX ISA(tcgen05 / Tensor Memory 명령어) 문서입니다. 세부 내용은 01-architecture.md를 참조하세요. |
| https://github.com/NVIDIA/cutlass | 링크만. CUTLASS / CuTe DSL (SM100 커널 예제). |
| https://cvw.cac.cornell.edu/gpu-architecture/horizon-gpus-blackwell-b200/b200_sm | 링크만(WebFetch로 확인). Cornell Virtual Workshop의 B200 SM 해설(제3자 자료, SM당 FP32 코어 128개와 TMEM 256 KB를 설명합니다). |
