# RTX PRO 6000 Blackwell — 1차 자료 목록

검색일(retrieval date): 2026-09-11 (전부). 모든 PDF는 `file`로 PDF임을 확인했고, 옆에 `pdftotext -layout`로 추출한 `.txt`가 있음. HTML은 `<title>`을 확인해 에러 페이지가 아님을 확인함.

## 다운로드한 자료

| 파일 | URL | 설명 |
|---|---|---|
| rtx-pro-6000-workstation-datasheet.pdf (.txt) | https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf | NVIDIA Workstation Edition(600 W) 공식 데이터시트(개정판 5349469, 2026년 6월): 4000 AI TOPS(sparse FP4), 126 TFLOPS FP32, 1792 GB/s, MIG 4x24GB |
| rtx-pro-6000-maxq-datasheet.pdf (.txt) | https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-6000-max-q/workstation-datasheet-blackwell-rtx-pro-6000-max-q-nvidia-3519233.pdf | NVIDIA Max-Q Workstation Edition(300 W) 공식 데이터시트(2025년 4월): 3511 AI TOPS, 110 TFLOPS FP32 |
| rtx-pro-6000-server-datasheet-nvidia.pdf (.txt) | https://dam-cdn.nvd.orangelogic.com/AssetLink/707m1632ypg4du1fj3ci1jo3h4w1k78j.pdf (https://resources.nvidia.com/en-us-rtx-pro-6000 에서 링크) | NVIDIA Server Edition 공식 데이터시트(4682150, 2025년 12월): 4 PFLOPS FP4, 1597 GB/s, 최대 600 W(설정 가능), Confidential compute 지원, 공랭/수랭 |
| rtx-pro-6000-server-datasheet-pny.pdf (.txt) | https://www.pny.com/en-eu/file%20library/professional/datasheet/data%20center%20cards/pny-nvidia-rtx-pro-6000-blackwell-server-edition-datasheet.pdf | 같은 데이터시트의 PNY 배포판(2025년 4월, PNY 부품번호 포함, "Passive"만 표기) |
| rtx-pro-6000-server-product-brief-SP-12355.pdf (.txt) | https://dam-cdn.nvd.orangelogic.com/AssetLink/3km2720jiy76r06ctf8mxg21743p1058.pdf | NVIDIA Server Edition Product Brief SP-12355-001_v02(2025년 6월): base/boost 클럭 1852/2430 MHz, 600/450 W 전력 모드, SR-IOV 48 VF, NVLink 미지원, BAR1 크기, vGPU 19.0+ |
| nvidia-rtx-pro-server-datasheet.pdf (.txt) | https://dam-cdn.nvd.orangelogic.com/AssetLink/vd0nu1glco03o8rfy86k41y1oid7fnh6.pdf | NVIDIA RTX PRO Server 데이터시트(2026년 3월): 2U/4U/6U(수랭) 섀시, GPU 최대 8장, CX-7/CX-8/BlueField-3 |
| nvidia-rtx-blackwell-pro-gpu-architecture-v1.1.pdf (.txt) | https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf | NVIDIA RTX Blackwell PRO GPU Architecture 백서 v1.1 — GB202, Workstation/Max-Q 전체 스펙 표(Table 4, dense/sparse), MIG 구성(Table 3), FP64 = FP32의 1/64 |
| nvidia-cc-deployment-guide-tdx.pdf (.txt) | https://docs.nvidia.com/cc-deployment-guide-tdx.pdf | NVIDIA Confidential Computing Deployment Guide DU-12302-001_v7.1(2026년 4월): Blackwell 지원 조합에 RTX PRO 6000 단일 GPU 포함 |
| lenovo-lp2263-server-edition.pdf (.txt) | https://lenovopress.lenovo.com/lp2263.pdf | Lenovo Press 제품 가이드(Server Edition): FP16/FP8/TF32 수치, NVLink "No", 450 W 전력 캡, vGPU 지원(vPC/vApps/vWS), DP 기본 비활성 |
| lenovo-lp2364-maxq.pdf (.txt) | https://lenovopress.lenovo.com/lp2364.pdf | Lenovo Press 제품 가이드(Max-Q): NVLink "No", MIG 4x24GB, vGPU "No support" |
| nvidia-product-rtx-pro-6000-workstation.html | https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/ | NVIDIA Workstation Edition 제품 페이지 |
| nvidia-product-rtx-pro-6000-maxq.html | https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-max-q/ | NVIDIA Max-Q 제품 페이지 |
| nvidia-product-rtx-pro-6000-family.html | https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/ | NVIDIA RTX PRO 6000 Blackwell 시리즈 페이지 |
| nvidia-product-rtx-pro-6000-server.html | https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/ | NVIDIA Server Edition 제품 페이지(스펙 표: FP4 4 PF, FP8 2 PF, FP16 1 PF, TF32 234 TF, FP32 120 TF, 수랭 단일 슬롯 FHXL) |
| nvidia-blog-rtx-pro-6000-server-edition.html | https://blogs.nvidia.com/blog/rtx-pro-6000-blackwell-server-edition/ | NVIDIA 블로그: Server Edition 발표, L40S 대비 성능 주장, MIG 4x24GB, Confidential Computing |
| nvidia-news-rtx-pro-workstations-servers.html | https://nvidianews.nvidia.com/news/nvidia-blackwell-rtx-pro-workstations-servers-agentic-ai | NVIDIA 뉴스룸 GTC 2025 발표(출시 시기, MIG 인스턴스 수, vGPU 일정) |
| nvidia-blog-vgpu-19-blackwell.html | https://developer.nvidia.com/blog/nvidia-vgpu-19-0-enables-graphics-and-ai-virtualization-on-nvidia-blackwell-gpus/ | NVIDIA Technical Blog: vGPU 19.0 + MIG로 GPU 1장당 VM 최대 48개, L40S 대비 최대 5.6배 |
| nvidia-blog-rtx-pro-4500-vgpu-20.html | https://developer.nvidia.com/blog/scaling-the-ai-ready-data-center-with-nvidia-rtx-pro-4500-blackwell-server-edition-and-nvidia-vgpu-20/ | NVIDIA Technical Blog: vGPU 20(MIG 위 mixed-size 프로파일, 수랭 RTX PRO 6000 SE 지원, GCP G4 / Azure NCv6 분할 인스턴스) |
| nvidia-mig-user-guide-getting-started.html | https://docs.nvidia.com/datacenter/tesla/mig-user-guide/getting-started-with-mig.html | NVIDIA MIG User Guide: RTX PRO 6000 조건(R575 ≥ 575.51.03, 최소 vBIOS, DisplayModeSelector로 compute 모드 전환) |
| nvidia-displaymodeselector.html | https://developer.nvidia.com/displaymodeselector | NVIDIA Display Mode Selector 도구 페이지 |
| nvidia-vgpu-supported-gpus.html | https://docs.nvidia.com/vgpu/gpus-supported-by-vgpu.html | NVIDIA vGPU 지원 GPU 목록(RTX PRO 6000은 Server Edition 19.0+, 수랭 20.0+만 등재) |
| nvidia-cuda-gpus.html | https://developer.nvidia.com/cuda-gpus | CUDA compute capability 표(RTX PRO 6000 전 에디션 = 12.0) |
| colfax-nvfp4-sm12x-tutorial.html | https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/ | Colfax Research(서드파티, 2026년 6월): SM12x NVFP4 block-scaled GEMM 튜토리얼(mma.sync, tcgen05/TMEM 없음) |

## 링크만 (다운로드 안 함 / 실패)

| URL | 설명 / 사유 |
|---|---|
| https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf | GeForce RTX Blackwell 백서(RTX 5090 스펙 표, FP32 누산 처리량). 5090 비교에 사용함. 링크만 — RTX 5090/아키텍처 문서 담당 에이전트와 중복을 피하려고 여기엔 저장하지 않음 |
| https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf | PRO 백서 v1.0. 링크만(v1.1로 대체) |
| https://resources.nvidia.com/en-us-rtx-pro-6000 | NVIDIA 문서 허브(Server 데이터시트/브리프 링크의 출처). 링크만 |
| https://www.nvidia.com/en-us/data-center/rtx-pro-server/ | RTX PRO Servers 제품 페이지 — 이 URL은 404를 반환해서 삭제함. 데이터시트에 적힌 URL은 nvidia.com/en-us/data-center/products/rtx-pro-server. 링크만 |
| https://support.exxactcorp.com/hc/en-us/articles/35746564117271-NVIDIA-MIG-Support-How-To-Enable-on-NVIDIA-Blackwell-RTX-Pro-5000-6000-GPUs | Exxact MIG 활성화 가이드 — 403. 링크만 |
| https://forums.developer.nvidia.com/t/mig-support-on-rtx-pro-6000-blackwell/334906 | NVIDIA 개발자 포럼: 워크스테이션 MIG 이슈. 링크만 |
| https://forums.developer.nvidia.com/t/seeking-confidential-computing-configuration-guidance-for-blackwell-pro-6000-workstation-edition/346649 | 포럼: Workstation Edition CC 문의. 링크만 |
| https://forums.developer.nvidia.com/t/run-ptx-mma-sync-aligned-kind-mxf8f6f4-block-scale-scale-vec-1x-m16n8k32-on-sm-120a/329702 | 포럼: sm_120a block-scaled mma 컴파일 이슈. 링크만 |
| https://www.tomshardware.com/pc-components/gpus/nvidia-doubles-rtx-pro-6000-blackwells-msrp-to-a-staggering-usd16-000-96gb-card-started-pre-orders-below-usd8-000-last-year | Tom's Hardware(2026-08-12): 가격 이력($8,565 → $13,250 → $16,000). 링크만 |
| https://videocardz.com/newz/nvidia-raises-rtx-pro-6000-blackwell-price-to-16000-now-87-above-original-msrp | VideoCardz 가격 기사 — 403. 링크만 |
| https://www.techpowerup.com/351549/nvidia-rtx-pro-6000-blackwell-96-gb-gpu-now-costs-usd-16-000 | TechPowerUp 가격 기사 — 봇 차단. 링크만 |
| https://www.akamai.com/blog/cloud/benchmarking-nvidia-rtx-pro-6000-blackwell-akamai-cloud | Akamai 벤치마크(Server Edition, NIM/TRT-LLM, FP8/FP4 vs H100 NVL). 링크만 |
| https://www.cloudrift.ai/blog/benchmarking-rtx6000-vs-datacenter-gpus | CloudRift 벤치마크(Workstation Edition, vLLM, vs H100/H200/L40S). 링크만 |
| https://www.databasemart.com/blog/vllm-gpu-benchmark-pro6000 | DatabaseMart vLLM 벤치마크(Server Edition). 링크만 |
| https://huggingface.co/blog/HOSTKEY/nvidia-rtx-6000-blackwell-server-edition-tests-ben | HOSTKEY 테스트(Server Edition, 300 W vs 600 W 영상 생성). 링크만 |
| https://www.hpcwire.com/2025/09/10/mlperf-inference-v5-1-results-land-with-new-benchmarks-and-record-participation/ | HPCwire MLPerf Inference v5.1 기사(HPE 8x RTX PRO 6000 SE 첫 제출) — 본문 추출 실패. 링크만 |
| https://nebius.com/blog/posts/mlperf-inference-v6-0-results | Nebius MLPerf v6.0 글(RTX PRO 6000 언급, 수치 없음). 링크만 |
| https://mlcommons.org/benchmarks/inference-datacenter/ | MLCommons 공식 결과 테이블(RTX PRO 6000 제출 수치 확인용). 링크만, 미확인 |
