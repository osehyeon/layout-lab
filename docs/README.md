# NVIDIA Blackwell 자료 모음 (B200 / RTX PRO 6000 / RTX 5090)

자료 수집일: **2026-09-11**. 모든 수치는 NVIDIA 1차 자료(데이터시트, 백서, CUDA/PTX 문서)를 우선했고, 서드파티 자료는 출처와 조건을 함께 적었다.

## 계획

랩 전체의 방향은 저장소 루트의 **`PLAN_<YYYY-MM-DD>.md`** 에 있다 — 파일명의 날짜가 마지막 갱신일이다. 세 갈래(Gluon 기반 컴파일러 · F₂ 논문 스터디 · LinearLayout × 양자화 효율)와 그 의존 관계, 검증 전 가설이 거기 있다. 보유 장비 기준의 측정 항목·일정은 [09-research-plan.md](09-research-plan.md).

## 문서

| 파일 | 내용 |
|---|---|
| [00-comparison.md](00-comparison.md) | **3개 GPU 스펙·기능·SW 지원 비교표**, roofline, 선택 가이드 — 먼저 읽을 것 |
| [01-architecture.md](01-architecture.md) | Blackwell 공통 아키텍처: compute capability, **sm_100 vs sm_120**, 5세대 Tensor Core, NVFP4/MXFP, 메모리 계층, SW 스택 지원 매트릭스, 컴파일 플래그 |
| [02-b200.md](02-b200.md) | B200 / HGX·DGX B200 스펙, 시스템 구성, MLPerf, 연구 주제, B300 메모 |
| [03-rtx-pro-6000.md](03-rtx-pro-6000.md) | RTX PRO 6000 Blackwell Workstation / Max-Q / Server Edition 스펙, MIG·vGPU·CC, 5090·RTX 6000 Ada와 비교 |
| [04-rtx-5090.md](04-rtx-5090.md) | RTX 5090 스펙, GeForce 제한(FP32 누산 반속, P2P 없음), 벤치마크, 연구 주제 |
| [05-research-topics.md](05-research-topics.md) | **연구 주제 × GPU 적합도 매트릭스**, 3개 GPU 비교 연구 아이디어, 개발 흐름 |
| [06-sm-features.md](06-sm-features.md) | **SM 타깃별(sm_70 → sm_121) 신규 기능 표**, 기능 × SM 매트릭스(base / `a` / `f`), SM별 HW 한도, CUDA 13.x 드롭 타깃, 포팅 메모 |
| [07-sm120-kernel-languages.md](07-sm120-kernel-languages.md) | **sm_120(RTX 5090 / PRO 6000) 커널 작성 언어·DSL별 지원 기능 매트릭스**(CUDA C++, CUTLASS, CuTe DSL, cuTile, Triton, Gluon, TileLang 등), 용도별 추천 |
| [08-helion-triton-gluon.md](08-helion-triton-gluon.md) | Helion · Triton · Gluon의 관계: 컴파일 경로(TTIR/GLIR → TTGIR), 누가 무엇을 결정하나, sm_120에서의 의미, 선택 기준 |
| [09-research-plan.md](09-research-plan.md) | **보유 장비(RTX 5090 + RTX PRO 6000) 기준 연구 계획**: 두 카드 기능 차이·역할 분담, 단계별 실험(Phase 0~5), SW 스택, 측정·보고 규칙 |
| [10-memory-model.md](10-memory-model.md) | **메모리 계층·접근 경로**(RF / L1 / SMEM / L2 / DRAM, 명령별로 어디를 지나는가, 캐시 연산자)와 **최적화 우선순위**(파이프라이닝 → coalescing → 스필 → 벡터화 → SMEM 용량 → 뱅크 충돌 → L2), **레지스터 안의 오퍼랜드 배치와 변환**(ISA가 고정하는 fragment 비트 배정, 블록 스케일 선택자, `blocked` layout ↔ coalescing·벡터화, LinearLayout의 세 층), 항목별 `ncu` 메트릭, PTX 표기 규칙, 세대별 가용성 |
| [11-linearlayout-history.md](11-linearlayout-history.md) | **Triton LinearLayout이 왜·어떻게 만들어졌나** — 저장소 이력(커밋 245건)에서 직접 확인. 도입 PR #3794의 동기 세 가지, 그것이 링크한 실제 버그, 동기 ②의 예가 도입 직전 커밋에서 SMEM 왕복으로 처리되던 경로, `invertAndCompose`가 세 번 고쳐진 과정, Blackwell 대응이 특수 코드를 지운 증거, 그리고 "모든 layout을 하나로"라는 원래 목표가 아직 미달성이라는 사실 |
| [12-f2-primer.md](12-f2-primer.md) | **F₂ 입문 — 비트를 벡터로 보는 법.** 처음 보는 사람 기준, 표기 약속(스칼라/벡터/행렬)부터. 체(field)와 항등원·역원 → F₂ 연산표 → `mod 2`와 XOR의 차이, `%`는 창 내기 → 비트열 = 벡터 → 기저·차원 → 선형사상과 행렬 → 자리올림이 없어야 선형이라는 조건 → **스위즐**을 그 대수로 읽기(`VEC`/`PHASES` 맞바꿈). 10·11 문서를 읽기 전 배경 |
| [13-transpose-vs-inverse.md](13-transpose-vs-inverse.md) | **`Aᵀ` 와 `A⁻¹` 은 다른 연산이다.** layout 을 행렬로 볼 때 가장 섞기 쉬운 둘 — 순열 배치에서는 우연히 같고 스위즐에서 갈라진다. 각각이 layout 에서 뜻하는 것, **`C = B⁻¹A` 가 `A x = y` 를 두 번 이어 붙인 것일 뿐이라는 것**(`A`·`B` 가 각각 무엇인지, 왜 이 순서인지 `LinearLayout.h` 원문과 함께), `A⁻¹` 이 없는 경우(브로드캐스트), F₂ 가우스 소거법이 단순해지는 이유, `tl.trans` 와 행렬 전치가 다른 말이라는 것 |
| [14-gaussian-elimination.md](14-gaussian-elimination.md) | **F₂ 에서 역행렬 구하기.** `[A | I]` 를 `[I | A⁻¹]` 로 만드는 절차와 왜 그게 되는지, 손으로 따라가는 4×4 예제, 피벗이 없으면 브로드캐스트라는 것, F₂ 에서 나눗셈·부호·수치오차가 전부 사라져 행 연산이 정수 XOR 한 번이 되는 이유 |
| [15-reshape-chain-walkthrough.md](15-reshape-chain-walkthrough.md) | **(심화) 한 예제를 끝까지.** `arange(8).reshape(2,4).T.reshape(8)` 을 세 번 돌리면 layout 기저가 `[1,2,4] → [2,4,1] → [4,1,2] → [1,2,4]` 로 **제자리에 온다**. numpy 배열 기저와 layout 기저는 서로 역이라는 것(§3, 가장 틀리기 쉬운 지점), 중간 단계 전개, 대응표 ↔ 기저 양방향 계산, 회차가 곧 기저 합성이라는 것, 주기 공식 `b / gcd(r, b)`(11가지 조합 검증), GMEM 이면 복사 3번 · 레지스터면 0번, 그리고 `C = B⁻¹A` 에서 **남는 축(`register`/`lane`/`warp`)이 비용 등급을 정한다**는 것 — 축별 A·B·C 사례표와 Triton 소스(`minimalCvtLayout`, `transferWithinThread`) 인용. 11~14 문서를 읽은 뒤 |
| [16-linearlayout-talk-primer.md](16-linearlayout-talk-primer.md) | **LinearLayout 발표 슬라이드(ASPLOS'26)를 처음 배우는 사람 순서로 재구성.** layout 이 왜 골칫거리였나 → 인덱스는 비트다 → `W = L × v` 정의를 슬라이드의 16×16 예로 읽기 → 합성·곱·우역원 세 연산자 → 코드 생성(브로드캐스트, 변환 3등급, SMEM 경유, 뱅크 충돌) → 실험 수치 → CuTe 비교. 슬라이드 표기가 이 저장소 규약과 다른 곳을 §10 에 표로 모음. 논문 본문 정독(계획 B0)은 아직 |
| [17-linearlayout-blog-ita9naiwa.md](17-linearlayout-blog-ita9naiwa.md) | **한 구현자가 보는 LinearLayout** — sm_120 `tt.dot_scaled` 를 넣은 Triton 기여자(ita9naiwa)의 블로그 다섯 편 정리. 1편(= `LinearLayout.h` 머리 주석의 정의) · 2편(축 이름, `identity1D`/`zeros1D`/`operator*`/`compose`/`invertAndCompose`, minor-to-major, MMA 누산기 예) · 배경 세 편(PTX 손 코딩, PR #7918, FlashAttention). 헤더 주석과 대조해 `operator*` XOR 예제의 오기를 잡음 |

## 실습 코드 (`examples/`)

| 파일 | 내용 |
|---|---|
| [examples/f2_playground.py](examples/f2_playground.py) | F₂ 위의 선형대수를 직접 돌려 본다 — 선형성 확인, 행렬 표현, 가우스 소거법으로 역행렬, `B⁻¹ ∘ A`, 브로드캐스트는 왜 비가역인가, `stride 6`은 왜 깨지는가 (12 문서 동반). GPU 없이 실행된다 |
| [examples/reshape_transpose_reshape.py](examples/reshape_transpose_reshape.py) | `reshape → transpose → reshape` 가 만드는 lane 기저를 F₂로 추적한다. `[1,2,4,8] → [4,8,1,2]` 가 나오고, 이것이 `blocked` layout이 표현할 수 있는 `[s,2s,4s,8s]` 꼴이 아님을 확인한다 (11 문서 §1의 예). GPU 없이 실행된다 |
| [examples/reshape_chain_twice.py](examples/reshape_chain_twice.py) | 같은 연산을 **세 번** 돌려 기저가 제자리에 오는 것을 확인한다 — 회차 추적, 표 ↔ 기저 양방향이 서로 역임, 회차 = 기저 합성(`f³ = I`), 복사 횟수 대비, 주기 공식 검증, 남는 축별 비용 등급, 거리가 아니라 경계임을 이동거리로 확인 (15 문서 동반). GPU 없이 실행된다 |
| [examples/tensor_layout_playground.py](examples/tensor_layout_playground.py) | PyTorch로 **주소 함수**를 직접 확인한다. shape·stride·offset으로 주소 계산, view와 copy 구분, 16×16 타일에서 A[3,5]의 바이트 오프셋(394 B), 뱅크 충돌과 XOR 스위즐 비교. GPU 없이 CPU만으로 실행된다 |
| [examples/linear_layouts_talk_example.py](examples/linear_layouts_talk_example.py) | 16 문서의 슬라이드 예제 재현 — 16×16 blocked layout 의 표와 행렬 `L`, `W = L × v` 확인, 브로드캐스트(0 열)와 우역원, 곱 연산자, 변환 등급(intra-thread / intra-warp / intra-CTA) 판정 (GPU 불필요) |
| [examples/ll_blog_swizzle_example.py](examples/ll_blog_swizzle_example.py) | 17 문서의 블로그 예제 재현 — 4×4 스위즐 표와 `apply(3,2) = (3,1)`, GF(2) 곱, 2편의 `blocked` 기저와 `flattenIns`, `operator*` 가 덧셈(밀어 붙임)인지 XOR 인지 (GPU 불필요) |

## 원본 자료 (`sources/`)

| 폴더 | 내용 |
|---|---|
| `sources/architecture/` | Blackwell 기술 브리프, RTX/RTX PRO 백서, PTX ISA, CUDA Programming Guide, CUDA 릴리스 노트, CUTLASS/TE/cuDNN/NCCL 문서, 관련 arXiv 논문 |
| `sources/b200/` | Blackwell/HGX/DGX 데이터시트, NVIDIA 블로그, MLPerf 결과 페이지, CC 논문 |
| `sources/rtx-pro-6000/` | 에디션별 데이터시트, RTX PRO 서버 자료, CC 가이드, Lenovo 스펙 가이드, 블로그 |
| `sources/rtx-5090/` | GeForce Blackwell 백서, 제품 페이지, CUDA 12.8/호환성 문서, 서드파티 리뷰·벤치마크 |
| `sources/sm-features/` | Volta~Hopper Tuning Guide, Hopper 호환성 가이드, V100/Turing/A100/GA102/Ada/H100 백서, CUDA 13.0 릴리스 노트, CUDA 12.9 Programming Guide(구판 CC 표) |
| `sources/kernel-languages/` | `nvidia/`: CCCL·CUTLASS·CuTe DSL·cuTile·cuBLAS 문서와 소스 파일. `community/`: Triton·Gluon·TileLang·ThunderKittens·Helion·PyTorch·JAX·MAX 소스와 릴리스 노트(고정 커밋). 각 폴더의 `findings.md`에 도구별 F1~F18 근거 표 |
| `sources/linear-layouts/` | LinearLayout ASPLOS'26 발표 슬라이드(사용자 제공 PDF, 원본 URL 미확인) + `.txt` 추출. 논문 자체(arXiv 2505.23819)는 링크만 |

`figures/`에는 문서에서 참조하는 SVG 그림이 있다. 마크다운에서 `![설명](figures/이름.svg)` 로 부른다 — GitHub가 SVG를 이미지로 렌더링하므로 CSS나 `currentColor` 에 기대지 말고 색을 명시해야 하고, 밝은 배경을 직접 칠해 다크 테마에서도 읽히게 한다.

- **원본 PDF·HTML 은 저장소에 포함하지 않는다**(`.gitignore`). 벤더 문서 재배포를 피하고 clone 을 가볍게 하기 위해서다. 커밋되어 있는 것은 `pdftotext -layout` 으로 뽑은 `.txt` 와 `SOURCES.md` 뿐이고, 문서가 실제로 `grep` 하는 대상이 그 `.txt` 다. 원본이 필요하면 `SOURCES.md` 의 URL 로 같은 위치에 받으면 된다.
- 각 폴더의 `SOURCES.md`에 파일명, 원본 URL, 수집일, 설명이 있다. 다운로드하지 못한 자료는 "링크만"으로 표시했다.
- PDF는 옆에 `pdftotext -layout`으로 추출한 `.txt`가 있어 `grep`으로 찾을 수 있다. 이미지뿐인 PDF(Blackwell 기술 브리프)는 OCR 결과(`.ocr.txt`)라서 수치는 원본 PDF로 재확인해야 한다.

## 표기 규칙

- 처리량은 NVIDIA 공식 수치를 **환산 없이** 적고, 값마다 **[sparse]**(2:4 희소성 적용, dense의 2배) 또는 **[dense]** 를 붙였다. 데이터시트·제품 페이지에 공식 표기가 있으면 그 값, 없으면 백서 표의 dense 값을 썼다.
- `(미확인)`: 1차 출처로 확인하지 못한 값. `(계산값)`: 공식 수치에서 계산한 값. `2차 출처`: GitHub 이슈, 커뮤니티 자료.
- 기술 용어는 영어 그대로 쓴다 (예: Tensor Memory, block-scaled MMA).

## 핵심 요약 5줄

1. **B200 = `sm_100`, RTX PRO 6000·RTX 5090 = `sm_120`.** 둘은 서로 호환되지 않는 별개 계열이다. `tcgen05`/TMEM/2-CTA MMA는 B200 전용이다.
2. 세 GPU 모두 5세대 Tensor Core로 **NVFP4/MXFP4/MXFP8**을 지원한다. FP4 공식 수치: B200 18 PF [sparse], PRO 6000 4,000 AI TOPS [sparse], 5090 3,352 AI TOPS [sparse].
3. **RTX 5090은 FP32 누산 FP16/BF16/FP8이 반속**이다. 같은 다이의 PRO 6000은 BF16 학습 GEMM 이론치가 약 2.4배다.
4. 멀티 GPU: B200만 NVLink 5(1.8 TB/s)가 있다. PRO 6000과 5090은 PCIe뿐이고, 5090은 P2P도 없다.
5. SW 최소: CUDA 12.8 + R570 드라이버, PyTorch 2.7 cu128. 라이브러리 지원은 sm_100이 먼저이고, sm_120은 늦다(FA4·DeepGEMM은 sm_100 전용).
