# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A Korean-language research knowledge base on NVIDIA Blackwell GPUs — **B200, RTX PRO 6000 Blackwell, GeForce RTX 5090** — covering specs, supported features, and research directions. No code or build system yet; everything lives in `docs/`. Start from `docs/README.md` (index) and `docs/00-comparison.md` (cross-GPU tables).

## 계획 문서

**파일명은 전부 영어로 짓는다** (내용은 한국어). 랩 전체의 방향은 저장소 루트의 `PLAN_<YYYY-MM-DD>.md` 하나뿐이다. **파일명의 날짜가 마지막 갱신일**이므로, 내용을 고치면 파일명도 같이 바꾸고(`mv`) 문서 맨 아래 **갱신 이력**에 한 줄 남긴다. 계획 파일이 둘 이상 생기지 않게 할 것. 거기 적힌 H1·H2 는 **검증 전 가설**이니 사실처럼 인용하지 말 것. 장비 기준 측정 항목은 `docs/09-research-plan.md` 에 있고 계획 문서에서 중복하지 않는다.

## Layout

- `docs/00-comparison.md` and `docs/05-research-topics.md` are **syntheses** of the per-topic docs (`01-architecture`, `02-b200`, `03-rtx-pro-6000`, `04-rtx-5090`). `docs/06-sm-features.md` is the per-SM-target (sm_70 → sm_121) feature reference; keep its B300/sm_120 facts consistent with `01-architecture.md` §2. `docs/07-sm120-kernel-languages.md` is the sm_120 kernel-language/DSL support matrix; its per-tool evidence (feature rows F1–F18) lives in `docs/sources/kernel-languages/{nvidia,community}/findings.md`, and tool versions/commits there are pinned — re-check before updating since these projects move fast. `docs/08-helion-triton-gluon.md` explains how Helion (generates Triton), Triton, and Gluon (enters the same stack at TTGIR) relate. `docs/09-research-plan.md` is the user's research plan for the hardware they actually own (one RTX 5090 + one RTX PRO 6000, both sm_120; no B200) — when Phase 0 measurements resolve a `(미확인)` item, update the affected doc (00/03/04/06/07) too. `docs/10-memory-model.md` covers the memory hierarchy and per-instruction access paths (§0: which units `ld.shared` / `ld.global` / `cp.async` / `ld.const` traverse, PTX cache operators, L1-vs-SMEM as two paths through the same SRAM) plus the memory-optimization priority list (pipelining → coalescing → spill → vectorization → SMEM capacity → bank conflicts → L2) with per-item `ncu` metrics; §0.6 covers what lives *inside* the register file — the per-operand fragment bit assignments the ISA fixes (per-instruction lane maps for `m16n8k16`/`k32`/`k64`: accumulator and A agree only at f16 — the accumulator formula never changes across the three while A/B widen from 2 to 4 to 8 columns per lane — and B is always their transpose), the block-scale `{byte-id, thread-id}` selectors, Triton `blocked` layout as the same numbers as coalescing (`order`) and vectorization (`sizePerThread`), and the three layers LinearLayout is often confused across (the `ttg.convert_layout` op stays; only its lowering went from per-pair to one). Its impact multipliers and the LinearLayout performance claim are rough or derived, not measurements — replace them when Phase 0 runs, and keep its sm_120 limits consistent with `06-sm-features.md`. `docs/11-linearlayout-history.md` (저장소 이력), `docs/12-f2-primer.md` (F₂ 입문), `docs/13-transpose-vs-inverse.md` (`Aᵀ` vs `A⁻¹`), `docs/14-gaussian-elimination.md` (`[A|I] → [I|A⁻¹]`), `docs/15-reshape-chain-walkthrough.md` (심화 예제) 는 하나의 묶음이다 — 규약과 함정은 아래 **LinearLayout 을 다룰 때** 절에 모아 두었으니 이 다섯 문서를 건드리기 전에 그쪽을 먼저 읽을 것. `docs/16-linearlayout-talk-primer.md` 는 ASPLOS'26 발표 슬라이드(`docs/sources/linear-layouts/`)를 초심자 순서로 재구성한 것이다 — 논문 본문 정독(계획 B0)은 아직이고, 슬라이드가 `L_S` 를 coordinate → offset 방향으로 적는 등 이 저장소 규약과 방향이 다른 곳은 그 문서 §10 표에 모아 두었으니 인용할 때 방향을 먼저 맞출 것. `docs/17-linearlayout-blog-ita9naiwa.md` 는 sm_120 `dot_scaled` 기여자의 블로그 다섯 편(`docs/sources/linear-layouts/ita9naiwa-*`)을 구현자 시점으로 정리한 것 — 그 블로그 2편 §3.3 의 `operator*` "XOR interaction" 예제는 헤더 주석과 어긋나는 오기이니 인용하지 말 것 (같은 출력 축은 밀려 붙는다, `identity1D(4)*identity1D(2) == identity1D(8)`). Figures live in `docs/figures/*.svg`, referenced as `![…](figures/x.svg)`; GitHub renders them as images, so specify every color explicitly (no CSS, no `currentColor`), paint a light background, and **render the SVG and look at it before committing** — a coordinate check alone will not catch malformed XML or clipped text. When a number or fact changes in a per-topic doc, update the syntheses and the summary in `docs/README.md` too.
- `docs/sources/<topic>/` holds downloaded primary sources (PDF/HTML) plus `pdftotext -layout` extracts (`.txt`) for grepping. Each folder has a `SOURCES.md` (filename, original URL, retrieval date, description; undownloadable items marked "링크만"). Add a row there whenever a source is added.
- **원본 PDF·HTML 은 `.gitignore` 로 제외되어 있다.** 커밋된 것은 `.txt` 추출본과 `SOURCES.md` 뿐이다 — 클론 직후에는 PDF 가 없으니, 원본 대조가 필요하면 `SOURCES.md` 의 URL 로 먼저 받을 것. 새로 받은 PDF 를 `git add` 하지 말 것.
- The Blackwell architecture tech brief is image-only; its `.ocr.txt` may contain OCR errors — verify numbers against the PDF.

## Writing conventions (keep consistent across docs)

- Korean prose, technical terms in English.
- Throughput: one value per cell — the official NVIDIA figure **as published, no conversion** — tagged `[sparse]` or `[dense]` (user preference; no separate dense/sparse columns, the reader halves sparse values themselves). Prefer the datasheet/product-page headline; if none, use the whitepaper's dense value. Mixed bases to watch: B200 datasheet tensor figures are all sparse; RTX PRO 6000 Server "234 TFLOPS TF32" is dense while its FP4/FP8/FP16 are sparse; B300 NVFP4 15 PF [dense] / 20 PF [sparse] is not 2x.
- Mark values not confirmed by a primary source `(미확인)`, derived values `(계산값)`, community/GitHub-issue sources `2차 출처`. Cite the original URL inline, not the local file.
- Use **shipping** HGX B200 specs (180 GB, 7.7 TB/s, 1,000 W, FP4 18 PF [sparse]), not the March-2024 pre-announcement (192 GB, 8 TB/s) or GB200-class figures (FP4 20 PF [sparse], 1,200 W).

## LinearLayout 을 다룰 때 (docs 11–15, `figures/`, `examples/`)

이 세션에서 실제로 틀렸다가 1차 출처로 잡은 것들이다. **되돌리지 말 것.**

### 규약 — 한 번 뒤집히면 전부 뒤집힌다

- **layout = 하드웨어 좌표 → 텐서 인덱스.** 기저는 항상 이 방향으로 읽는다. 표를 인용할 때 **왼쪽이 좌표인지 인덱스인지 먼저 밝힐 것.**
- **`C = A.invertAndCompose(B)` 는 `C = B⁻¹ ∘ A` 다.** `A` = 값이 지금 놓인 배치(출발, 헤더의 `R`=레지스터), `B` = 소비자가 요구하는 배치(도착, `S`=SMEM). 근거는 `include/triton/Tools/LinearLayout.h`:
  ```
  // if C = A.invertAndCompose(B), then ... A(x) = B(C(x)).  If B is invertible,
  // then C(x) = B^-1(A(x)), which is how this function gets its name.
  ```
  문서 전체가 한때 `B ∘ A⁻¹` 로 잘못 적혀 있었고 전부 고쳤다. `B ∘ A⁻¹` 은 인덱스 → 인덱스가 되어 말이 안 된다.
- **numpy 배열은 layout 의 역행렬이다.** `np.arange(8).reshape(2,4).T.reshape(8)` → `[0,4,1,5,2,6,3,7]` (배열 기저 `[4,1,2]`, 새 자리 → 옛 원소) 인데 **layout 은 `[0,2,4,6,1,3,5,7]` = 기저 `[2,4,1]`** (lane → 새 인덱스). 12 문서 §11 과 `examples/reshape_transpose_reshape.py` 가 후자를 쓴다. **배열 출력을 그대로 기저라고 적지 말 것.**
- 15 문서의 사이클은 layout 기준 **`[1,2,4] → [2,4,1] → [4,1,2] → [1,2,4]`** 다.
- 이 사슬은 `f³ = I` 라 **`f⁻¹ = f²`** — 1회차를 거꾸로 읽으면 2회차 layout 이 그대로 나온다. 그래서 방향을 틀려도 그럴듯해 보인다.
- `Aᵀ = A⁻¹` 은 **순열 배치에서만** 성립한다. 문서의 예가 거의 전부 순열이라 차이가 숨는다(13 문서).
- 주기 공식은 `b / gcd(r, b)` (`b = log2(원소 수)`, `r = log2(행 수)`). **`(4,4)`·16원소는 주기 2** 이므로 "주기 = 비트 수"라고 쓰면 틀린다.

### 비용 등급 — 거리가 아니라 경계다

- Triton 은 `B⁻¹A` 를 그대로 보지 않고 `minimalCvtLayout` 으로 **이미 항등인 축을 느린 쪽부터 quotient** 한다. 남는 축이 등급을 정한다:

  | 남는 축 | 나가는 것 | 근거 함수 |
  |---|---|---|
  | 없음 | 명령 0개 (`replaceOp(op, src)`) | `ConvertLayoutOpToLLVM.cpp` |
  | `register` 만 | SSA 값 재배열 — **공짜** | `transferWithinThread` |
  | `lane` | `shfl` | `cvtNeedsWarpShuffle` |
  | `warp` / `block` | SMEM / DSMEM 왕복 + 배리어 | `cvtNeedsSharedMemory` |

  → **`C ≠ I` 인데 비용 0 인 경우가 실제로 있다** (`register` 만 남을 때). "항등이어야만 공짜"가 아니다.
- 판정은 항등 기저 `[1,2,4,8,16,32]` 를 기준자로 읽는다 — **자리 = 출발 축, 값 = 도착 축.** 자리 `j` 의 기저값이 `j` 와 다른 칸 출신이면 그 경계를 넘은 것이다.
- **거리는 무관하다.** lane 비트 전부 뒤집기 = 최대 이동 21인데 `shfl`, lane↔warp 한 비트 = 최대 이동 16인데 SMEM 왕복 (`examples/reshape_chain_twice.py` §7 에서 확인).
- 워프 경계가 결정적인 이유: **다른 워프의 레지스터를 읽는 명령이 ISA 에 없다.**
- 칸 폭 중 **`lane` 만 하드웨어가 5비트로 고정**한다. `register`·`warp` 폭은 컴파일러·런치 설정이 정하므로 **`num_warps` 를 바꾸면 같은 변환의 등급이 바뀔 수 있다.**
- `minimalCvtLayout` 은 `dstLayout.invertAndCompose(srcLayout)` = **`src⁻¹ ∘ dst`** (gather 방향) 를 쓴다. 문서의 `B⁻¹A` (scatter) 와 서로 역이고 내용은 같다 — 소스를 인용할 때 방향을 섞지 말 것.

### 레지스터 수치 (PG 13.4 Table 31)

- 레지스터 1개 = **32-bit = 4 B**. 스레드당 최대 **255개 = 1,020 B**. SM당 **64 K개 = 256 KB**.
- 상주 워프 ≈ `65,536 / (스레드당 레지스터 × 32)` → 255개를 다 쓰면 워프 8개뿐.
- `.f16x2` 는 1칸에 2개, `.f64` 는 2칸을 먹는다. `mma.sync.m16n8k16` f16 하나가 lane 당 A 4 + B 2 + acc 4 = 10칸.

### 도입 전에는 어땠나

- 변환 코드는 **배치 쌍마다** 있었다 (lane 마다가 아니다). N개면 `C(N,2)` 가지 — 3→3, 6→15, 9→36. 새 배치 하나 추가 = **+(N−1) 개 함수**. 지금은 **+1 개 기저 목록**.
- 다 만들 수 없어서 신경 쓴 조합만 `ldmatrix`·벡터화·`shfl` 을 쓰고 나머지는 SMEM 왕복 fallback 이었다 — 이 대비는 `(계산값 · 미측정)` 이므로 측정치처럼 쓰지 말 것.
- **`ttg.convert_layout` op 자체는 지금도 있다.** 없어진 것은 op 이 아니라 쌍마다 손으로 내리던 lowering 이다.

### 작업 방식

- **Triton 에 대한 주장은 클론해서 확인한다** (논문이나 기억이 아니라). 클론은 scratchpad 에 두므로 세션이 끝나면 사라진다 — 필요하면 다시 받을 것. 검증에 쓴 함수 이름: `minimalCvtLayout`, `cvtReordersRegisters`, `cvtNeedsWarpShuffle`, `cvtNeedsSharedMemory`, `transferWithinThread`, `invertAndCompose`.
- PTX 주장은 `docs/sources/architecture/ptx-isa.txt` 에서 절 번호까지 확인한다.
- **예제 숫자는 전부 스크립트로 검증하고 그 스크립트를 `docs/examples/` 에 남긴다.** 이 세션에서 잡힌 오류(합성 순서, 기저 방향, 주기 공식)는 전부 실행해 보고 나서야 드러났다.
- 표기: 스칼라 = 소문자 이탤릭, 벡터 = 소문자 볼드, 행렬 = 대문자 볼드. **정렬된 코드 블록 안에 한글을 넣지 않는다** (칸이 어긋난다) — 설명은 블록 밖이나 표로 뺀다.

## Domain facts that are easy to get wrong

- `sm_100` (B200) and `sm_120` (RTX PRO 6000 / RTX 5090) are separate families: `tcgen05`/TMEM/2-CTA MMA exist only on sm_10x/sm_11x; sm_120 uses warp-level `mma.sync` with block-scale extensions and 99 KB SMEM/block. TMA multicast is PTX-legal on sm_120 but not on NVIDIA's advised targets, and CUTLASS fixes GeForce clusters to 1x1x1 ("no multicast").
- B300 (CC 10.3) is not a superset of B200: no tcgen05 `kind::i8` (PTX) and no FP64 Tensor Core, FP32:FP64 64:1 (Programming Guide 13.4 Table 33/30). `wgmma` exists only on sm_90a. Kernels/libraries for one don't run on the other.
- RTX 5090 runs FP16/BF16/FP8 with FP32 accumulate at half rate (and TF32 at half per SM·clock) vs RTX PRO 6000 on the same GB202 die; FP4 and INT8 are not halved.
- RTX 5090 has no NVLink, no MIG, no PCIe P2P; RTX PRO 6000 has MIG (up to 4) but no NVLink; vGPU and Confidential Computing are Server Edition only.
