# 17. 한 구현자가 보는 LinearLayout — ita9naiwa (Hyunsung Lee) 블로그 다섯 편

[16 문서](16-linearlayout-talk-primer.md)가 **설계자**(논문 저자)의 시점, [11 문서](11-linearlayout-history.md)가 **저장소 이력**의 시점이라면, 이 문서는 **구현자**의 시점이다. Hyunsung Lee (GitHub `ita9naiwa`) 는 Triton 에 sm_120 용 `tt.dot_scaled` 를 넣은 기여자이고 — 즉 이 저장소가 가진 바로 그 하드웨어(RTX 5090 · RTX PRO 6000)의 layout 코드를 쓴 사람이다 — 2025년에 LinearLayout 을 배우면서 겪은 것을 블로그에 남겼다. 다섯 편을 읽고, 그가 LinearLayout 을 **무엇이라고 보는지**, 그 시점이 어디서 왔는지, 이 저장소 규약과 어디가 같고 다른지를 정리한다.

| 날짜 | 글 | 이 문서에서 |
|---|---|---|
| 2025-01-31 | Implementing FlashAttention V1 naively | §3.3 — LinearLayout 이전의 출발점 |
| 2025-08-29 | Adding Scaled Dot Product to Triton | §3.2 — sm_120 scale layout 을 직접 짠 경험 |
| 2025-10-05 | PTX Mental Model | §3.1 — MMA 조각 배치를 손으로 겪은 기록 |
| 2025-10-27 | Linear Layout in Triton (1): Basic Idea | §1 — 그가 보는 정의 |
| 2025-10-27 | Linear Layout in Triton (2): LL Examples | §2 — 그가 쓰는 API 와 작업 방식 |

원문은 [`sources/linear-layouts/`](sources/linear-layouts/SOURCES.md) 에 `.txt` 로 있다 (CC-BY-NC-4.0). 블로그의 예제 숫자는 전부 [`examples/ll_blog_swizzle_example.py`](examples/ll_blog_swizzle_example.py) 로 재현했고, API 의미는 Triton HEAD (`570e5b4d`, 2026-09-15) 의 `include/triton/Tools/LinearLayout.h` 주석과 대조했다.

---

## 0. 한 줄로 — 그가 보는 LinearLayout

> **하드웨어 위치 (warp, lane, register) → 논리 인덱스 (row, col) 함수인데, 2의 거듭제곱 지점의 값(기저)만 적으면 나머지는 XOR 로 전부 나온다. 그래서 이름 있는 layout 여럿을 하나의 표현으로 통일(unified)하고, 합성(composable)할 수 있다.**

방향이 이 저장소의 규약(하드웨어 좌표 → 텐서 인덱스)과 같다. 슬라이드(16 문서 §10)처럼 말과 식이 어긋나는 곳이 없다 — 그의 첫 예가 `L(thread=4, warp=0) = (row=8, col=0)` 이다.

그리고 그가 강조하는 두 단어 **unified · composable** 은 각각 11 문서의 "N² → 1" 과 16 문서 §5 의 "연산이 닫혀 있다" 에 대응한다. 1편은 첫 번째를, 2편은 두 번째를 다룬다.

---

## 1. 1편 "Basic Idea" — 정의를 어떻게 소개하나

### 1.1 세 단계

| 단계 | 그가 하는 말 | 이 저장소의 대응 |
|---|---|---|
| **정의** | LL 은 하드웨어 위치 → 논리 인덱스 대응. 임의가 아니라 `L(t, 0)`, `L(0, w)` (t, w 가 2의 거듭제곱) 만 정하면 나머지가 정해진다 | 12 문서 §6 "기저와 차원" |
| **선형성 규칙** | `L(t₁ ⊕ t₂, w₁ ⊕ w₂) = L(t₁, w₁) ⊕ L(t₂, w₂)`. 이 규칙의 결과로 `L(0,0) = (0,0)` 은 기저와 무관하게 성립 | 12 문서 §7 |
| **GF(2)** | 덧셈 = XOR, 곱셈 = AND. "Linear Layout is just linear algebra over GF(2)". 열을 정수로 압축해 `[0001, 0010, 1110, 1100] × 0110 = 1100` | 12 문서 §3–§5, 그림 [`basis-as-matvec.svg`](figures/basis-as-matvec.svg) |

### 1.2 예제 — 4 스레드 × 4 워프의 스위즐

기저 넷 `L(0,1) = (0,1)`, `L(0,2) = (0,2)`, `L(1,0) = (1,1)`, `L(2,0) = (2,2)` 로 16칸을 채우면 이렇다 (스크립트 출력).

```
        w=0    w=1    w=2    w=3
t=0   (0,0)  (0,1)  (0,2)  (0,3)
t=1   (1,1)  (1,0)  (1,3)  (1,2)
t=2   (2,2)  (2,3)  (2,0)  (2,1)
t=3   (3,3)  (3,2)  (3,1)  (3,0)
```

이것이 `(t, w) → (t, w ⊕ t)` 스위즐이다. 그가 든 이유는 "SMEM 뱅크 충돌을 피하는 고전적 패턴"이고, 이 저장소에서는 12 문서 §10 과 16 문서 §6.5 가 같은 것을 다룬다.

코드 예도 이 예제다. `LinearLayout swizzled({{kThread, {{1,1},{2,2}}}, {kWarp, {{0,1},{0,2}}}}, {kDim0, kDim1})` 을 만들고 `apply({{kThread, 3}, {kWarp, 2}})` 가 `(3, 1)` 을 낸다 — thread 3 = 1 ⊕ 2 → (1,1) ⊕ (2,2) = (3,3), warp 2 → (0,2), 합쳐서 (3, 3 ⊕ 2) = (3, 1). 스크립트가 확인한다.

### 1.3 이 절은 어디서 왔나

1편 §1 의 표·규칙·계산 예 네 줄은 Triton `LinearLayout.h` **머리 주석을 거의 그대로 옮긴 것**이다 (블로그도 `LinearLayout.h:33-36` 을 인용한다). HEAD 의 주석과 대조하면 표의 물음표 위치, `L(0,0) = L(1 ⊕ 1, 0 ⊕ 0) = …` 네 줄, "basis vectors or bases" 라는 표현까지 같다. 즉 **그가 보는 LinearLayout 의 정의는 Justin Lebar 가 헤더에 적어 둔 정의**이고, 블로그는 거기에 GF(2) 절(§2)과 코드 예(§3.3)를 붙인 것이다. 이건 흠이 아니라 정보다 — 구현자들이 공유하는 "정본"이 헤더 주석이라는 뜻이고, 논문(16 문서)보다 이쪽이 실무자의 입구다.

작은 어긋남 하나. §1.2 는 `L(t, w) = (col, row)` 라고 적는데, §1.1 의 예는 `(row=8, col=0)`, 코드는 `(dim0, dim1)` 이다. 표의 값이 `(t, w⊕t)` 로 첫 성분이 t 라서 어느 쪽으로 읽어도 계산은 같지만, 좌표 순서를 인용할 때는 코드의 `(dim0, dim1)` 을 기준으로 삼는 것이 안전하다.

---

## 2. 2편 "LL Examples" — 그가 쓰는 API 와 작업 방식

2편은 "how to map hardware dimensions to layout inputs **without memorising folklore**" 를 목표로 내건다. 구현자가 실제로 겪는 순서대로 읽는다.

### 2.1 어휘 — 입력 축과 출력 축

| | 입력 축 | 뜻 |
|---|---|---|
| Distributed (레지스터) | `register` → `lane` → `warp` → `block` | 가장 잔 것부터. `lane` 은 NVIDIA 0–31, AMD 0–63 |
| Shared (SMEM) | `offset` (+ `block`) | "replace the fine-grained hardware axes with a single offset dimension" |
| 출력 | `dim0`, `dim1`, … | 항상 논리 텐서 축, 원래 순서 |

11 문서의 그림 [`rf-smem-coordinate-systems.svg`](figures/rf-smem-coordinate-systems.svg) 와 같은 구조다 — `register·lane·warp` 셋이 `offset` 하나로 접히고 `block` 은 양쪽에 있다.

### 2.2 이름 있는 layout 을 기저로 풀어 보기

그의 첫 실습은 `blocked` 를 `toLinearLayout` 으로 풀어 기저를 읽는 것이다.

```
#blocked<sizePerThread=[4,2], threadsPerWarp=[8,4], warpsPerCTA=[2,2], order=[1,0]>   shape (64,16)

register  1→(0,1)  2→(1,0)  4→(2,0)
lane      1→(0,2)  2→(0,4)  4→(4,0)  8→(8,0)  16→(16,0)
warp      1→(0,8)  2→(32,0)
```

스크립트가 이 저장소의 `blocked` 모형으로 같은 기저를 낸다. 읽는 법은 11 문서 §1 의 stride 절과 같다 — `order=[1,0]` 이라 `dim1` 부터 채우고, `register` 가 `sizePerThread`, `lane` 이 `threadsPerWarp`, `warp` 가 `warpsPerCTA` 만큼 맡는다.

같은 방법으로 `swizzled_shared<vec=8, perPhase=2, maxPhase=4>` 를 풀면 `offset=32 → (2, 8)` 처럼 **행 비트 하나가 열 비트도 뒤집는** 기저가 나온다. 그는 이것을 "SMEM 뱅크 충돌을 IR 수준에서 따질 때 쓰라"고 적는다 — 12 문서 §10 의 스위즐 정의가 실제 코드에서 어떻게 보이는지의 예다.

### 2.3 minor-to-major — 그가 가장 강조하는 규칙

> All LinearLayout operations interpret dimensions in **minor-to-major** order: the first dimension is the most minor.

`flattenIns` 는 `register → lane → warp` 순서로 이어 붙여 `register` 하나로 만들고 (스크립트: `register=8 → (0,2)`, `256 → (0,8)`, `512 → (32,0)`), 다른 순서가 필요하면 `transposeIns` 를 먼저 한다. `reshapeIns` / `reshapeOuts` 전에 확인할 것 셋 — 총 크기, 전치 계획, `getInDimSize` 로 크기 확인 — 을 체크리스트로 준다.

15 문서가 `reshape` 사슬에서 겪은 함정(기저 방향, 주기 공식)이 바로 이 규칙을 놓쳐서 생기는 것이다. 그는 "silent dimension drops are a common source of bugs" 라고 적는다.

### 2.4 "Arithmetic view" — 비트필드가 겹치지 않으면 덧셈이다

2편 §2 의 도입부가 이 문서에서 가장 쓸모 있는 한 문단이다.

> When two inputs contribute to the same output dimension via `operator*` and their basis bits **do not overlap**, the XOR accumulation is equivalent to standard integer addition over disjoint bitfields.

```
identity1D(4, lane → dim0) * strided1D(8, 4, register → dim0)
   dim0 = (lane % 4) + ((register % 8) << 2)
```

lane 이 하위 2비트, register 가 그 위 3비트를 맡으니 XOR 이 곧 덧셈이다. **겹치면 진짜 XOR** 이고 그것이 스위즐이다. 12 문서 §9 (자리올림) 와 11 문서 §1 의 "stride 가 2의 거듭제곱이면 기저의 특수한 경우" 를 구현자의 말로 한 것이다.

### 2.5 연산 — 그가 정리한 의미와 헤더 대조

| API | 그의 설명 | 헤더 주석 (HEAD) | 판정 |
|---|---|---|---|
| `identity1D(n, in, out)` | `L(x) = x` | 같음 | ✓ |
| `zeros1D(n, in, out)` | `L(x) = 0` — 브로드캐스트. `outDimSize` 로 codomain 크기만 기록 가능 | 같음 | ✓ (16 문서 §6.1 의 "0 열") |
| `strided1D` | 언급만 | 헤더에 있음 | ✓ |
| `operator*` | 입력 공간을 이어 붙이고, **왼쪽이 더 minor**. 같은 출력 축은 "XOR-accumulating" | `identity1D(4,"i","o") * identity1D(2,"i","o") == identity1D(8,"i","o")` — 같은 출력 축이면 **뒤 인자의 기저가 앞 인자 크기만큼 밀려 붙는다** | 왼쪽 minor ✓. "XOR-accumulating" 은 아래 2.6 참조 |
| `compose(outer)` | `(outer ∘ this)(x) = outer(this(x))`. this 의 출력 축 = outer 의 입력 축 | 같음 | ✓ |
| `invertAndCompose(outer)` | "Solve `this(x) = outer(C(x))` for C" — **the workhorse for layout conversions**. outer 는 전사여야 하고, 단사가 아니면 "smallest pre-image" | 헤더: `A(x) = B(C(x))`, S 는 surjective, non-injective 면 smallest offset | ✓ 이 저장소의 `C = B⁻¹A` 와 같은 식 |
| `apply` / `applyLinearLayout` | 정수로 시험 / MLIR `Value` 로 lowering 코드 생성. "dimension names 를 어긋내면 silent wrong-code" | 같음 | ✓ |

`invertAndCompose` 의 쓰임을 그는 이렇게 적는다 — *"Which shared memory offset should (register=r, lane=l, warp=w) write to?"* — `regLayout.invertAndCompose(sharedLayout)`. 11 문서 §1 의 `invertAndComposeLocal(sharedLayout, regLayout)` 과 같은 것이고, 방향도 같다 (`sharedLayout` 은 offset → 텐서 인덱스).

### 2.6 어긋난 곳 하나 — `operator*` 의 "XOR interaction" 예

2편 §3.3 예제 2 는 이렇게 적는다.

```
L1 = identity1D(4, lane → dim0),  L2 = identity1D(8, register → dim0),  L = L1 * L2
"Both contribute to dim0, so their outputs XOR:  L(lane=2, register=3) = 2 ⊕ 3 = 1"
```

그런데 **같은 구성**을 §2.1 예제 1 은 `dim0 = (lane % 4) + ((register % 8) << 2)` 로 풀고 `apply(register=1) = 4`, `apply(register=2, lane=3) = 11` 로 검증한다. 그 식으로 `(lane=2, register=3)` 은 `2 + 12 = 14` 다. 헤더 주석도 후자다 — 같은 출력 축을 겹치면 뒤 인자가 **밀려 붙어** `identity1D(8)` 이 된다. §3.3 예제 2 는 §2 의 "Arithmetic view" 를 스스로 어긴 **오기(誤記)로 보인다.** 스크립트가 14 를 낸다. `operator*` 로 진짜 XOR 겹침을 만들 수는 없고, 겹치는 기저는 §3.2 처럼 **손으로 적어야** 한다 — 그의 스위즐 예제가 그렇게 만들어졌다.

### 2.7 MMA 누산기 예제 — PTX 를 `identity1D` 사슬로

2편 §2.2 는 `m16n8k16` 누산기 조각을 이렇게 만든다.

```
ctaLayout * identity1D(kWidth, register → dimN)
          * identity1D(4,      lane     → dimN)
          * identity1D(8,      lane     → dimM)
          * identity1D(m/8,    register → dimM)
          * identity1D(n/(kWidth·4), register → dimN)
```

읽으면 이렇다 — lane 하위 2비트가 열(4 lane 한 묶음), lane 상위 3비트가 행(8행), register 가 그 사이의 열 폭과 행 반복을 맡는다. 10 문서 §0.6 이 PTX 표에서 뽑은 `lane = (행 % 8) × 4 + 열 / 2` 와 같은 배치이고, 그는 이것을 `identity1D` 곱 다섯 개로 적은 뒤 파이썬 `mapper` / `demapper` 로 PTX 표와 맞춰 본다. **"곱 = 비트필드 이어 붙이기"** 라는 그의 시각이 가장 잘 드러나는 예다.

---

## 3. 배경 세 편 — 이 시점이 어디서 왔나

### 3.1 PTX Mental Model (2025-10-05) — MMA 조각을 손으로 겪다

LinearLayout 글 3주 전에 쓴 글이다. CUDA + inline PTX 로 `m16n8k16` f16 행렬곱 하나를 처음부터 짠다.

| 단계 | 명령 | 그가 적어 둔 것 |
|---|---|---|
| GMEM → SMEM | `cp.async.cg.shared.global` 16 B | `.cg` 로 L1 우회 — 한 번만 읽는 스트림이라 |
| SMEM → RF | `ldmatrix.sync.aligned.m8n8.x4.trans` (A), `.x2.trans` (B) | lane 을 `quad = lane >> 3`, `row = lane & 7` 로 쪼개 주소를 만든다 |
| MMA | `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` | 누산기는 lane 마다 2×2 |
| RF → GMEM | scatter | `quad = lane / 4`, `col = (lane % 4) × 2`, 행 `r0` 와 `r0 + 8` |

마지막 줄이 핵심이다. **누산기 조각의 "누가 무엇을 갖는가"를 정수 산술로 손으로 적어 본 경험**이, 3주 뒤 2편 §2.2 에서 같은 것을 `identity1D` 곱으로 다시 적는 동기다. 이 저장소에서는 10 문서 §0.6 (비트 배정표) 과 16 문서 §3 (행렬 `L`) 이 같은 조각을 두 방식으로 적는다.

### 3.2 Adding Scaled Dot Product to Triton (2025-08-29) — sm_120 scale layout 을 짜다

그의 첫 Triton PR. 저장소 이력으로 확인하면 이렇다.

| 날짜 | 커밋 | 내용 |
|---|---|---|
| 2025-08-30 | `001ec4b2` | "[NVIDIA] Add native MXFP FP8 scaled_dot for SM120 (#7918)" 병합 |
| 2025-09-02 | `6ec5e0c8` | 되돌림 (#8029) |
| 2025-09-09 | `27f406c6` | 다시 적용 (#8129) — 12 파일, +846/−140 |

블로그는 "merged" 만 적었고 되돌림·재적용은 이력에서만 보인다. 바뀐 파일에 `LinearLayoutConversions.{h,cpp}` 와 `LinearLayoutConversionsTest.cpp` 가 있고, 그가 든 함수 `chooseScaledNvidiaScaleLayout` 은 HEAD 의 `LinearLayoutConversions.h` 에 있다.

글의 요점은 하나다 — *"The tricky part: scale tensor layout mismatch."* `mma.sync.…block_scale.scale_vec::1X` 는 scale 을 특정 lane 이 공급해야 하고, Triton 의 scale 텐서 배치를 그 lane 들에 정확히 맞춰야 했다. 그 "맞추는 코드"가 LinearLayout 으로 scale 텐서의 배치를 **고르는** 함수다. 즉 그는 LinearLayout 을 배우기 전에 이미 LinearLayout 으로 배치를 고르는 코드를 짰고, 두 달 뒤에 그 도구를 설명하는 글을 썼다.

이 저장소와의 접점은 둘이다. 첫째, 이것이 우리 하드웨어다 — 10 문서 §0.6 의 block-scale `{byte-id, thread-id}` 선택자가 바로 이 PR 이 맞춰야 했던 배치다. 둘째, 11 문서 §6-2 의 #11262 (2026-08) "Support batched SM120 scaled-dot scale layouts" 는 이 PR 이 rank-2 로 고정해 둔 것을 일반화한 후속이다.

그가 적은 벤치마크 (RTX 5090, vLLM Llama3-8B-Instruct, `2차 출처` · 우리 측정 아님):

| 경로 | 시간 |
|---|---|
| fp8 × fp8 | 42.83 s |
| mxfp8 × mxfp8, native `dot_scaled` | 44.45 s |
| mxfp8 × mxfp8, 에뮬레이션 (main) | 76.44 s |

### 3.3 FlashAttention V1 naively (2025-01-31) — 출발점

가장 이른 글이고 스스로 "my limitation of skills" 라고 적는다. 16×16 블록, `tx = threadIdx.x / 16`, `ty = threadIdx.x % 16` 로 스레드를 나눠 `__shfl_down_sync` 로 행 최대값을 줄인다. 배치를 고르는 결정이 전부 손으로, 정수 산술로 이뤄지고, 그 제약("COL_BLOCK_SIZE 를 32 로 못 늘려 `__shfl_xor_sync` 가 어려웠다")이 그대로 성능 한계가 된다. 9개월 뒤의 그가 LinearLayout 을 "folklore 를 외우지 않고 배치를 다루는 법"이라고 소개하는 이유가 여기 있다.

---

## 4. 이 저장소 규약과 대조

| 항목 | 블로그 | 이 저장소 | 판정 |
|---|---|---|---|
| layout 의 방향 | 하드웨어 위치 → 논리 인덱스 | 같음 | ✓ (슬라이드와 달리 말과 식이 일치) |
| 변환 공식 | `this(x) = outer(C(x))` 를 C 에 대해 풀기 | `C = A.invertAndCompose(B)` 는 `B⁻¹ ∘ A` | ✓ 같은 식 |
| `sharedLayout` 의 방향 | offset → (dim0, dim1) | 같음 | ✓ (슬라이드의 `L_S` 는 반대였다) |
| 브로드캐스트 | `zeros1D`, "all basis vectors are zero" | 0 열 (16 문서 §6.1), 피벗 없음 (14 문서 §5) | ✓ |
| `operator*` 의 같은 출력 축 | §2.1 "덧셈" / §3.3 "XOR" — **서로 모순** | 헤더: 밀려 붙음 (덧셈 쪽) | §3.3 예제 2 는 오기로 보임 |
| 좌표 순서 표기 | `(col, row)` / `(row, col)` / `(dim0, dim1)` 혼용 | `(dim0, dim1)` | 코드 기준으로 읽을 것 |
| minor-to-major | 모든 연산의 기본 규칙으로 강조 | 15 문서의 함정과 같은 뿌리 | ✓ 보탬 |
| PR #7918 | "merged" | 병합 → 되돌림 → 재적용 (#8129) | 이력이 보탬 |
| 성능 수치 | RTX 5090 vLLM 3개 | 미측정 | `2차 출처` 로만 인용 |

---

## 5. 세 시점을 어떻게 겹쳐 읽나

| 시점 | 문서 | 답하는 질문 |
|---|---|---|
| 설계자 | [16](16-linearlayout-talk-primer.md) (슬라이드) | **왜** F₂ 인가, 비용 등급이 어떻게 나오나 |
| 이력 | [11](11-linearlayout-history.md) | **언제 무엇이** 바뀌었나, 무엇이 아직 안 됐나 |
| 구현자 | 이 문서 (블로그) | **어떻게 쓰나** — 어휘, 생성자, 합성, 검증 습관 |

처음 배우는 사람에게 권하는 순서는 블로그 1편(= 헤더 주석) → 12 문서 → 16 문서 §3–§5 → 블로그 2편 → 11 문서다. 블로그 2편의 검증 습관 — `apply` 로 몇 점 찍어 보기, `flattenIns` 전에 `transposeIns` — 은 이 저장소가 "예제 숫자는 전부 스크립트로 검증한다"고 정한 것과 같은 규칙이다.

---

## 6. 확인한 것과 남긴 것

- 1편의 표·`apply(3,2) = (3,1)`·GF(2) 곱, 2편의 `blocked` 기저·`flattenIns`·§2.1 검증값은 스크립트로 재현했다. 2편 §3.3 예제 2 는 재현되지 않고 헤더 주석과도 어긋난다.
- `operator*`·`compose`·`invertAndCompose`·`zeros1D`·`strided1D` 의 의미는 HEAD `LinearLayout.h` 주석과 대조했다. 2편의 swizzled shared 출력(`vec=8, perPhase=2, maxPhase=4`)과 `compose` 예제 출력은 블로그 값을 그대로 적었고 따로 계산하지 않았다.
- PR #7918 의 병합·되돌림·재적용은 커밋 로그에서 확인했다. `chooseScaledNvidiaScaleLayout` 이 HEAD 헤더에 있는 것도 확인했다.
- 블로그가 예고한 "Part 3" 이후는 2026-09-16 시점에 확인하지 않았다.
