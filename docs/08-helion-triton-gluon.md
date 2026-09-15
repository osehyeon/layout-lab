# 08. Helion · Triton · Gluon의 관계

> 조사일: 2026-09-11. 기준 버전: **Triton 3.8.0**(main `66aa2f8`), **Helion 1.4.0**(`cfb135e`), PyTorch 2.14.0. 원본은 [`sources/kernel-languages/community/`](sources/kernel-languages/community/SOURCES.md)에 있다.
> sm_120(RTX 5090 / PRO 6000)에서의 도구별 지원 범위는 [07-sm120-kernel-languages.md](07-sm120-kernel-languages.md), 상세 근거 표는 [community/findings.md](sources/kernel-languages/community/findings.md) §1·§2·§6을 참조한다.

## 0. 한 줄 요약

**셋은 같은 Triton 컴파일러 스택 위에서 추상화 높이만 다르다.**
- **Helion**: Triton 코드를 *생성*한다.
- **Triton**: tile 수준 언어이며, layout·파이프라인·비동기 처리를 컴파일러가 정한다.
- **Gluon**: 같은 스택에 *더 낮은 층*으로 들어가서, Triton 컴파일러가 대신 정하던 것을 사용자가 직접 정한다.

```
            추상화 높음 ────────────────────────────────▶ 낮음
  ┌───────────────┐   생성    ┌───────────────┐          ┌───────────────┐
  │   Helion      │ ───────▶ │   Triton      │          │   Gluon       │
  │ PyTorch + 타일 │          │ @triton.jit   │          │ @gluon.jit    │
  │ + autotune    │          │ (tl.*)        │          │ (gl.*)        │
  └───────────────┘          └──────┬────────┘          └──────┬────────┘
                                    │ TTIR                     │ GLIR
                                    ▼                          ▼
                          make_ttgir (자동 최적화 패스)   gluon_to_ttgir (최소 패스)
                                    └────────────┬─────────────┘
                                                 ▼ TTGIR (공통)
                                    make_llir → make_ptx → make_cubin (공통 백엔드)
```

## 1. 공식 정의 (원문)

| 도구 | 공식 설명 | 출처 |
|---|---|---|
| **Helion** | "a Python-embedded domain-specific language (DSL) for authoring machine learning kernels, designed to compile down to Triton … Helion aims to raise the level of abstraction compared to Triton." / "Helion can be viewed either as *PyTorch with tiles* or as *a higher-level Triton*." | `helion-README.md` L13-24 |
| **Gluon** | "Gluon is a GPU programming language based on the same compiler stack as Triton. But unlike Triton, Gluon is a lower-level language that gives the user more control and responsibility when implementing kernels." | `gluon-tutorial-01-intro.py` |
| **Triton (Gluon 문서가 본 관점)** | Triton은 "defers to the compiler to manage tile layouts, memory allocation, data movement, and asynchronity." 컴파일러가 대부분 잘하지만 "it can be beaten by hand-tuned low-level code. When this happens, there is little the user can do". | 같은 파일 |

- Gluon과 Triton은 "tile 기반 SPMD 모델"과 "같은 frontend·JIT 인프라"를 공유한다. Gluon 커널도 Triton 커널과 같은 방식(PyTorch 텐서 인자, grid 지정)으로 실행한다(`gluon-tutorial-01-intro.py`).
- Gluon은 `triton.experimental.gluon`에 있다. API가 아직 experimental이며, 3.8.0에서도 breaking change가 있었다.

## 2. 컴파일 경로: 어디서 갈라지고 어디서 합쳐지나

Triton NVIDIA 백엔드(`third_party/nvidia/backend/compiler.py`)의 `add_stages`는 언어별로 앞단만 다르다.

```python
if language == Language.TRITON:
    stages["ttir"]  = make_ttir
    stages["ttgir"] = make_ttgir          # 자동 최적화 패스 다수
elif language == Language.GLUON:
    stages["glir"]  = (그대로)
    stages["ttgir"] = gluon_to_ttgir      # 최소 패스
stages["llir"] → stages["ptx"] → stages["cubin"]   # 공통
```

**차이의 핵심은 `make_ttgir`가 하는 일을 Gluon에서는 누가 하느냐**다.

| `make_ttgir`의 자동 패스 (Triton) | 하는 일 | Gluon에서는 |
|---|---|---|
| `add_convert_to_ttgpuir`, `add_coalesce`, `add_remove_layout_conversions` | 텐서마다 layout(스레드↔원소 매핑)을 정하고 불필요한 변환을 제거 | **사용자가 layout 명시**(`gl.BlockedLayout` 등). `gluon_to_ttgir`는 `infer_coalesced_encodings`/`resolve_auto_encodings`로 "auto"로 남긴 것만 채운다 |
| `add_accelerate_matmul` | `tl.dot`에 쓸 MMA 버전·명령 선택(sm_120 → MMAv2) | **사용자가 명령 선택**(`mma_v2`, `wgmma`, `tcgen05_mma` 중 하나를 직접 호출) |
| `add_assign_latencies`, `add_schedule_loops`, `add_pipeline` | `num_stages`에 따른 소프트웨어 파이프라이닝 | **사용자가 multi-buffer·mbarrier로 직접 구성** |
| `add_warp_specialize`(cc ≥ 10), `add_hopper_warpspec`(cc 8·9) | 자동 warp specialization | **사용자가 `gl.warp_specialize`로 역할 분할** |
| `add_hoist_tmem_alloc`, `add_promote_lhs_to_tmem` 등 | TMEM 할당·배치(sm_100) | 사용자가 `allocate_tensor_memory` 등으로 명시 |
| `add_tma_lowering`(cc ≥ 9) | TMA descriptor 연산 lowering | **공통**(Gluon 경로에도 있음) |

> 근거: `triton-nvidia-backend-compiler.py`의 `make_ttgir`(L299~)와 `gluon_to_ttgir` 본문. 참고로 sm_120(capability 120)은 `capability // 10 >= 10` 분기를 타므로 Triton의 Blackwell용 패스 목록이 적용된다. 다만 MMA는 MMAv2가 선택되고(`AccelerateMatmul.cpp`), TMEM 관련 패스는 할 일이 없다.

## 3. Helion은 Triton 위에서 무엇을 자동화하나

```python
import torch, helion, helion.language as hl

@helion.kernel()
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return out
```
(`helion-README.md` L70-86 그대로)

- `for` 루프 **밖**은 CPU에서 도는 평범한 PyTorch 코드다(출력 할당, shape 계산).
- `for` 루프 **안**은 **Triton 커널 하나로 컴파일**된다. Helion 커널 하나는 항상 GPU 커널 하나다.
- 루프 안의 PyTorch 연산(`torch.addmm` 등)은 **TorchInductor**를 통해 Triton 연산으로 매핑된다.
- 바깥 `hl.tile` 루프는 grid로, 안쪽 루프는 커널 내부 루프로 바뀐다. tile 크기와 grid는 autotuner가 정한다.

**Helion autotuner가 탐색하는 Triton 수준 선택지**(README와 `helion.Config` 예시):
- `block_sizes`, `loop_orders`, `l2_groupings`, PID → tile 매핑(`pid_type`)
- `indexing`: pointer / block pointer / **TensorDescriptor(TMA)**
- `num_warps`, `num_stages`, `range_num_stages`, `range_multi_buffers`
- **`range_warp_specializes`** → Triton의 `warp_specialize`
- 한 번 탐색에 수백~천여 개 config를 평가한다(README 예시: 1,520개, 약 586초). 운영 환경에서는 미리 튜닝한 config를 고정하도록 권장한다.

**백엔드**: 기본은 Triton이다. 추가로 Triton-TileIR 백엔드("compute capability 10.x/12.x")와 실험적 CuTe DSL 백엔드가 있다(`helion-README.md`, `helion-_compat.py`). **README에 Gluon 백엔드 언급은 없다.**

## 4. 무엇을 누가 결정하나 (비교표)

| 결정 사항 | Helion | Triton | Gluon |
|---|---|---|---|
| 커널 작성 단위 | PyTorch 연산 + `hl.tile` 루프 | 프로그램(블록)당 tile 연산 `tl.*` | 프로그램당 tile 연산 `gl.*` + 하드웨어 명령 |
| tile 크기·루프 순서·grid | **autotuner** | 사용자(`constexpr`) + `@triton.autotune` | 사용자 |
| 메모리 접근 방식(pointer/TMA) | **autotuner**(`indexing`) | 사용자(`tl.make_tensor_descriptor` 등) | 사용자(`tma` 모듈) |
| 텐서 layout | Triton 컴파일러 | **컴파일러** | **사용자**(`BlockedLayout`, `NVMMASharedLayout` …) |
| MMA 명령 | Triton 컴파일러 | **컴파일러**(`accelerate_matmul`) | **사용자**(`mma_v2` / `wgmma` / `tcgen05_mma`) |
| 파이프라이닝 | autotuner가 `num_stages` → 컴파일러 | 컴파일러(`num_stages` 힌트) | **사용자**(버퍼·mbarrier 직접) |
| warp specialization | autotuner(`range_warp_specializes`) → 컴파일러 | 컴파일러(`tl.range(warp_specialize=True)`) | **사용자**(`gl.warp_specialize`, 레지스터 수 지정) |
| 인라인 어셈블리 | `hl.inline_asm_elementwise` | `tl.inline_asm_elementwise` | `inline_asm_elementwise`(Gluon language `_core.py`에 정의. `gl.` 경로로 쓰는 예제는 미확인) |
| 성능 상한 | Triton 컴파일러의 상한 | 컴파일러 상한 | 손으로 짠 커널 수준(단, 노력이 큼) |
| 필요한 지식 | PyTorch | tile 프로그래밍 | GPU 하드웨어(layout, 비동기, 동기화) |

Gluon의 layout 예시(`gluon-tutorial-02-layouts.py`):
```python
gl.BlockedLayout(
    size_per_thread=[2, 4],
    threads_per_warp=[16, 2],
    warps_per_cta=[2, 2],
    order=[1, 0],
)   # block shape = [64, 16], 각 스레드가 2×4 부분 tile을 레지스터에 가짐
```

## 5. sm_120(RTX 5090 / PRO 6000)에서의 의미

| | Helion | Triton 3.8 | Gluon |
|---|---|---|---|
| MMA | Triton 상속(MMAv2) | MMAv2 `mma.sync` 고정("Exclude consumer Blackwell (sm120)") | `mma_v2`만 쓸 수 있음(`wgmma`·`tcgen05`는 sm_120에 없음) |
| FP4/MXFP8 block-scaled | `hl.dot_scaled` → Triton 제약 상속 | `tl.dot_scaled` native(3.6.0+, 같은 포맷끼리만) | **API 없음**(`tcgen05_mma_scaled`는 sm_100 전용) |
| TMA | TensorDescriptor 인덱싱(cc ≥ 9) | O(scatter 제외) | O(`hopper.tma`) |
| warp specialization | 설정 가능하나 sm_120 미검증 | 패스는 돌지만 테스트 대상 아님(#10284 open) | API O(`gl.warp_specialize`). sm_120 CI는 미확인 |
| cluster | X | X(`supportClusterOps()`가 12.x 제외) | X(같은 게이트) |
| 추가 백엔드 | TileIR(12.x 명시 지원) | – | – |

정리하면 다음과 같다.
- **Helion과 Triton은 sm_120에서 기능 범위가 사실상 같다.** Helion의 이점은 autotune이 99 KB SMEM을 넘는 config를 걸러 준다는 점이다.
- **Gluon은 sm_120에서 "Ampere/Ada식 `mma_v2` + TMA + mbarrier + warp specialization"을 명시적으로 짜는 용도로는 쓸 수 있다.** 다만 FP4 Tensor Core는 쓸 수 없다.

## 6. 언제 무엇을 쓰나

| 상황 | 선택 | 이유 |
|---|---|---|
| PyTorch 연산자를 빠르게 fused 커널로 만들고 튜닝은 기계에 맡기고 싶다 | **Helion** | PyTorch 문법 그대로, autotune이 Triton 선택지를 탐색 |
| 새 연산자·attention 변형을 tile 수준으로 직접 쓰고 싶다 | **Triton** | 생산성과 성능의 균형. `torch.compile`과 같은 백엔드 |
| Triton 결과가 손으로 짠 커널보다 느리고, 원인이 layout·파이프라인·warp 배치에 있다 | **Gluon** | 컴파일러가 숨기던 결정을 직접 제어. 같은 JIT·런처 사용 |
| sm_120에서 FP4 GEMM | Triton `tl.dot_scaled` 또는 CUTLASS / CuTe DSL | Gluon에는 sm_120 block-scaled API가 없다 |

**전형적인 개발 흐름**:
1. Helion이나 Triton으로 정확성과 기본 성능을 확보한다.
2. 병목 커널만 Gluon으로 내려간다.
3. Triton이 생성한 TTGIR(layout, 파이프라인)을 참고해 Gluon 코드의 출발점으로 삼을 수 있다. 이것이 두 도구가 같은 TTGIR에서 합쳐지기 때문에 가능한 흐름이다.

## 7. 참고 원본

- `helion-README.md` (Helion 정의, 예제, autotune 설정)
- `gluon-tutorial-01-intro.py`, `gluon-tutorial-02-layouts.py`, `gluon-tutorial-08-warp-specialization.py` (Gluon 정의, layout, warp specialization. 08은 "Hopper and newer"이며 예제는 tcgen05 기반)
- `triton-nvidia-backend-compiler.py` (`add_stages`, `make_ttgir`, `gluon_to_ttgir`)
- `triton-AccelerateMatmul.cpp`, `triton-TargetFeatures.h` (sm_120 MMA 선택, cluster 게이트)
- `gluon-nvidia-ampere-init.py`, `gluon-nvidia-blackwell-init.py` (Gluon NVIDIA 모듈 구성)
