"""PyTorch로 '주소 함수'를 직접 확인하는 실습 스크립트.

문서 07~08의 layout 논의 중 **메모리 주소 쪽**(HBM·SMEM 단계)을 파이썬에서 눈으로 보는 것이 목적이다.
레지스터 소유(MMA fragment)는 PyTorch에서 보이지 않는다. 마지막 절 참고.

실행:  python3 tensor_layout_playground.py      (GPU 불필요, CPU만으로 동작)
"""

import torch


def rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------- 1
rule("1. tensor = (storage, shape, stride, offset) — 주소는 계산으로 나온다")

t = torch.arange(24, dtype=torch.int32).reshape(4, 6)
print("shape :", tuple(t.shape))
print("stride:", t.stride(), "  <- 각 축으로 한 칸 갈 때 건너뛰는 '원소' 수")
print("offset:", t.storage_offset())
print(t)

r, c = 2, 3
linear = t.storage_offset() + r * t.stride(0) + c * t.stride(1)
flat = t.reshape(-1)  # contiguous 이므로 view
print(f"\nt[{r},{c}] = {t[r, c].item()}   직접 계산한 선형 인덱스 = {linear} -> {flat[linear].item()}")
print("=> 주소 함수 addr(r,c) = offset + r*stride0 + c*stride1")


# ---------------------------------------------------------------- 2
rule("2. view vs copy — 무엇이 메모리를 건드리는가")

base = torch.arange(12, dtype=torch.int32).reshape(3, 4)


def info(name, x, ref=None):
    same = (x.data_ptr() == ref.data_ptr()) if ref is not None else True
    print(f"{name:<26} shape={tuple(x.shape)!s:<10} stride={x.stride()!s:<12}"
          f" contig={x.is_contiguous()!s:<5} 같은 메모리={same}")


info("base", base, base)
info("base.T (전치)", base.T, base)
info("base.permute(1,0)", base.permute(1, 0), base)
info("base.view(2,6)", base.view(2, 6), base)
info("base.T.contiguous()", base.T.contiguous(), base)
info("base.T.reshape(-1)", base.T.reshape(-1), base)
print("\n전치는 stride만 바꾼다(복사 없음). contiguous()는 새로 복사한다.")
print("reshape는 view가 가능하면 view, 아니면 복사한다 -> 위 마지막 줄에서 '같은 메모리=False'")


# ---------------------------------------------------------------- 3
rule("3. 문서 그림과 맞춰 보기 — A[3,5]의 바이트 오프셋")

LD = 64          # 원본 행렬의 leading dimension (원소 개수)
M = 16           # 타일 크기
big = torch.zeros(M, LD, dtype=torch.float16)
tile = big[:, :M]                      # 16x16 타일 (열 앞쪽 16개)
print("big  :", tuple(big.shape), "stride", big.stride())
print("tile :", tuple(tile.shape), "stride", tile.stride(), " <- 행 간격은 그대로 64")

r, c = 3, 5
elem_off = r * tile.stride(0) + c * tile.stride(1)
byte_off = elem_off * tile.element_size()
print(f"\nA[{r},{c}] 원소 오프셋 = {r}*{tile.stride(0)} + {c}*{tile.stride(1)} = {elem_off}")
print(f"바이트 오프셋 = {elem_off} * {tile.element_size()} B = {byte_off} B   (문서 그림의 394 B)")
print("타일이 연속인가?", tile.is_contiguous(), "  <- 행마다 48칸씩 건너뛰므로 False")


# ---------------------------------------------------------------- 4
rule("4. swizzle은 '어디에 둘지'를 정하는 함수다 — 뱅크 충돌 세어 보기")

BANKS, BANK_B, ESZ = 32, 4, 2     # SMEM 뱅크 32개(각 4 B), fp16
ROWS, COLS = 32, 64               # 한 행 = fp16 64개 = 128 B = 뱅크 32개 전부


def addr_plain(r, c, pitch):
    """행 우선 그대로."""
    return r * pitch + c


def addr_swizzled(r, c, pitch):
    """청크(2칸 = 4 B = 뱅크 1개) 자리를 XOR로 어긋낸다."""
    chunk, within = divmod(c, 2)
    slot = chunk ^ (r % 32)
    return r * pitch + slot * 2 + within


def bank_histogram(addr_fn, pitch):
    """warp 32 lane이 서로 다른 행에서 같은 열을 4 B씩 읽을 때의 뱅크 분포."""
    used = {}
    for lane in range(32):
        off = addr_fn(lane, 0, pitch)
        bank = (off * ESZ // BANK_B) % BANKS
        used.setdefault(bank, []).append(lane)
    return len(used), max(len(v) for v in used.values())

for name, fn, pitch in (("행 우선", addr_plain, COLS),
                        ("행 끝 패딩(+2칸)", addr_plain, COLS + 2),
                        ("XOR 스위즐", addr_swizzled, COLS)):
    distinct, worst = bank_histogram(fn, pitch)
    print(f"{name:<16} 서로 다른 뱅크 {distinct:2d}개 / 최대 {worst:2d}-way 충돌")

print("""
행 우선은 32 lane이 모두 같은 뱅크를 치므로 접근이 32번 직렬화된다.
패딩은 행 간격을 어긋내서, 스위즐은 행마다 자리를 바꿔서 같은 문제를 푼다.""")

# 스위즐은 데이터를 바꾸지 않는다 — 넣을 때와 뺄 때 같은 함수를 쓰면 원본 그대로다.
src = torch.arange(ROWS * COLS, dtype=torch.int32).reshape(ROWS, COLS)
smem = torch.empty_like(src).reshape(-1)
for r in range(ROWS):
    for c in range(COLS):
        smem[addr_swizzled(r, c, COLS)] = src[r, c]
back = torch.stack([torch.stack([smem[addr_swizzled(r, c, COLS)] for c in range(COLS)])
                    for r in range(ROWS)])
print("스위즐해서 넣고 같은 규칙으로 읽으면 원본과 같은가?", torch.equal(back, src))
print("물리적 순서는 바뀌었는가?", not torch.equal(smem.reshape(ROWS, COLS), src))


# ---------------------------------------------------------------- 5
rule("5. 여기서 멈추는 지점 — PyTorch로는 보이지 않는 것")

print("""보이는 것 : 주소 함수 (shape, stride, offset, 연속성, 복사 여부)
보이지 않는 것 : 소유 함수 (어느 lane의 몇 번째 값이 이 원소를 갖는가)

레지스터 소유는 커널 컴파일 결과에만 존재한다. 보고 싶다면 Triton 커널을 컴파일한 뒤
TTGIR 단계의 layout 어트리뷰트를 열어 보면 된다. 예를 들어

    compiled = my_kernel.warmup(..., grid=(1,))
    print(compiled.asm["ttgir"])        # 키 이름은 Triton 버전에 따라 다를 수 있다

여기에 #blocked, #mma, #linear 같은 인코딩이 붙어 있고, 그것이 문서에서 말한
distributed layout(= LinearLayout)이다.""")
