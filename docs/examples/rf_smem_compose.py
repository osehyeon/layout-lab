"""(register, lane, warp, block) 와 (offset, block) 이 어떻게 이어지는지 — F2 기저로 직접 합성한다.

11-linearlayout-history.md §1 의 figures/rf-smem-coordinate-systems.svg 숫자를 만든다. GPU 불필요.

두 layout 은 입력 축이 다르지만 출력이 같다 (텐서 인덱스). 그래서
    cvt = shared⁻¹ ∘ reg : (register, lane, warp, block) → (offset, block)
이 항상 정의된다. Triton 의 invertAndComposeLocal(sharedLayout, regLayout) 이 이 식이다.
"""

LANES = 32
N = 32  # 원소 수 = 5 비트


def apply(bases, x):
    """기저 목록에 좌표 x 를 적용: 켜진 비트의 기저를 XOR."""
    out = 0
    for j, b in enumerate(bases):
        if x >> j & 1:
            out ^= b
    return out


def invert(bases, nbits):
    """순열 layout 의 역: 인덱스 → 좌표 표 (전수 조사, 5비트라 32개)."""
    table = {}
    for x in range(1 << nbits):
        y = apply(bases, x)
        assert y not in table, "가역이 아니다"
        table[y] = x
    return table


# ── 레지스터 쪽: (register, lane, warp, block) → 인덱스 ─────────────────────
# 원소 32개를 워프 하나가 하나씩 든다. register·warp·block 은 비트가 0개, lane 만 5비트.
reg_bases = {"register": [], "lane": [1, 2, 4, 8, 16], "warp": [], "block": []}

# ── SMEM 쪽: (offset, block) → 인덱스 ────────────────────────────────────────
# (8,4) row-major 로 놓으면 offset = 인덱스 (항등). 스위즐 예: offset 비트 2 가 인덱스 비트 0 도 뒤집는다.
shared_plain = {"offset": [1, 2, 4, 8, 16], "block": []}
shared_swz = {"offset": [1, 2, 4 ^ 1, 8, 16], "block": []}


def compose(shared, reg):
    """cvt(lane) = shared⁻¹(reg(lane)). 워프 하나·CTA 하나라 lane 축만 살아 있다."""
    inv = invert(shared["offset"], 5)
    return [inv[apply(reg["lane"], lane)] for lane in range(LANES)]


for name, sh in (("row-major (항등)", shared_plain), ("스위즐 offset[2]=4^1", shared_swz)):
    cvt = compose(sh, reg_bases)
    print(f"sharedLayout = {name:22s} offset 기저 {sh['offset']}")
    print(f"  cvt: lane j → offset  {cvt[:8]} … (앞 8개)")

# 스위즐 경우의 닫힌 식 확인: offset = j XOR ((j>>2)&1)
cvt_swz = compose(shared_swz, reg_bases)
assert cvt_swz == [j ^ ((j >> 2) & 1) for j in range(LANES)]
assert compose(shared_plain, reg_bases) == list(range(LANES))
print()
print("스위즐 cvt 는 j ^ ((j>>2)&1) 과 일치  ✓")

# ── block 축은 양쪽에서 같은 것이다 ────────────────────────────────────────────
# 두 layout 모두 combineCtaCgaWithShape(ctaLayout, cgaLayout, shape) 로 block 축을 얻는다
# (lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp). CTA 두 개짜리 클러스터에서 원소 64개를
# 반씩 나눠 들면 block 비트 하나가 인덱스 비트 5 (=32) 를 맡는다 — 레지스터 쪽도 SMEM 쪽도 같은 기저다.
reg_cluster = {"lane": [1, 2, 4, 8, 16], "block": [32]}
shared_cluster = {"offset": [1, 2, 4, 8, 16], "block": [32]}
# 합성: block 비트가 인덱스 32 로 가고, shared⁻¹ 가 그것을 다시 block 비트로 되돌린다 → block → block 항등.
# 이때 cvt 의 block 부분이 항등이면 각 CTA 가 자기 SMEM 만 쓴다. 항등이 아니면 (예: reg 는 block→32 인데
# shared 는 block→0, 즉 SMEM 타일이 CTA 마다 복제) 다른 CTA 의 SMEM 을 읽어야 하고 그것이 DSMEM 경로다.
idx_of_block1 = apply(reg_cluster["block"], 1)
print(f"클러스터 예: reg  block=1 → 인덱스 {idx_of_block1} ; shared block=1 → 인덱스 {apply(shared_cluster['block'], 1)}"
      "  → cvt 의 block→block 은 항등 (자기 CTA 의 SMEM)")
