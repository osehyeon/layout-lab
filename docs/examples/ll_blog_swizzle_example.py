"""ita9naiwa 블로그 "Linear Layout in Triton (1)" 의 예제를 그대로 재현한다. GPU 불필요.

17-linearlayout-blog-part1.md 의 숫자를 만든다.
  - 4 스레드 × 4 워프 → 4×4 텐서. 기저 넷: thread {1,1},{2,2}, warp {0,1},{0,2}
  - 선형성 규칙 L(t1⊕t2, w1⊕w2) = L(t1,w1) ⊕ L(t2,w2) 로 16칸을 전부 채운다
  - 결과가 (t, w⊕t) 스위즐과 같은지, 블로그의 apply(thread=3, warp=2) = (3, 1) 이 맞는지 확인
  - GF(2) 행렬-벡터 곱 예제 (2.3 / 2.4 절) 의 답 0b1100 도 확인
"""

# ── 기저: 입력 축마다 2의 거듭제곱 지점의 값 (dim0, dim1) ──
bases = {
    "thread": [(1, 1), (2, 2)],   # L(thread=1, warp=0) = (1,1),  L(thread=2, warp=0) = (2,2)
    "warp":   [(0, 1), (0, 2)],   # L(thread=0, warp=1) = (0,1),  L(thread=0, warp=2) = (0,2)
}


def apply(**hw):
    """선형성 규칙: 켜진 비트의 기저를 성분별 XOR."""
    d0 = d1 = 0
    for name, v in hw.items():
        for j, (b0, b1) in enumerate(bases[name]):
            if v >> j & 1:
                d0 ^= b0
                d1 ^= b1
    return d0, d1


print("L(t, w) 표  (행 = thread, 열 = warp)")
print("      " + "".join(f"  w={w}  " for w in range(4)))
for t in range(4):
    print(f"t={t}  " + "".join(f" {apply(thread=t, warp=w)} " for w in range(4)))

# 블로그의 표와 스위즐 식 (t, w⊕t) 대조
for t in range(4):
    for w in range(4):
        assert apply(thread=t, warp=w) == (t, w ^ t)
print("\n표 전체가 (t, w ⊕ t) 스위즐과 일치  ✓")

# 블로그 3.3 의 apply 예
r = apply(thread=3, warp=2)
print(f"apply(thread=3, warp=2) = {r}")
assert r == (3, 1)
print("블로그의 결과 (3, 1) 과 일치  ✓   ← thread 3 = 1⊕2 → (1,1)⊕(2,2) = (3,3),  warp 2 → (0,2),  합쳐서 (3, 3⊕2) = (3, 1)")

# L(0,0) 은 기저와 무관하게 (0,0)
assert apply(thread=0, warp=0) == (0, 0)

# ── 2.3 / 2.4 절: GF(2) 행렬-벡터 곱 ──
# 열을 정수로 읽은 행렬 [0001, 0010, 1110, 1100], 입력 0110 → 켜진 비트 1, 2 의 열을 XOR
cols = [0b0001, 0b0010, 0b1110, 0b1100]
a = 0b0110
out = 0
for j, c in enumerate(cols):
    if a >> j & 1:
        out ^= c
print(f"\n행렬 열 {[format(c, '04b') for c in cols]} × 입력 {a:04b} = {out:04b}")
assert out == 0b1100
print("블로그의 답 1100 과 일치  ✓")

# ── 2편 §1.1.1: blocked → LinearLayout 의 기저 ──
# #blocked<sizePerThread=[4,2], threadsPerWarp=[8,4], warpsPerCTA=[2,2], order=[1,0]>, shape (64,16).
# blocked 는 order 의 빠른 축(dim1)부터 register → lane → warp 순서로 비트를 채운다.
def blocked_bases(size_per_thread, threads_per_warp, warps_per_cta, order):
    out = {"register": [], "lane": [], "warp": []}
    for name, per in (("register", size_per_thread), ("lane", threads_per_warp), ("warp", warps_per_cta)):
        for d in order:                      # 빠른 축부터
            stride = 1
            # 이 축에서 앞 단계(register/lane)가 이미 덮은 크기만큼 stride 를 띄운다
            if name == "lane":
                stride = size_per_thread[d]
            elif name == "warp":
                stride = size_per_thread[d] * threads_per_warp[d]
            n = per[d]
            k = 1
            while k < n:
                v = [0, 0]
                v[d] = stride * k
                out[name].append(tuple(v))
                k *= 2
    return out

bb = blocked_bases([4, 2], [8, 4], [2, 2], [1, 0])
print("\n2편 §1.1.1 blocked 의 기저 (dim0, dim1):")
for name in ("register", "lane", "warp"):
    print(f"  {name:8s}", " ".join(f"{1 << k}→{b}" for k, b in enumerate(bb[name])))
expected = {"register": [(0, 1), (1, 0), (2, 0)],
            "lane": [(0, 2), (0, 4), (4, 0), (8, 0), (16, 0)],
            "warp": [(0, 8), (32, 0)]}
assert bb == expected, bb
print("블로그가 적은 ll 출력과 일치  ✓")

# flattenIns: register → lane → warp 순서로 (minor → major) 이어 붙인 것이 새 register 기저
flat = bb["register"] + bb["lane"] + bb["warp"]
assert flat[3] == (0, 2) and flat[8] == (0, 8) and flat[9] == (32, 0)   # register=8 → (0,2), 256 → (0,8), 512 → (32,0)
print("flattenIns: register=8→(0,2), 256→(0,8), 512→(32,0)  — 블로그 출력과 일치  ✓")

# ── 2편 §2.1 예제 1: identity1D(4, lane) * identity1D(8, register) → dim0 = lane + 4·register ──
# 곱은 같은 출력 축에서 뒤 인자의 기저를 앞 인자의 크기만큼 밀어 붙인다 (블로그 §2 "Arithmetic view" 와 §2.1 검증값).
def prod_apply(lane, register):
    return (lane & 3) + ((register & 7) << 2)
assert prod_apply(0, 1) == 4 and prod_apply(1, 0) == 1 and prod_apply(3, 2) == 11
print("2편 §2.1: apply(reg=1)=4, apply(lane=1)=1, apply(reg=2,lane=3)=11  ✓")
# 같은 구성을 2편 §3.3 예제 2 는 "L(lane=2, register=3) = 2 ⊕ 3 = 1" 이라고 적었다. 위 식으로는 2 + 12 = 14 다.
print(f"   같은 layout 에서 (lane=2, register=3) → {prod_apply(2, 3)}  (블로그 §3.3 예제 2 의 '1' 과 다르다 — 본문 §4 참조)")
