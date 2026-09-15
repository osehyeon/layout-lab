"""reshape → transpose → reshape 를 세 번 돌리면 제자리로 온다.

15-reshape-chain-walkthrough.md 의 동반 스크립트. GPU 없이 CPU만으로 실행된다.
numpy 만 쓴다.
"""
import numpy as np

N = 3                                   # 8원소 = 2^3, 기저 3개


# ─────────────────────────────────────────────────────────────
# 한 회차 = reshape(2,4) → .T → reshape(8)
# ─────────────────────────────────────────────────────────────
def step(a):
    return a.reshape(2, 4).T.reshape(8).copy()


def basis_from_table(own):
    """대응표 → 기저.  비트 하나만 켜진 자리 세 줄만 읽는다."""
    return [int(own[1 << j]) for j in range(N)]


def invert_table(t):
    """표를 뒤집는다.  numpy 배열(새 자리 → 옛 원소) ↔ layout(lane → 새 인덱스)."""
    out = [0] * len(t)
    for pos, e in enumerate(t):
        out[e] = pos
    return out


def layout_basis(arr):
    """numpy 배열에서 layout 기저를 얻는다 — 배열은 layout 의 역이다."""
    return basis_from_table(invert_table(arr))


def table_from_basis(bases):
    """기저 → 대응표.  켜진 비트의 기저값을 XOR."""
    out = []
    for p in range(1 << N):
        v = 0
        for j, b in enumerate(bases):
            if p >> j & 1:
                v ^= b
        out.append(v)
    return out


def compose(f, g):
    """(f ∘ g)(x) = f(g(x)).  g 의 기저값을 f 에 통과시킨다."""
    return [table_from_basis(f)[v] for v in g]


if __name__ == "__main__":
    print("── 1. 세 회차를 돌린다 ──")
    print("  배열은 '새 자리 → 옛 원소' 이고, layout 은 그 역인 'lane → 새 인덱스' 다.")
    a = np.arange(8)
    hist = [a.tolist()]
    print(f"  0회  {a.tolist()}   배열기저 {basis_from_table(a)}"
          f"   layout {layout_basis(a.tolist())}")
    for k in range(1, 4):
        a = step(a)
        hist.append(a.tolist())
        print(f"  {k}회  {a.tolist()}   배열기저 {basis_from_table(a)}"
              f"   layout {layout_basis(a.tolist())}")
    print(f"  3회가 0회와 같은가? {hist[3] == hist[0]}")
    print()

    print("── 2. 두 방향이 서로 역이다 ──")
    for k in (1, 2):
        own = hist[k]
        b = basis_from_table(own)
        print(f"  {k}회  표 → 기저  {own} → {b}")
        print(f"        기저 → 표  {b} → {table_from_basis(b)}"
              f"   일치? {table_from_basis(b) == own}")
    print()

    print("── 2b. 배열의 기저와 layout 기저는 서로 역이다 ──")
    arr = hist[1]
    print(f"  numpy 배열   {arr}   기저 {basis_from_table(arr)}"
          "   ← 새 자리 p 에 들어온 '옛 원소'")
    print(f"  layout      {invert_table(arr)}   기저 {layout_basis(arr)}"
          "   ← lane 이 든 값의 '새 인덱스'  (규약)")
    print("  layout 쪽이 LinearLayout 의 정의(하드웨어 좌표 → 텐서 인덱스)와 같다.")
    print(f"  두 기저가 서로 역인가? "
          f"{invert_table(table_from_basis(basis_from_table(arr))) == invert_table(arr)}")
    print()

    print("── 3. 회차는 기저의 합성이다 ──")
    f = layout_basis(hist[1])             # 한 회차가 layout 에 하는 일
    cur = [1, 2, 4]                        # 항등
    for k in range(1, 4):
        cur = compose(f, cur)
        print(f"  f^{k} = {cur}   기록과 일치? {cur == layout_basis(hist[k])}")
    print()

    print("── 4. 복사 횟수 ──")
    print("  GMEM  : 회차마다 8칸을 새 버퍼로 옮긴다        → 3회면 복사 3번")
    print("  레지스터: 기저만 갈아 끼운다                    → 3회면 복사 0번")
    print("  그리고 3회차 기저가 항등이므로 낼 명령도 0개다.")

    print()
    print("── 5. 주기는 b / gcd(r, b) 다 ──")
    import math
    print("   n  shape        실측  b/gcd(r,b)")
    for n in (8, 16, 32, 64):
        b = n.bit_length() - 1
        for rows in (2, 4, 8):
            if rows >= n:
                continue
            a = np.arange(n)
            k = 0
            while True:
                a = a.reshape(rows, n // rows).T.reshape(n).copy()
                k += 1
                if (a == np.arange(n)).all():
                    break
            r = rows.bit_length() - 1
            pred = b // math.gcd(r, b)
            mark = "✓" if k == pred else "✗"
            print(f"  {n:3d}  ({rows},{n//rows})".ljust(18)
                  + f"{k:5d} {pred:10d}   {mark}")

    print()
    print("── 6. C = B⁻¹A 에서 남는 축이 비용을 정한다 ──")
    NB = 6                     # register 2 + lane 2 + warp 2
    GRP = {0: "register", 1: "register", 2: "lane", 3: "lane",
           4: "warp", 5: "warp"}

    def ap6(x, b):
        y = 0
        for j, v in enumerate(b):
            if x >> j & 1:
                y ^= v
        return y

    def inv6(b):
        a, e = list(b), [1 << j for j in range(NB)]
        for c in range(NB):
            piv = next(k for k in range(c, NB) if a[k] >> c & 1)
            a[c], a[piv] = a[piv], a[c]
            e[c], e[piv] = e[piv], e[c]
            for k in range(NB):
                if k != c and (a[k] >> c & 1):
                    a[k] ^= a[c]
                    e[k] ^= e[c]
        return e

    def remaining(C):
        t = set()
        for j in range(NB):
            if C[j] == (1 << j):
                continue
            t.add(GRP[j])
            for i in range(NB):
                if C[j] >> i & 1:
                    t.add(GRP[i])
        return [d for d in ("register", "lane", "warp") if d in t]

    def verdict(rem):
        if not rem:
            return "없음"
        if rem == ["register"]:
            return "SSA 재배열 (공짜)"
        if "warp" in rem:
            return "SMEM 왕복 + 배리어"
        return "shfl"

    IDN = [1 << j for j in range(NB)]
    for lab, Bb in [("두 배치가 같다        ", IDN),
                    ("레지스터 두 칸 뒤바뀜  ", [2, 1, 4, 8, 16, 32]),
                    ("lane 두 비트 뒤바뀜   ", [1, 2, 8, 4, 16, 32]),
                    ("레지스터 ↔ lane      ", [4, 2, 1, 8, 16, 32]),
                    ("lane ↔ warp         ", [1, 2, 16, 8, 4, 32])]:
        C = [ap6(v, inv6(Bb)) for v in IDN]
        rem = remaining(C)
        print(f"  {lab} C = {str(C):26s} 남는 축 {str(rem):32s} {verdict(rem)}")

    print()
    print("── 7. 거리가 아니라 경계다 (lane 5비트 + warp 1비트) ──")
    AXW = ["lane"] * 5 + ["warp"]

    def ap7(x, b):
        y = 0
        for j, v in enumerate(b):
            if x >> j & 1:
                y ^= v
        return y

    for lab, Cw in [("lane 비트 전부 뒤집기  ", [16, 8, 4, 2, 1, 32]),
                    ("lane 하나 ↔ warp     ", [1, 2, 4, 8, 32, 16])]:
        mx = cross = 0
        for x in range(1 << 6):
            y = ap7(x, Cw)
            mx = max(mx, abs(y - x))
            cross += (y >> 5) != (x >> 5)
        axes = sorted({AXW[j] for j, v in enumerate(Cw) if v != (1 << j)} |
                      {AXW[v.bit_length() - 1]
                       for j, v in enumerate(Cw) if v != (1 << j)})
        print(f"  {lab} 최대 이동 {mx:3d}   워프 넘는 값 {cross:2d}개   "
              f"{'SMEM 왕복' if 'warp' in axes else 'shfl'}")
    print("  → 더 멀리 옮기는 쪽이 더 싸다. 기준은 거리가 아니라 경계다.")
