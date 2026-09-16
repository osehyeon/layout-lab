"""16-linearlayout-talk-primer.md 의 슬라이드 예제들을 F2 위에서 직접 재현한다. GPU 불필요.

재현하는 것 (ASPLOS'26 발표 슬라이드 기준):
  A. "Defining Linear Layouts" — 16×16 blocked layout 의 표(Location → Register/Thread/Warp) 와 행렬 L,
     W = L × v 확인. 슬라이드가 W = (0,1) 로 적은 항목이 같은 슬라이드의 표에서는 (1,0) 이라는 것도 본다.
  B. "Broadcasting" — 0 열(zero column) = 여러 스레드가 같은 값. 단사(injective)는 아니지만 전사(surjective)라
     우역원(right inverse) 이 있다.
  C. "Product" — L1: Reg → offset, L2: Thread → offset 의 곱 = 레지스터 무늬를 스레드마다 반복.
  D. "Layout Conversions" — B⁻¹∘A 가 Thread·Warp 블록에서 항등이면 intra-thread, Warp 블록에서만 항등이면
     intra-warp, 둘 다 아니면 intra-CTA(SMEM).

규약은 저장소와 같다: layout = 하드웨어 좌표 → 텐서 인덱스. 슬라이드의 W = L × v 도 v 가 하드웨어 비트,
W 가 논리 좌표이므로 같은 방향이다.
"""

from itertools import product as iproduct


# ───────────────────────── 공통 도구 ─────────────────────────
def apply(bases, x):
    """기저 목록(비트 j 가 켜지면 bases[j] 를 XOR) 을 정수 x 에 적용."""
    out = 0
    for j, b in enumerate(bases):
        if x >> j & 1:
            out ^= b
    return out


class Layout:
    """입력 축 = 이름 있는 비트 묶음 (예: reg 2비트, thr 5비트, wrp 1비트). 출력 = (i, j) 두 축을 하나의 정수로.

    out 은 (i << JBITS) | j 로 편다. 기저값도 같은 방식으로 적는다.
    """

    def __init__(self, in_dims, bases, jbits):
        self.in_dims = in_dims            # [("reg", 2), ("thr", 5), ("wrp", 1)]
        self.bases = bases                # {"reg": [..], "thr": [..], "wrp": [..]}  값 = (i<<jbits)|j
        self.jbits = jbits

    def coord(self, **hw):
        """하드웨어 좌표 → (i, j)."""
        out = 0
        for name, _ in self.in_dims:
            out ^= apply(self.bases[name], hw.get(name, 0))
        return out >> self.jbits, out & ((1 << self.jbits) - 1)

    def flat_in(self, **hw):
        """(reg, thr, wrp) 를 하나의 정수로 이어 붙인다 — 표 만들기용."""
        x, shift = 0, 0
        for name, nb in self.in_dims:
            x |= (hw.get(name, 0) & ((1 << nb) - 1)) << shift
            shift += nb
        return x

    def all_inputs(self):
        ranges = [range(1 << nb) for _, nb in self.in_dims]
        for vals in iproduct(*ranges):
            yield dict(zip([n for n, _ in self.in_dims], vals))

    def matrix(self, ibits):
        """행 = 출력 비트 (j0..j_{jbits-1}, i0..), 열 = 입력 비트 (in_dims 순서). 0/1 리스트의 리스트."""
        cols = []
        for name, nb in self.in_dims:
            for k in range(nb):
                cols.append(self.bases[name][k])
        rows = []
        for r in range(self.jbits + ibits):
            rows.append([(c >> r) & 1 for c in cols])
        return rows


def fmt_bits(x, n):
    return format(x, f"0{n}b")


# ───────────────────────── A. 슬라이드의 16×16 blocked layout ─────────────────────────
# 그림: dim1(가로) 16, dim0(세로) 16. 스레드 하나가 2×2 (r0 r1 / r2 r3), 스레드 8개가 가로로, 4줄이 세로로 → 워프 하나가 8×16.
# 워프 둘이 세로로 → 16×16.
JB = 4  # j 는 4비트 (16 열)
def ij(i, j):
    return (i << JB) | j

slide16 = Layout(
    in_dims=[("reg", 2), ("thr", 5), ("wrp", 1)],
    bases={
        "reg": [ij(0, 1), ij(1, 0)],                        # r 비트0 → j+1 (r0→r1),  r 비트1 → i+1 (r0→r2)
        "thr": [ij(0, 2), ij(0, 4), ij(0, 8), ij(2, 0), ij(4, 0)],  # t 비트0..2 → j+2,4,8 ; t 비트3,4 → i+2,4
        "wrp": [ij(8, 0)],                                  # w 비트0 → i+8
    },
    jbits=JB,
)

print("A. 슬라이드 'Defining Linear Layouts' — 표 다시 만들기 (Location → Register / Thread / Warp)")
table = {}
for hw in slide16.all_inputs():
    table[slide16.coord(**hw)] = hw
for loc in [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (2, 2), (2, 3), (3, 2), (3, 3)]:
    hw = table[loc]
    print(f"   {loc} / (0b{fmt_bits(loc[0], 2)}, 0b{fmt_bits(loc[1], 2)})   r{hw['reg']} / 0b{fmt_bits(hw['reg'], 2)}"
          f"   t{hw['thr']} / 0b{fmt_bits(hw['thr'], 4)}   w{hw['wrp']} / 0b{fmt_bits(hw['wrp'], 2)}")
# 슬라이드의 표와 대조
assert table[(0, 1)] == dict(reg=1, thr=0, wrp=0)
assert table[(1, 0)] == dict(reg=2, thr=0, wrp=0)
assert table[(2, 2)] == dict(reg=0, thr=9, wrp=0)
assert table[(3, 3)] == dict(reg=3, thr=9, wrp=0)
print("   슬라이드 표의 4개 항목과 일치  ✓")
print()
print("   행렬 L (행 = j0 j1 j2 j3 | i0 i1 i2 i3,  열 = reg0 reg1 | thr0..thr4 | wrp0):")
M = slide16.matrix(ibits=4)
for r, row in enumerate(M):
    label = f"j{r}" if r < JB else f"i{r - JB}"
    print(f"   {label}  " + " ".join(map(str, row[:2])) + " | " + " ".join(map(str, row[2:7])) + " | " + str(row[7]))
W = slide16.coord(reg=2, thr=0, wrp=0)
print(f"   v = (reg=2, thr=0, wrp=0) → W = L × v = (i, j) = {W}")
assert W == (1, 0)
print("   슬라이드는 이 항목을 W = (0, 1) 로 적었지만 같은 슬라이드의 표는 (1, 0) 이다 — 좌표 순서 표기 차이로 보인다.")
print()

# ───────────────────────── B. 브로드캐스트 = 0 열 ─────────────────────────
# 슬라이드: Reg 2비트, Thread 4비트인데 thread 비트 2,3 열이 전부 0 → 스레드 t, t+4, t+8, t+12 가 같은 값.
# 타일 2행 × 8열: j = reg0, thr0, thr1 (3비트), i = reg1 (1비트).
bc = Layout(in_dims=[("reg", 2), ("thr", 4)],
            bases={"reg": [ij(0, 1), ij(1, 0)], "thr": [ij(0, 2), ij(0, 4), 0, 0]}, jbits=JB)
print("B. 브로드캐스트 — thread 비트 2, 3 의 열이 0")
print(f"   L(r1, t5) = {bc.coord(reg=1, thr=5)},  L(r1, t1) = {bc.coord(reg=1, thr=1)}  → 같다")
assert bc.coord(reg=1, thr=5) == bc.coord(reg=1, thr=1)
img = {bc.coord(**hw) for hw in bc.all_inputs()}
print(f"   서로 다른 출력 {len(img)}개 (2×8 타일 = 16)  vs  입력 {4 * 16}개 → 단사 아님, 전사 맞음")
assert len(img) == 16
# 우역원: 출력마다 대표 입력 하나를 고른다 (Gaussian elimination 의 해 중 하나). L(R(w)) = w 확인.
right_inv = {}
for hw in bc.all_inputs():
    right_inv.setdefault(bc.coord(**hw), hw)
assert all(bc.coord(**right_inv[w]) == w for w in img)
print("   우역원 R: 출력마다 입력 하나를 고르면 L∘R = I  ✓  (R∘L 은 I 가 아니다 — 브로드캐스트된 스레드는 대표로 접힌다)")
print()

# ───────────────────────── C. 곱 연산자 ─────────────────────────
# L1: Reg(2비트) → 2×2 offset (r0 r1 / r2 r3),  L2: Thread(2비트) → 2×2 (t0 t1 / t2 t3).
# 곱 L1 × L2: Reg × Thread → 4×4. 슬라이드 그림: r0 r1 r0 r1 / r2 r3 r2 r3 / r0 r1 r0 r1 / r2 r3 r2 r3.
JB2 = 2
def ij2(i, j):
    return (i << JB2) | j
prod = Layout(in_dims=[("reg", 2), ("thr", 2)],
              bases={"reg": [ij2(0, 1), ij2(1, 0)], "thr": [ij2(0, 2), ij2(2, 0)]}, jbits=JB2)
print("C. 곱 — Reg 무늬(2×2)를 스레드마다 반복해 4×4")
grid = {}
for hw in prod.all_inputs():
    grid[prod.coord(**hw)] = f"r{hw['reg']}"
for i in range(4):
    print("   " + " ".join(grid[(i, j)] for j in range(4)))
assert [grid[(0, j)] for j in range(4)] == ["r0", "r1", "r0", "r1"]
assert [grid[(1, j)] for j in range(4)] == ["r2", "r3", "r2", "r3"]
print("   슬라이드 그림과 일치  ✓")
print()

# ───────────────────────── D. 변환 등급 ─────────────────────────
def tier(A, B):
    """B⁻¹∘A 를 표로 만들고, thr·wrp 가 그대로인지 본다. 두 layout 은 같은 입력 축·같은 텐서를 가진다고 가정."""
    invB = {B.coord(**hw): hw for hw in B.all_inputs()}
    thr_same = wrp_same = True
    for hw in A.all_inputs():
        dst = invB[A.coord(**hw)]
        thr_same &= dst["thr"] == hw["thr"]
        wrp_same &= dst["wrp"] == hw["wrp"]
    if thr_same and wrp_same:
        return "intra-thread (register 재배열, 명령 0개)"
    if wrp_same:
        return "intra-warp (shfl)"
    return "intra-CTA (SMEM 경유)"

# A: 위의 slide16.  B1: 레지스터 비트만 뒤바꿈 (r 비트0 → i, r 비트1 → j)
B1 = Layout(slide16.in_dims, {"reg": [ij(1, 0), ij(0, 1)], "thr": slide16.bases["thr"], "wrp": slide16.bases["wrp"]}, JB)
# B2: 스레드 비트 순서를 바꿈 (thr 비트0 ↔ 비트3: j+2 와 i+2 를 맞바꿈) — 워프는 그대로
B2 = Layout(slide16.in_dims, {"reg": slide16.bases["reg"],
                              "thr": [ij(2, 0), ij(0, 4), ij(0, 8), ij(0, 2), ij(4, 0)], "wrp": slide16.bases["wrp"]}, JB)
# B3: 워프가 세로(i+8) 대신 가로(j+8)를 맡고, 그 자리를 스레드 비트2가 맡음 → 워프 경계를 넘는다
B3 = Layout(slide16.in_dims, {"reg": slide16.bases["reg"],
                              "thr": [ij(0, 2), ij(0, 4), ij(8, 0), ij(2, 0), ij(4, 0)], "wrp": [ij(0, 8)]}, JB)
print("D. 변환 등급 — B⁻¹∘A 에서 어느 블록이 항등으로 남는가")
for name, B in (("B1 (reg 비트만 다름)", B1), ("B2 (thr 비트 순서 다름)", B2), ("B3 (warp 가 맡는 축이 다름)", B3)):
    print(f"   A → {name:28s}: {tier(slide16, B)}")
assert tier(slide16, B1).startswith("intra-thread")
assert tier(slide16, B2).startswith("intra-warp")
assert tier(slide16, B3).startswith("intra-CTA")
print("   슬라이드의 세 등급(Intra Thread / Intra Warp / Intra CTA)이 그대로 나온다  ✓")
