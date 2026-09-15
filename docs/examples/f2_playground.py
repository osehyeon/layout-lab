"""F₂(원소가 0, 1 둘뿐인 체) 위의 선형대수를 직접 확인한다.

12-f2-primer.md 의 동반 스크립트. GPU 없이 CPU만으로 실행된다.

비트열을 F₂ 벡터로 보면 GPU 배치(layout)가 행렬이 되고,
행렬 곱·역행렬·가우스 소거법이 그대로 쓰인다.
"""

N = 5                            # 5차원 = 2^5 = 32개 벡터 = 워프 하나


# ─────────────────────────────────────────────────────────────
# 1. 선형사상은 기저에서의 값만으로 정해진다
# ─────────────────────────────────────────────────────────────
def apply(x, bases):
    """켜진 비트의 기저값을 XOR 한다."""
    y = 0
    for j, b in enumerate(bases):
        if x >> j & 1:
            y ^= b
    return y


def check_linear(bases):
    """f(x ^ y) == f(x) ^ f(y) 를 모든 쌍에 대해 확인."""
    n = 1 << len(bases)
    return all(apply(x ^ y, bases) == apply(x, bases) ^ apply(y, bases)
               for x in range(n) for y in range(n))


# ─────────────────────────────────────────────────────────────
# 2. 행렬 표현 — 기저값을 세로로 세운 것
# ─────────────────────────────────────────────────────────────
def show(bases, label):
    print(f"{label}   기저값 {bases}")
    print("        입력 비트 " + " ".join(str(j) for j in range(len(bases))))
    for r in range(N):
        row = " ".join(str(b >> r & 1) for b in bases)
        print(f"  출력 비트 {r} │ {row}")
    print()


# ─────────────────────────────────────────────────────────────
# 3. 가우스 소거법으로 역행렬 — 뺄셈이 XOR 이라 부호가 없다
# ─────────────────────────────────────────────────────────────
def inverse(bases):
    """열 벡터 목록으로 주어진 정사각 F₂ 행렬의 역행렬. 없으면 None."""
    a = list(bases)                       # 원본
    b = [1 << j for j in range(N)]        # 단위행렬
    for col in range(N):
        piv = next((k for k in range(col, N) if a[k] >> col & 1), None)
        if piv is None:
            return None                   # 가역이 아니다
        a[col], a[piv] = a[piv], a[col]
        b[col], b[piv] = b[piv], b[col]
        for k in range(N):
            if k != col and (a[k] >> col & 1):
                a[k] ^= a[col]            # 빼기가 아니라 XOR
                b[k] ^= b[col]
    return b


def compose(f, g):
    """(f ∘ g)(x) = f(g(x)) — 행렬 곱."""
    return [apply(b, f) for b in g]


# ─────────────────────────────────────────────────────────────
# 4. 자리올림이 나면 선형이 아니다
# ─────────────────────────────────────────────────────────────
def stride_is_linear(s, nbits=3):
    """좌표 c 의 기여 c*s 가 XOR 로 표현되는가."""
    bases = [s << j for j in range(nbits)]
    return all(apply(c, bases) == c * s for c in range(1 << nbits))


if __name__ == "__main__":
    IDENTITY = [1, 2, 4, 8, 16]
    SWAPPED = [4, 8, 16, 1, 2]        # reshape(4,8) → transpose → reshape

    print("── 1. 선형성 ──")
    print("IDENTITY 선형?", check_linear(IDENTITY))
    print("SWAPPED  선형?", check_linear(SWAPPED))
    print()

    print("── 2. 행렬 표현 ──")
    show(IDENTITY, "단위행렬 ")
    show(SWAPPED, "치환행렬 ")

    print("── 3. 역행렬과 합성 ──")
    inv = inverse(SWAPPED)
    print("SWAPPED⁻¹        =", inv)
    print("SWAPPED ∘ 역     =", compose(SWAPPED, inv), " (단위행렬이어야 한다)")
    # A = 값이 지금 놓인 배치, B = 소비자가 요구하는 배치. 둘 다 좌표 → 인덱스.
    # 변환은 출발 좌표 → 도착 좌표이므로 B⁻¹ ∘ A 다 (Triton: A.invertAndCompose(B)).
    A, B = SWAPPED, IDENTITY
    conv = compose(inverse(B), A)                    # B⁻¹ ∘ A
    print("B⁻¹ ∘ A          =", conv)
    print("  → 단위행렬인가?", conv == IDENTITY, " (아니면 변환이 필요하다)")
    print()

    print("── 4. 브로드캐스트는 가역이 아니다 ──")
    BCAST = [1, 2, 0, 8, 16]          # 기저값 0 = 그 비트가 아무 데도 안 간다
    print("기저값에 0 이 있으면 역행렬:", inverse(BCAST), " (None = 비단사)")
    print("  lane 0 과 lane 4 가 같은 원소:",
          apply(0, BCAST), apply(4, BCAST))
    print()

    print("── 5. stride 는 2의 거듭제곱이어야 한다 ──")
    for s in (1, 2, 4, 6, 8, 3):
        mark = "✓" if stride_is_linear(s) else "✗  자리올림"
        print(f"  stride {s:2d}  →  {mark}")
