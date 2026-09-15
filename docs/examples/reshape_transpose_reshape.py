"""reshape -> transpose -> reshape 가 만드는 배치를 F2 기저로 추적한다.

11-linearlayout-history.md §1의 예. GPU 없이 CPU만으로 실행된다.

원소 32개를 워프 하나(lane 32개)가 하나씩 나눠 가진 상태에서 시작한다.
데이터는 한 바이트도 움직이지 않는데, 결과 배치는 blocked layout으로
표현할 수 없는 꼴이 된다.
"""

LANES = 32
R, C = 4, 8                       # reshape (32,) -> (4, 8) -> transpose -> (8, 4)

lane_bases = [1, 2, 4, 8, 16]     # 시작: lane 비트 j -> 인덱스 비트 j (항등)

to_2d = lambda b: (b // C, b % C)  # i = 8r + c
transp = lambda rc: (rc[1], rc[0])
to_1d = lambda rc: rc[0] * R + rc[1]  # (8,4) 에서 i = 4r + c

result_bases = [to_1d(transp(to_2d(b))) for b in lane_bases]


def owner(bases):
    """기저에서 lane -> 원소 대응표를 만든다 (켜진 비트의 몫을 XOR)."""
    out = []
    for lane in range(LANES):
        e = 0
        for j, b in enumerate(bases):
            if lane >> j & 1:
                e ^= b
        out.append(e)
    return out


print("시작 기저 :", lane_bases)
print("결과 기저 :", result_bases)
print()
own = owner(result_bases)
print("lane -> 원소 :", own)
inv = [0] * LANES
for lane, e in enumerate(own):
    inv[e] = lane
print("원소 -> lane :", inv, "  <- 그림 ④ 줄")
print()

# blocked layout 의 1D lane 기저는 [s, 2s, 4s, 8s, 16s] 꼴뿐이다.
cands = [[s * (1 << j) for j in range(5)] for s in (1, 2, 4, 8, 16)]
print("blocked 후보 :", cands)
print("표현 가능?   :", result_bases in cands)
