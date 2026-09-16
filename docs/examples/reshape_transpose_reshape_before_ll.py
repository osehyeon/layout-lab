"""LinearLayout 이전의 Triton 이 reshape -> transpose -> reshape 를 어떻게 처리했는지 재현한다.

11-linearlayout-history.md §1 "도입 전에는 실제로 어떻게 계산됐나" 의 숫자를 만든다.
GPU 없이 CPU 만으로 실행된다. figures/reshape-transpose-reshape-before-ll.svg 가 이 출력을 그린 것이다.

기준 소스: triton f6c3318e (2024-05-08, PR #3794 병합 직전 커밋). 아래 규칙은 그 커밋의
  - BlockedEncodingAttr 빌더           include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td
  - inferTransOpEncoding               lib/Dialect/TritonGPU/IR/Dialect.cpp
  - inferReshapeOpNoReorderEncoding    lib/Dialect/TritonGPU/IR/Dialect.cpp
  - lowerDistributedToDistributed      lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp
  - getScratchConfigForCvtLayout       lib/Analysis/Allocation.cpp
를 파이썬으로 옮긴 것이다 (소스 판독 · 미실행 — 당시 빌드를 돌린 것이 아니다).

이름 있는 blocked layout 밖으로 나가는 순간 (마지막 reshape) 컴파일러가 convert_layout 을 끼워
SMEM 왕복으로 실체화했음을 보인다. reshape_transpose_reshape.py 가 같은 예를 LinearLayout 으로 푼다.
"""

from math import prod

LANES = 32
ELEM_BYTES = 4  # f32 가정 — SMEM 바이트 수 (계산값) 에만 쓰인다


def blocked(shape, order, size_per_thread=None, lanes=LANES, warps=1):
    """TritonGPUAttrDefs.td 의 BlockedEncodingAttr 빌더 — order 의 빠른 축부터 lane 을 채운다."""
    rank = len(shape)
    spt = size_per_thread or [1] * rank
    tpw, wpc = [0] * rank, [0] * rank
    rem_lanes, rem_threads, rem_warps = lanes, lanes * warps, warps
    prev_lanes = prev_warps = 1
    for d in range(rank - 1):
        i = order[d]
        threads = max(1, min(rem_threads, shape[i] // spt[i]))
        tpw[i] = max(1, min(threads, rem_lanes))
        wpc[i] = max(1, min(threads // tpw[i], rem_warps))
        rem_warps //= wpc[i]
        rem_lanes //= tpw[i]
        rem_threads //= threads
        prev_lanes *= tpw[i]
        prev_warps *= wpc[i]
    tpw[order[-1]] = lanes // prev_lanes
    wpc[order[-1]] = warps // prev_warps
    return dict(shape=tuple(shape), sizePerThread=tuple(spt), threadsPerWarp=tuple(tpw),
                warpsPerCTA=tuple(wpc), order=tuple(order))


def default_blocked(shape):
    """getDefaultBlockedEncoding: sizePerThread 전부 1, order = 역순 (row-major)."""
    return blocked(shape, list(reversed(range(len(shape)))))


def fmt(enc):
    return (f"#blocked<sizePerThread={list(enc['sizePerThread'])}, "
            f"threadsPerWarp={list(enc['threadsPerWarp'])}, order={list(enc['order'])}>")


def lane_to_coord(enc, lane):
    """lane 이 드는 (유일한) 좌표. sizePerThread 가 전부 1 이고 워프 하나일 때만 쓴다."""
    assert all(s == 1 for s in enc["sizePerThread"]) and prod(enc["warpsPerCTA"]) == 1
    coord = [0] * len(enc["shape"])
    rem = lane
    for d in enc["order"]:  # 빠른 축부터 자릿수를 뗀다
        coord[d] = rem % enc["threadsPerWarp"][d]
        rem //= enc["threadsPerWarp"][d]
    return tuple(coord)


def owner_map(enc):
    """좌표 -> lane."""
    return {lane_to_coord(enc, l): l for l in range(LANES)}


def infer_trans(enc, perm):
    """inferTransOpEncoding (blocked 가지): 필드를 perm 으로 섞고 order 는 inv(perm)∘order."""
    inv = [perm.index(i) for i in range(len(perm))]
    return dict(shape=tuple(enc["shape"][p] for p in perm),
                sizePerThread=tuple(enc["sizePerThread"][p] for p in perm),
                threadsPerWarp=tuple(enc["threadsPerWarp"][p] for p in perm),
                warpsPerCTA=tuple(enc["warpsPerCTA"][p] for p in perm),
                order=tuple(inv[o] for o in enc["order"]))


def infer_reshape_no_reorder(src_enc, dst_shape):
    """inferReshapeOpNoReorderEncoding 의 우리 예에 필요한 부분.

    반환: (성공 여부, dst 인코딩 또는 실패 사유). 합쳐지는 src 차원들이 물리적으로 연속
    (order 의 큰 쪽부터 붙어 있음) 이어야 한다 — "Dimensions [...] must be physically consecutive."
    """
    if src_enc["shape"] == tuple(dst_shape):
        return True, src_enc
    if src_enc == default_blocked(src_enc["shape"]):
        return True, default_blocked(dst_shape)  # default -> default 는 항상 nop
    # 여기서는 rank-2 -> rank-1 (전부 하나로 합침) 만 다룬다
    assert len(dst_shape) == 1
    src_dims = list(range(len(src_enc["shape"])))
    inv_order = [src_enc["order"].index(d) for d in src_dims]  # dim d 가 몇 번째로 minor 인지
    seq = list(reversed([inv_order[d] for d in src_dims]))
    consecutive = all(seq[i] == seq[i - 1] + 1 for i in range(1, len(seq)))
    if not consecutive:
        return False, (f"Cannot do a non-reordering reshape given this src encoding order. "
                       f"Dimensions {src_dims} must be physically consecutive.")
    raise NotImplementedError("이 예에서는 닿지 않는 경로")


def scratch_config(src_enc, dst_enc):
    """getScratchConfigForCvtLayout: rep shape 에 마지막(order[0]) 축을 max(inVec,outVec) 만큼 pad."""
    shape = list(dst_enc["shape"])
    rank = len(shape)
    inner = rank - 1
    in_ord, out_ord = src_enc["order"], dst_enc["order"]
    in_vec = 1 if (out_ord[0] != inner or in_ord[0] != inner) else src_enc["sizePerThread"][in_ord[0]]
    out_vec = 1 if out_ord[0] != inner else dst_enc["sizePerThread"][out_ord[0]]
    padded = shape[:]
    padded[out_ord[0]] += max(in_vec, out_vec)
    return padded, in_vec, out_vec


def smem_round_trip(src_enc, dst_enc, lane_values):
    """lowerDistributedToDistributed: 각 lane 이 src 좌표에 st.shared, bar.sync, dst 좌표에서 ld.shared."""
    padded, _, _ = scratch_config(src_enc, dst_enc)
    smem = {}
    for lane in range(LANES):  # processReplica(stNotRd=True)
        smem[lane_to_coord(src_enc, lane)] = lane_values[lane]
    # barrier()
    out = [None] * LANES
    for lane in range(LANES):  # processReplica(stNotRd=False)
        out[lane] = smem[lane_to_coord(dst_enc, lane)]
    return out, smem, padded


def show_grid(title, enc, cell):
    rows, cols = enc["shape"]
    print(f"{title}  {fmt(enc)}")
    for r in range(rows):
        print("   " + " ".join(f"{cell((r, c)):2d}" for c in range(cols)))


# ───────────────────────── ①②③ 이름만 바뀐다 ─────────────────────────
x_enc = default_blocked((32,))
lane_values = list(range(LANES))  # lane i 가 x[i] 를 든다
print("① x  (32,)     ", fmt(x_enc), "— lane i 가 x[i]")

ok, y_enc = infer_reshape_no_reorder(x_enc, (4, 8))
assert ok and y_enc == default_blocked((4, 8))
print("② reshape(4,8) ", fmt(y_enc), "— default → default: 이름만, 명령 0개")
own_y = owner_map(y_enc)
assert all(own_y[(r, c)] == 8 * r + c for r in range(4) for c in range(8))

z_enc = infer_trans(y_enc, (1, 0))
print("③ trans (8,4)  ", fmt(z_enc), "— order 를 뒤집어 이름만, 명령 0개")
own_z = owner_map(z_enc)
assert all(own_z[(r, c)] == 8 * c + r for r in range(8) for c in range(4))
show_grid("   칸 (r,c) 를 드는 lane:", z_enc, lambda rc: own_z[rc])
print()

# ───────────────────────── ④ 여기서 갈린다 ─────────────────────────
ok, why = infer_reshape_no_reorder(z_enc, (32,))
assert not ok
print("④ reshape(32,)  inferReshapeOpNoReorderEncoding →", "실패")
print("   ", why)
print("    → ttg.convert_layout", fmt(z_enc), "→", fmt(default_blocked((8, 4))), "삽입")

# RemoveLayoutConversions 가 이 convert 를 ② 앞으로 밀 수 있나? — trans 를 거꾸로 통과시키면
# (4,8) order=[0,1] 이 되고, 그것을 (32,) 로 합치는 것도 같은 검사에 걸린다.
y_alt = infer_trans(default_blocked((8, 4)), (1, 0))
assert y_alt["shape"] == (4, 8) and y_alt["order"] == (0, 1)
ok_back, _ = infer_reshape_no_reorder(y_alt, (32,))
assert not ok_back
print("    (convert 를 ① 쪽으로 밀어도", fmt(y_alt), "→ (32,) 가 같은 이유로 실패 → 못 없앤다)")
print()

w_enc = default_blocked((8, 4))
after, smem, padded = smem_round_trip(z_enc, w_enc, lane_values)
padded_shape, in_vec, out_vec = scratch_config(z_enc, w_enc)
print("   convert_layout 은 lowerDistributedToDistributed = SMEM 왕복:")
print(f"     st.shared  lane 마다 1개 (inVec={in_vec})  — smem[r][c] ← 자기 값")
print("     bar.sync   1회 (store 와 load 사이)")
print(f"     ld.shared  lane 마다 1개 (outVec={out_vec}) — lane i 가 smem[i//4][i%4]")
print(f"   scratch = {padded_shape[0]}×{padded_shape[1]} (열 {padded_shape[1] - 4} 칸 pad) × {ELEM_BYTES} B"
      f" = {prod(padded_shape) * ELEM_BYTES} B (계산값, f32)")
show_grid("   SMEM 에 놓인 x 인덱스 (pad 열 제외):", z_enc, lambda rc: smem[rc])
own_w = owner_map(w_enc)
show_grid("   ld.shared 뒤 — 칸 (r,c) 를 드는 lane:", w_enc, lambda rc: own_w[rc])
print()

ok, final_enc = infer_reshape_no_reorder(w_enc, (32,))
assert ok and final_enc == default_blocked((32,))
print("   마지막 reshape(32,) ", fmt(final_enc), "— 다시 default → default, 이름만")
print("   결과: lane i 가 든 x 인덱스 =", after)
expected = [8 * (i % 4) + i // 4 for i in range(LANES)]
assert after == expected, (after, expected)
print("        = x[8·(i%4) + i//4]  ✓")
print()

# ───────────────────────── 이후 (LinearLayout) 와 대조 ─────────────────────────
# reshape_transpose_reshape.py: 데이터는 그대로, lane i 가 w[((i&7)<<2) | (i>>3)] 를 든다.
ll_holds_w = [((i & 7) << 2) | (i >> 3) for i in range(LANES)]
# 두 경로가 같은 텐서 w 를 계산하는지 — w[j] = x[8·(j%4) + j//4]
w_of_x = expected  # 옛 경로: lane j = 자리 j 가 x[...] 를 든다 → w[j]
assert all(w_of_x[ll_holds_w[i]] == i for i in range(LANES)), "두 경로의 w 가 다르다"
# 하드웨어 최소 비용 (figures/convert-layout-hw-paths.svg): 옛 경로의 이동은 lane 축에서만 일어나는 순열이다.
# lane j 가 필요로 하는 값의 출발 lane = 8·(j%4) + j//4 — 워프 하나 안이므로 lane 당 shfl.sync.idx 1개면 되고 메모리는 0 B.
src_lane = [8 * (j % 4) + j // 4 for j in range(LANES)]
assert sorted(src_lane) == list(range(LANES)), "순열이 아니다"
assert all(0 <= s < 32 for s in src_lane), "워프 밖으로 나간다"
print("하드웨어 최소 비용: lane j ← lane", src_lane[:8], "… (순열, 워프 안) → shfl.sync.idx 1개/lane, SMEM 0 B (계산값)")
print()
print("LinearLayout 경로: 데이터 이동 없음 — lane i 가 w[(i&7)<<2 | i>>3] 를 든다 (기저 [4,8,16,1,2])")
print("두 경로가 계산한 w 는 같다  ✓  차이는 배치와 비용뿐:")
print("   이전  st.shared 1 + bar.sync 1 + ld.shared 1 (lane 당), SMEM",
      prod(padded_shape) * ELEM_BYTES, "B, 최종 lane 기저 [1,2,4,8,16]")
print("   이후  명령 0개, SMEM 0 B, 최종 lane 기저 [4,8,16,1,2]")
