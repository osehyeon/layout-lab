## Table of Contents
- [Dialect & Frontend](#dialect--frontend)
- [Backend & Compiler](#backend--compiler)
- [AMD/HIP Backend](#amdhip-backend)
- [NVIDIA Backend](#nvidia-backend)
- [Gluon & Layout Improvements](#gluon--layout-improvements)
- [Kernels & Benchmarks](#kernels--benchmarks)
- [Proton Profiling](#proton-profiling)
- [Testing & CI](#testing--ci)
- [Build & Infrastructure](#build--infrastructure)
- [Documentation](#documentation)
- [Breaking Changes](#breaking-changes)
- [Contributors](#contributors)

---

## Dialect & Frontend

### New Features
- **`tl.squeeze` / `tl.unsqueeze`**: Added `tl.squeeze` and `tl.unsqueeze` operations to the standard library (#8924)
- **Scaled BMM**: Added support for scaled batched matmul in the frontend (#9000)
- **FP8 Constants**: Frontend can now create FP8 constants directly (#8882)
- **Returning Constexpr from JIT**: Functions can return `constexpr` values from JIT-compiled code (#8785)
- **`get_int_attr` for Out-of-Tree Walk**: Added `get_int_attr` to `Operation` to support out-of-tree IR walks (#8892)
- **Optional Device Arg to `preload`**: Added optional device argument to `preload` and guardrails for cross-target preload (#8951, #8952, #9234)
- **`tl.cat(can_reorder=False)`**: Added a non-reordering variant of `tl.cat` with broadcast support (#9312, #9163)
- **Round f32→tf32 in Descriptor**: Added option to round f32 to tf32 inside tensor descriptors (#9295)
- **Plugin Hooks & Out-of-Tree Dialects**: Added support for out-of-tree TTIR/TTGIR passes and Triton Dialect Plugins, with example documentation (#8401, #8523, #8815)

### Bug Fixes
- **`desc.shape` for FP4 Padded**: Fixed `desc.shape` values for fp4-padded tensor descriptors (#9012)
- **Setting Attr on Constexpr Argument**: Fixed setting attributes on constexpr arguments (#9053)
- **Named Tuples in Constexpr Functions**: Preserved named tuples through `constexpr_functions` (#8876)
- **`must_use_result` for Methods**: Fixed `must_use_result` check for methods (#8902)
- **`_semantic` Default to None**: Defaulted `_semantic` parameter to None (#8909)
- **`make_tensor_descriptor` Error Typo**: Fixed typo in `make_tensor_descriptor` error message (#8912)
- **`tl.cat` Determinism**: Made `tl.cat` deterministic via permute+reshape+join, then reverted (#9312, #8854, #8878)
- **Deprecation Warning for `make_block_ptr`**: Emitted a deprecation warning when `make_block_ptr` is used (#9667)

### Improvements
- **Frontend Performance**: Pre-computed `inspect.signature` for builtins, lazily computed tuple type names, avoided `find_paths_if` and `inspect.getclosurevars`, removed outdated `catch_warnings` blocks — all to reduce JIT overhead (#8843, #8844, #8846, #8845, #8881)
- **Revert Deep Copy on Scope Entry**: Removed deep copy when entering a new scope (#8832)
- **Default 32-bit Dot Precision Change**: Briefly changed default 32-bit dot precision to TF32x3, then reverted (#9080, #9090)
- **Tutorial Updates** (#8565, #8853, #8982)
- **Interpreter Cleanups**: Typing and efficiency cleanups in the interpreter (#9072)

---

## Backend & Compiler

### LLVM Updates
- **LLVM Bumps**: Multiple LLVM uprevs through the cycle, with one bump reverted on the release branch for stability (#8766, #8840, #8919, #8987, #9264, #9333, #9431, #9942)
- **llvm-head Merge**: Merged changes from llvm-head (#8842)
- **Infinite Rewrite Loop in Latest LLVM**: Fixed an infinite rewrite loop introduced by a newer LLVM revision (#9249)

### 2CTA / Multicast / TMA
- **2CTA Mode End-to-End**: Gluon multi-cta + 2CTA support, M=64 2CTA mode, removed unnecessary synchronization in 2CTA MMA, and proper TMEM deallocation timing (#8684, #8874, #8922, #8986)
- **TMA + Multicast**: Backend support for TMA with multicast (#9005)
- **`tcgen05.mma` + Multicast**: Added multicast support for `tcgen05.mma` (#9071)
- **TMA Index Translation**: Moved TMA index translation from mid-end to lowering (#9082)
- **`tcgen05.mma` Verifier & Errors**: Throw a clear error instead of miscompiling very large `tcgen05.mma` along N (#8915)
- **MMAv5 Illegal Instruction Fix**: Fixed illegal instruction in MMAv5 lowering (#8910)

### Warp Specialization
- **Nested Loops**: Nested-loop support in warp specialization (#8687)
- **Partition Scheduling**: Improved partition scheduling pass; correct stage/cluster annotations for block-arg producers (#7312, #8883)
- **WS Lowering Hardening**: Variable naming fix in `LowerAref`, per-partition `asyncOp` storage, explicit captures to `WarpSpecializePartitionsOp`, skip `InsertTmemAref` when WS isn't used (#8978, #9007, #9023, #9133, #9212)
- **`RegionBranchInterface`**: Made `WarpSpecializePartitionsOp` implement `RegionBranchInterface` (#8799)
- **Mixed TMA / non-TMA Loads**: Fixed AutoWS when mixing TMA and non-TMA loads (#9111)
- **`aref.get` Filtering**: `aref.get` creation now filters results not in the scheduled loop (#9114)
- **Multibuffering Acc Logic**: Improved multibuffering accumulator logic in WS (#8950)

### Code Generation & Analysis
- **`tt.scan` Layout Fixes**: Fixed `tt.scan` with broadcasted layouts and additional scan layout issues (#9185, #9189)
- **Reduce/Scan Verifier**: Verify reduce/scan op axis values (#9061)
- **Pipelined Loops Skip Asserts/Prints**: Loops containing `assert` or `print` are no longer pipelined (#9055)
- **Async Op Semantics**: Added explicit semantics for async ops (#8966)
- **WGMMA Wait Delay**: Delay `wgmma wait(0)` to first use of the accumulator (#9021, #9179)
- **WGMMA Register Pipelining**: Added missing waits in WGMMA RHS register pipelining (#8964, #8970, #8997)
- **WGMMA RS Split Limit**: Limit RS-dot splitting to two splits (#9152)
- **Layout Hoisting Fix**: Fixed handling of conflicting layouts when hoisting convert into conditionals (#9083)
- **Rematerialization Cost**: Consider rematerialisation cost when hoisting over `ext`; improved robustness of ext slice rematerialization (#9194, #9019)
- **AxisInfo Improvements**: Enhanced divisibility handling in `AxisInfo` for add/sub; reland of unvisited-operand handling (#9297, #8758)
- **Layout Picker for Small `async_cp`**: Pick better layouts for small `async_cp` (#9183)
- **Skip Conversion-Backward-Slice Cycle**: Skip values with existing conversions in `getConvertBackwardSlice` (#8291)
- **Membar Improvements**: Consider `memdesc_slice` in Membar; extended membar with third-party ops via traits; AMD-aware `membarFilter` (#8755, #8798, #9265)
- **Reduce Op Lowering**: Improvements to `ReduceOp` lowering, later reverted on release branch (#9192, #9214)
- **Clamp on Scalars**: Support clamp optimization on scalars (#8796)
- **`kReg` smem Padding**: Separated additive `kReg` shared-memory padding contribution (#9286)
- **`tcgen05.mma` + multicast support** and Generalized Encodings: continued generalization of TMEM and shared-memory layouts (#9071)
- **`SwizzledShared` Layout**, **uniform hint** on `ttg.warp_id`, and CGAEncoding rename (#9286, #9073, #8850, #9040, #9125)
- **Pipelining Barrier Location**: Fixed barrier placement in loop lowering for MMA ops with non-pipelined operands (#8732)
- **Properly Async wgmma Loop Detection**: Fixed `dotCanBeProperlyAsync` when wgmma is not yielded by the loop and an associated infinite loop (#9274, #9282)
- **`FuncOpToLLVM` Refactors**: Moved `handleArgPtrDatatype` to `Utility.h`; support for LLVM struct/array types in `DITypeAttr` (#9120, #9124)
- **Cache Robustness**: Handle corrupted on-disk cache (#8923)
- **Async Sentinel**: Added a sentinel when async-compiling (#9251)
- **`JITFunction` in `preload`**: Support `JITFunction` in `preload` (#8794)

### CONSAN (Concurrency Sanitizer) & Debug
- Buffer-region analysis, aliasing support, false-positive deadlock fix, overflow-check disable, compile-time optimization, reduced coverage configurations, TMEM allocation handling, and removal of TMEM size verification (#8837, #8939, #9046, #8940, #9240, #9294, #8787, #8782)
- **Debug Info**: Fixed missing kernel arguments in LLVM debug info; fixed address-sanitizer stack-use-after-scope (#9002, #9088)

---

## AMD/HIP Backend

> 3.7 is heavy on **gfx1250 (RDNA4)** maturation, **warp specialization on AMD**, **Tensor Data Movement (TDM)**, and a new **warp-pipeline** path.

### Warp Specialization & Warp Pipelining on AMD
- **Warp-Pipeline Support**: New AMD warp-pipeline path with Gluon and LLVM lowering (#8586, #8975, #8980)
- **Warp Specialization on gfx1250** (#8947, #8968)
- **Warp-Pipeline Fixes**: Priority hints and Gluon fixes for the new pipeline (#9301)
- **`ttg.warp_id` and AMD Conversions** (#8659)

### gfx1250 / RDNA4 Maturation
- **Mixed-Precision Scaled Dot**: Enabled mixed-precision (scaled) dot in Triton on gfx1250 (#8938)
- **4-Warp / 8-Warp MXFP GEMM**: 4-warp scheduling and 8-warp pingpong + MXGEMM refactor (#9031, #9356)
- **Persistent WS f16 GEMM**: Persistent variant and persistent subtiled variant for WS f16 GEMM (#8990, #9052)
- **F16 GEMM Examples Updates**: Updated MXFP FA example and f16 GEMM examples (#9326, #8972)
- **Buffer Atomics for RDNA4**: Enabled buffer atomics on RDNA4 (#8778)
- **`v_permlane16_swap`**: Enabled for `convert_layout` and `reduceOp` on GFX1250 (#8724)
- **Extended FP Conversion**: Including RTZ rounding fixes for GFX1250 (#8821, #8965)
- **libdevice for ROCm 7.1**: Updated libdevice bitcode files (#8807)
- **Cluster Loads / Multi-CTA**: Multi-CTA GEMM example for gfx1250, multi-CTA support for `AMDWmmaEncodingAttr`, scalar-pointer cluster-load avoidance (#9342, #9340, #9129)
- **Gluon `AMDWMMALayout` Rank Consistency** (#9127)
- **WMMA Database Additions**: Added `i8xi8xi32` v3, missing `f64.16x16x4.f64`, and clamp operand on WMMA int intrinsic (#9267, #9271, #9291, #9359)
- **Wavefront Scheduling**: Fixed waitcnt for gfx1250 (#8835)
- **Gluon Stream-K**: 4- and 8-warp stream-k Gluon kernels for gfx1250 (#9370)
- **Roll-up Updates**: Bundled small gfx1250 fixes (#9365)

### Tensor Data Movement (TDM)
- **Multi-CTA & Multicast for TDM** (#8790)
- **Host-Side TDM Descriptor**: 1D-5D support on gfx1250 (#8977)
- **TDM L2 Prefetch**: Backend and Gluon exposure (#9086, #9148)
- **TDM Predicate**: Use TDM predicate in f16 GEMM variants (#9054)
- **TDM Async Wait**: Support TDM `AsyncWait` in `UpdateAsyncWaitCount` (#9352)
- **TDM Padding in Store**: Support padding when interval equals the inner dimension (#9360)
- **TDM Async Scatter/Gather**: Tensor async scatter/gather support and fixed OOB handling (#9299, #9313, #9371)
- **TDM Shape Adjustment**: Account for CGA offset in TDM shape adjustment (#9341)
- **4D+ TDM Bug Fix**: Fixed TDM behavior when `dim > 2` (#8994)
- **Some TDM Features Enabled** (#9283)

### Async Copy / LDS
- **AsyncCopy Default On**: Enabled `AsyncCopy` by default for gfx950 and gfx1250 — later reverted on release/3.7.x (#9445, #9087)
- **Async Copy Block Dim Duplication**: Allow async load global-to-load block-dim duplication (#8788)
- **Direct-to-LDS Refactors**: Fixed shared-order selection on GFX9, refactored coalescing checks, contiguity hints, vector-size fixes for padded encodings (#9028, #9041, #9048, #9089, #9149)
- **`v_perm` for `convert_layout`** (#9014)
- **Padded Layout Heuristic**: Relaxed heuristics for smaller block sizes (#9074)

### Reorder / Pipelining Cleanup
- **`ReorderInstructions`**: Removed `sinkSecondLoad`, `sinkDotConversion`, and `moveUpTranspose` optimizations (#9119, #9139, #9204, #9229)
- **Replace `ReorderInstructions` with `MoveUpPrologueLoads`** (#9328)
- **`UpdateAsyncWaitCount`**: Support single-block `execute` regions (#9126)
- **`OptimizeLDSUsage` Removal** (#8282)

### libdevice / Layouts / Misc
- **`finite`/`isfinited`**, **`rint`**, **`clampf` via `v_med3`**: libdevice and codegen additions (#9097, #9166, #9256)
- **`BlockPingpong` Improvements**: Debug messages and dot-dominates-predecessors fix (#8804, #9027)
- **`kWidth` mandatory for WMMA v3** (#8783)
- **`copysign` Replacement**: Replaced LLVM `copysign` intrinsic (#8789)
- **WMMA Layout CTA Fields**: Generalized (#8946)
- **TDM with `CanonicalizePointers`**: Support `MakeTensorDescOp` in `CanonicalizePointers` (#9228)
- **`PartitionedSharedEncodingAttr`**: Introduced and reverted (#9314, #9367)
- **`scf.if` Combining**: Added `PrepareIfCombining` pass (#9253)
- **Fine-Grained Cluster Barrier**: New AMD cluster barrier exposed to Gluon (#9206)
- **`SinkLayoutConversions` Pass** (#9168)
- **MIR Swap**: Option to swap MIR; `addOccurrence` for proper LLVM-option disabling; `ScopedNoAliasAAWrapperPass` in MIR swap pipeline (#8711, #9311, #9309)

### AMD Bug Fixes (selected)
- **`atomic_cas` Fixes**: Wrong struct index for atomic-CAS pattern, ignored sem/scope, and atomic-CAS for non-int types (#8867, #9042, #9116)
- **Atomic-RMW Mask Vectorization**: Fixed wrong vectorization width for masked atomic-RMW (#9142)
- **BroadcastedRegisters in Compilation**: Fixed compilation crash (#8828)
- **`uniformSum` Crash**: Fixed null `uniformSum` in `CanonicalizePointers` (#8991)
- **Cooperative Groups Support**: Driver check (#8935)
- **FP8/BF8 WMMA Selection** on release/3.7.x: Fixed mixed FP8 promotion / instruction selection (#9567, #9581)
- **True16 on gfx11**: Disabled True16 for assembler on gfx11 (#9447, #9476)
- **`RangeAnalysis` `tripCount`**: Fixed trip-count calculation (#9383, #9944)
- **Padded-Layout Async Copy OOM**: Fixed OOM in pipelining with padded async copy on GFX950 (#9442, #9945)
- **`BlockPingpong` for non-MFMA dot** (#9618, #9948)
- **`CanonicalizePointers` Different Bases** (#9541, #9950)
- **Backend cherry-pick dance** (#9487, #9502, #9673, #9675)
- **FP4 Matmul Tests Skipping**: Skip tests packed along M/N for gfx1250 (#9176)

---

## NVIDIA Backend

### Blackwell & Newer SMs
- **`tcgen05` MMA on sm110 (Jetson Thor)** (#9160)
- **`tcgen05.ld.red` on sm103**: Implemented in Gluon (#9151)
- **x Scale Swizzling for Blackwell + Batched Matmul** (#8863)
- **Block-Scaled Matmul Baselining**: mxfp8/nvfp4 block-scaled cuBLAS baselines (#9044)
- **ptxas for Blackwell**: Repeated ptxas-version uprev/revert; final state on release/3.7.x cherry-picks the GB300/Spark/THOR-required commits (#8941, #9011, #9016, #8983, #9363, #9621)
- **NVMMA Variadic CUDA Launcher**: Variadic-argument pre-compiled CUDA launcher (#6788)
- **`NVIDIA::canSkipBarSync`**: Resurrected (#9246)

### TMA
- **TMA im2col Mode**: End-to-end im2col TMA support — `AsyncTMACopyGlobalToLocalOp`, tensor-descriptor support, fix for `tma load`, and driver support (#9202, #9225, #9303, #9305)
- **TMA Encoding Verification**: Verify encodings on TMA ops (#8886)
- **TMA Descriptor Mitigation**: Mitigation against potential TMA descriptor creation errors (#9235)

### Hopper / WS
- **`tt.split`/`join` in WS Data Partition**: Hopper WS support for `tt.split`/`tt.join` (#456, #9147)
- **mx8 `w_scale` Mask**: Fixed Hopper mask (#8974)
- **Small-Batch Hopper**: Bench fixes for small batches on Hopper (#8877)
- **SM89 ptxas Workaround Reverted**: Removed the older workaround for the SM89 ptxas bug now that it is unnecessary (#9756)

---

## Gluon & Layout Improvements

### New Features
- **Local Scatter/Gather**: Added local scatter/gather support to Gluon (#8480)
- **`get_view()`**: Added `get_view()` for Gluon layouts (#9270)
- **Finer Cluster Fences**: Exposed finer-grained cluster fences (#9076)
- **Multi-CTA Refactor of `PaddedSharedLayouts`** (#9336)
- **"Illegal Instruction" Sanitize Mode**: Tightened TMA op verifiers and added an "illegal instruction" sanitize mode (#9112)
- **Verifier Improvements**: Tightened Gluon dialect verifiers and moved checks into C++ (#8981, #9018, #9033)
- **TensorMemory in `to_linear_layout`**: Allow TM layouts in `to_linear_layout` for printing (#8682)
- **More Blackwell Tutorials** (#8982)

### Layouts & Shared Encodings
- **LinearEncoding Tightening**: Tightened LinearEncoding checks (#9215)
- **`SharedLinearEncoding`**: Continued lowering generalization (carry-over from 3.6 with backend updates).

---

## Kernels & Benchmarks

### Persistent Matmul
- **Persistent Matmul Heuristics**: Fixed and refined heuristics (#8791, #8813)
- **Hopper HBM Swizzling**: Persistent matmul now supports Hopper HBM swizzling (#8917)
- **Hopper FP4 Swizzled, num_warps=4** (#9029)
- **Don't Flatten Mixed-Precision Hopper Persistent Matmul** (#9279)
- **High-Occupancy Persistent Matmul**: Re-enabled (#9248)
- **4-Warp Persistent Kernel**: Re-enabled after fixes (#9331)
- **Strided Layout Handling for Persistent**: Fixed when setting `requires_persistent` (#9198)
- **Mxfp Non-Persistent Strided Layout**: Allow non-persistent mx matmul with strided layout (#8808)

### Triton Kernels Refactor
- **Matrix-Multiplication Refactor**: Major refactor of triton_kernels matmul (#8765)
- **Tensor/Layout/Distributed Refactor**: Reland of the tensor/layout/distributed refactor; small follow-ups (#9134, #9140, #9186, #9187, #9213)
- **Closure-Based Output Mapping**: For peer shards (#8999)
- **Distributed Tests**: Distributed routing kernels test fix (#9258)
- **Device Descriptor Allocator**: Keep a pool to fix descriptor allocator behavior (#9259)
- **Reduce Kernel**: Unfuse FMA for numeric stability, unpadded batch handling, global scale (#9320, #9332, #9372)
- **`Tensor.clone`**: Briefly added `clone` for `triton_kernels.tensor.Tensor`, then reverted (#9178, #9208)

### MXFP / Scaled-Dot Kernels
- **Force `mxfp4→bf16` Conversion via `mul.bf16x2`** (#8967)
- **Hopper mxfp4 Swizzled, num_warps=4** (#9029)
- **swiglu Optimizations**: Save instructions, then partial revert; later use of `ex2.approx.ftz` for swiglu (#8801, #8905, #9164)
- **matmul Output mxfp Format Fixes** (#8865)
- **Symmetric Memory in Bench**: Release symmetric memory between runs (#8900)
- **`distributed.py` / `bench_utils.py`**: Extracted common code from `bench_mlp.py` and `distributed.py` (#8866)
- **`num_stages` Adjustment**: For bf16/fp16 × mxfp (#8773)

### Other
- **X Scale Swizzling for Ragged** (#8897)
- **`reduce_forward` Metadata**: Improved performance (#9068)
- **TF32 Rounding in MoE** (#9296)
- **`p_matmul` Asserts & Fixes** (#9376)
- **Distributed `symm_mem_pool` by Argument** (#9092, #9155)

---

## Proton Profiling

### Highlights
- **Hardware Trace on Blackwell**: Enabled low-overhead hardware trace (#9307)
- **Significant `deactivate` / `get_data` Overhead Reduction**: Especially for CUDA-graph profiling; exposed `get_data_msgpack` (#9030)
- **Periodic Dumping**: Periodic profile dumping; metadata profiling with periodic flushing (#9150, #9236)
- **Multi-Device Metric Profiling**: Fixed metric buffer deadlock and added multi-device support (#8943)
- **Capture-on-Error**: Capture traces even when code exits with an error (#8955)
- **Vector Metrics**: New vector metric type (#9329)

### API & Internals
- **`get_data` API**: Export profile data directly in Python (#8928)
- **`clear_data` API**: Remove pre-deactivation data (#8971)
- **`finalize` Cleanup**: Clean up context source after teardown (#9069)
- **Fewer Locks**: Further reduce unnecessary locks (#9257)
- **Runtime/Metric Correlation**: Simplified to reduce overhead (#9132)
- **Selective Kernel Metadata**: Allow Proton to record metadata for selective kernels (#9158)
- **Metric Type Restrictions**: Restrict frontend metric types (#8858)
- **Init/Final Timestamps**: Added to Chrome trace (#8870)
- **`GlobalScratchAllocOp` Deprecation**: Deprecated Proton's own op in favor of TritonGPU's, with a custom backend (#8976)
- **Drop Invalid-Time Kernels** (#8961)
- **Ignore Metric-Kernel Timing** (#9058)
- **Documented Experimental APIs** (#9056)
- **HW Trace Default Fix**: Fixed default value for `TRITON_ENABLE_HW_TRACE` in `CuptiProfiler` (#9324)
- **Tensor Descriptor & 2-CTA Tests** (#9070)
- **AMD Proton Test Fixes** (#8763)

---

## Testing & CI
- **Gluon TMA + MMA Hopper/Blackwell Test** (#8873)
- **AMD Shadow CI**: New AMD runner setup, then reverted (#9032, #9049)
- **`fresh_knobs` Default Behavior** (#9184)
- **`tl.dot` BF16xN Nondeterminism** (#8818)
- **Disable Stack Traces in Performance Remarks** (#8884)
- **Pin pandas < 3.0** (#9273)
- **Fix pytorch Deprecation Warning in CI** (#8857)
- **Reduce Wheel Size, Pin `DOCKER_API_VERSION` (release/3.7.x)** (#10244)
- **Increase Release Wheel Timeout** (#10250)
- **Skip Tests for RDNA / gfx1250**: Various AMD test skips and enables (#9210, #9176, #9177, #9232, #9343, #9095)
- **Triton's `assert_close`**: Propagate `err_msg` to numpy (#9170)
- **Float8 × MX Tolerance** (#9316, #9338)
- **NumPy 2.4 Compatibility**: Explicit numpy-array-to-scalar conversion (#9172)
- **`test_line_info_ir_source` Flake Fix** (#9161)

---

## Build & Infrastructure
- **`CMAKE_LIBRARY_OUTPUT_DIRECTORY`**: Fixed build with empty directory (#8810)
- **`llvm_update_compile_flags` Removal** (#9167)
- **`LLVM_BUILD_SHARED_LIBS` Canonicalization** (#8933)
- **actions/checkout v5 → v6** (#8826)
- **Version Bumps**: 3.5.0 → 3.6.0 ; 3.6.0 → 3.7.0 (#8836, #9885, #9888)
- **`TRITON_EXT_ENABLED` for Wheels** (#9935, #9959)
- **`nvidia-toolchain-version.json` Update**.
- **`TRITON_DEFAULT_BACKEND`**: Control `driver.active` via this env var (#9144)
- **`TRITON_PTXAS_BLACKWELL_PATH`**: Allow override of `ptxas-blackwell` binary (#8945)
- **Release to PyPI** (#10251)
- **`topk` in Plugin Example**: Increment index in plugin example (#9315)
- **HIP Support in `link.py`** (#9084)

---

## Documentation
- **Divisibility Reset Logic**: Clarified for contiguous dimensions in `AxisInfo` (#9266)
- **`topk` Operation**: Added to language documentation (#9345)
- **Plugin Example README**: Added a second pass-plugin README example (#8815)
- **Conference Materials**: Updated README (#9009)
- **Community Meetup Notes**: Added 2026-01-06 meetup notes (#9288)
- **`warp_specialize` Docs**: Updated `gl.warp_specialize` docs (#8553)
- **`LinearLayout` Output Matrix Comment**: Doc fix (#9243)

---

## Breaking Changes
- **`triton_kernels` matmul refactor (BC-breaking)**: The matrix-multiplication refactor introduces a backwards-incompatible API surface. Downstream users of `triton_kernels.matmul_*` should review call sites (#8765)
- **`tcgen05.cp` Lowering Generalization & `tcgen05.mma` Encoding Acceptance**: Continued from 3.6, with new verifier behavior and stricter encoding checks.
- **Proton `GlobalScratchAllocOp` Deprecated**: Replaced with TritonGPU's `GlobalScratchAllocOp` + custom backend. Out-of-tree consumers must migrate (#8976)
- **`make_block_ptr` Deprecated**: A deprecation warning is now emitted; users should migrate to tensor descriptors (#9667)
- **Default 32-bit Dot Precision Reverted**: Default 32-bit dot precision was briefly TF32x3 — the default in 3.7 remains as in 3.6. Note the new "round f32→tf32 in descriptor" option (#9080, #9090, #9295)
- **AsyncCopy Default for gfx950 / gfx1250**: Was enabled by default and then reverted on the release branch. Users must opt in explicitly in 3.7 (#9087, #9445)
- **SM89 ptxas Workaround Reverted**: The ptxas workaround introduced earlier is removed on release/3.7.x (#9756, #7067)

---

## Contributors

This release includes contributions from engineers at:

- Meta
- AMD
- NVIDIA
- OpenAI
- Intel
- Google
- And many individual contributors

Special thanks to all contributors who submitted bug reports, feature requests, and code improvements!


