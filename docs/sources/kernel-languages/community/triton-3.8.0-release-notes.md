# Triton 3.8.0 Release Notes

## Table of Contents

- [Dialect & Frontend](#dialect--frontend-38)
- [Backend & Compiler](#backend--compiler-38)
- [AMD/HIP Backend](#amdhip-backend-38)
- [NVIDIA Backend](#nvidia-backend-38)
- [Gluon & Layout Improvements](#gluon--layout-improvements-38)
- [Kernels & Benchmarks](#kernels--benchmarks-38)
- [Proton Profiling](#proton-profiling-38)
- [Testing & CI](#testing--ci-38)
- [Build & Infrastructure](#build--infrastructure-38)
- [Documentation](#documentation-38)
- [Breaking Changes](#breaking-changes-38)
- [Contributors](#contributors-38)

---

## Dialect & Frontend<a id="dialect--frontend-38"></a>

### New Features

- **Aggregate types:** `@triton.aggregate` and `@gluon.aggregate` are now public APIs. Aggregates support inherited fields, default values, generated constructors, immutable instances, and `aggregate_replace()` ([#10095](https://github.com/triton-lang/triton/pull/10095), [#9572](https://github.com/triton-lang/triton/pull/9572))
- **`tl.topk`:** Added a `descending` argument. Set `descending=False` to return the smallest values ([#9355](https://github.com/triton-lang/triton/pull/9355))
- **Tensor descriptors:** Tensor descriptors can be passed inside tuple-valued kernel arguments ([#9422](https://github.com/triton-lang/triton/pull/9422))
- **Interpreter:** Added support for `tl.dot_scaled` ([#10311](https://github.com/triton-lang/triton/pull/10311))

### Improvements

- **Autotuning listener:** Added a listener that reports the selected configuration, measured timings, tuning duration, and disk-cache status ([#10125](https://github.com/triton-lang/triton/pull/10125))
- **JIT cache keys:** Dependency cache keys are now generated deterministically ([#10494](https://github.com/triton-lang/triton/pull/10494))

### Bug Fixes

- **Division and atomics:** `tl.fdiv(..., ieee_rounding=True)` now emits IEEE-rounded division, and floating-point `atomic_min` now returns a value-typed result ([#10074](https://github.com/triton-lang/triton/pull/10074), [#10485](https://github.com/triton-lang/triton/pull/10485))
- **Interpreter NaN handling:** `argmin`, `argmax`, `minimum`, `maximum`, and `clamp` now match compiled behavior more closely when inputs contain NaNs ([#10298](https://github.com/triton-lang/triton/pull/10298), [#10333](https://github.com/triton-lang/triton/pull/10333), [#10699](https://github.com/triton-lang/triton/pull/10699))
- **Block-pointer padding:** Fixed zero padding for block-pointer loads ([#11252](https://github.com/triton-lang/triton/pull/11252))
- **Python 3.14 annotations:** Updated annotation handling for PEP 649, including aggregate field discovery ([#10581](https://github.com/triton-lang/triton/pull/10581))

---

## Backend & Compiler<a id="backend--compiler-38"></a>

### LLVM Updates

- **Correctness fixes:** Updated the pinned LLVM revision with fixes for a GFX950 BF16 miscompilation and SLP-vectorizer issues ([#10719](https://github.com/triton-lang/triton/pull/10719), [#11356](https://github.com/triton-lang/triton/pull/11356))

### Multi-CTA / Multicast / TMA

- **Generic multi-CTA support:** Extended multi-CTA support to layout conversion, reductions, local gather/scatter, TMA gather/scatter, and multicast, with corresponding updates to barrier insertion and memory analysis ([#9317](https://github.com/triton-lang/triton/pull/9317), [#9221](https://github.com/triton-lang/triton/pull/9221), [#9977](https://github.com/triton-lang/triton/pull/9977), [#10472](https://github.com/triton-lang/triton/pull/10472), [#9318](https://github.com/triton-lang/triton/pull/9318), [#9615](https://github.com/triton-lang/triton/pull/9615))
- **TMA store waits:** `tma.store_wait` now accepts a `read_only` argument. The default remains `True`; use `read_only=False` when the store must reach global memory before a release operation ([#10415](https://github.com/triton-lang/triton/pull/10415), [#10419](https://github.com/triton-lang/triton/pull/10419))

### Code Generation & Analysis

- **Layout rematerialization:** Fixed stale rematerialized values and missing rematerialization mappings in `RemoveLayoutConversions` ([#10646](https://github.com/triton-lang/triton/pull/10646), [#11029](https://github.com/triton-lang/triton/pull/11029))

### Sanitizers & Debugging

- **FpSan:** Added compiler instrumentation for checking whether kernel variants preserve the same symbolic floating-point computation. FpSan supports NVIDIA targets and AMD gfx942, gfx950, and gfx1250, including dot, scaled-dot, WGMMA, and MMAv5 paths, and adds `tl.expect_zero` ([#9337](https://github.com/triton-lang/triton/pull/9337), [#9455](https://github.com/triton-lang/triton/pull/9455), [#9714](https://github.com/triton-lang/triton/pull/9714), [#10112](https://github.com/triton-lang/triton/pull/10112), [#10330](https://github.com/triton-lang/triton/pull/10330), [#10461](https://github.com/triton-lang/triton/pull/10461))
- **GSan:** Added an experimental detector for data races in memory managed by the GSan allocator. It covers loads and stores, atomics, selected asynchronous operations, symmetric memory, and multi-node topologies ([#9478](https://github.com/triton-lang/triton/pull/9478), [#9568](https://github.com/triton-lang/triton/pull/9568), [#9699](https://github.com/triton-lang/triton/pull/9699), [#9700](https://github.com/triton-lang/triton/pull/9700), [#9493](https://github.com/triton-lang/triton/pull/9493), [#10577](https://github.com/triton-lang/triton/pull/10577))
- **ConSan:** Added AMD support and broader coverage for multi-CTA kernels, barriers, TMA, multicast, Cluster Launch Control (CLC), and barrier reinitialization errors ([#9692](https://github.com/triton-lang/triton/pull/9692), [#9843](https://github.com/triton-lang/triton/pull/9843), [#9934](https://github.com/triton-lang/triton/pull/9934), [#10052](https://github.com/triton-lang/triton/pull/10052), [#9591](https://github.com/triton-lang/triton/pull/9591))
- **Scratch allocation:** FpSan and ConSan now use a driver-provided default allocator for global scratch, removing the custom-allocator requirement ([#9596](https://github.com/triton-lang/triton/pull/9596))

### Extensions & Tooling

- **Out-of-tree extensions:** Extensions can now define custom Python DSL operations, inspect additional MLIR value properties, pass string arguments to registered passes, extend AxisInfo analysis, and check plugin versions ([#9626](https://github.com/triton-lang/triton/pull/9626), [#9866](https://github.com/triton-lang/triton/pull/9866), [#9691](https://github.com/triton-lang/triton/pull/9691), [#9736](https://github.com/triton-lang/triton/pull/9736), [#9937](https://github.com/triton-lang/triton/pull/9937))

---

## AMD/HIP Backend<a id="amdhip-backend-38"></a>

### gfx1250 / CDNA 5

- **Tensor Data Movement:** Expanded gfx1250 support for TDM software pipelining, descriptor gather/scatter, multi-CTA and multicast transfers, partitioned shared-memory layouts, and descriptor updates ([#9302](https://github.com/triton-lang/triton/pull/9302), [#10157](https://github.com/triton-lang/triton/pull/10157), [#10674](https://github.com/triton-lang/triton/pull/10674), [#9374](https://github.com/triton-lang/triton/pull/9374), [#10225](https://github.com/triton-lang/triton/pull/10225))
- **WMMA and atomics:** Added scaled WMMA 32x16 variants, FP32 WMMA support, scale-factor-16 E4M3 support for scaled dot, hardware floating-point upcasts, and buffer atomics ([#10082](https://github.com/triton-lang/triton/pull/10082), [#9886](https://github.com/triton-lang/triton/pull/9886), [#9561](https://github.com/triton-lang/triton/pull/9561), [#9449](https://github.com/triton-lang/triton/pull/9449), [#9744](https://github.com/triton-lang/triton/pull/9744))
- **Warp pipelining:** Added flat and back-to-back warp-pipeline support and enabled loop unrolling for Gluon warp-pipelined kernels ([#9929](https://github.com/triton-lang/triton/pull/9929), [#9666](https://github.com/triton-lang/triton/pull/9666))
- **CDNA5 target name:** Gluon exposes `cdna5` as an alias for the gfx1250 target ([#11383](https://github.com/triton-lang/triton/pull/11383))

### Other Targets

- **GCN 5.1:** Added AMD backend target support for `gfx906` ([#9628](https://github.com/triton-lang/triton/pull/9628))
- **In-thread transpose:** Enabled in-thread transpose on RDNA 3 and RDNA 3.5 and enabled it by default on RDNA 4 (`gfx120x`) targets ([#10390](https://github.com/triton-lang/triton/pull/10390), [#10185](https://github.com/triton-lang/triton/pull/10185))
- **HIP helpers:** Added `num_threads`, `num_warps`, and `smid` to `tl.extra.hip`, and added `clz` and `popc` to the HIP libdevice ([#9604](https://github.com/triton-lang/triton/pull/9604), [#10651](https://github.com/triton-lang/triton/pull/10651))

### Bug Fixes

- **Code generation:** Fixed direct-to-LDS and buffer-load paths, refined gfx1250 scheduling controls and defaults, and corrected i32 accumulation for small-K int8 dot products ([#10928](https://github.com/triton-lang/triton/pull/10928), [#10635](https://github.com/triton-lang/triton/pull/10635), [#11256](https://github.com/triton-lang/triton/pull/11256), [#10721](https://github.com/triton-lang/triton/pull/10721), [#11282](https://github.com/triton-lang/triton/pull/11282))

---

## NVIDIA Backend<a id="nvidia-backend-38"></a>

### Rubin

- **Rubin:** Added initial NVIDIA Rubin (SM107) support, including MMA updates, multicast barrier arrival, and a Rubin-specific Gluon module ([#10936](https://github.com/triton-lang/triton/pull/10936), [#10941](https://github.com/triton-lang/triton/pull/10941), [#10953](https://github.com/triton-lang/triton/pull/10953))
- **Packed arithmetic:** Added four-lane FP8 and FP4 operations for Rubin, including `add4`, `sub4`, `mul4`, and `fma4` ([#11084](https://github.com/triton-lang/triton/pull/11084))

### Matrix Instructions

- **MMAv5 and scaled dot:** Added int8 MMAv5 support and native block-scaled dot on SM121 ([#8463](https://github.com/triton-lang/triton/pull/8463), [#10010](https://github.com/triton-lang/triton/pull/10010))
- **FP64 and TF32:** Added FP64 matrix-multiply support on Blackwell, native 8x8x4 FP64 MMAv2 tiles, and N=8/K=8 TF32 tiles ([#10520](https://github.com/triton-lang/triton/pull/10520), [#10060](https://github.com/triton-lang/triton/pull/10060), [#10234](https://github.com/triton-lang/triton/pull/10234))

### TMA / Cluster Launch Control

- **Cluster Launch Control:** Blackwell Gluon kernels can use Cluster Launch Control (CLC) for dynamic work distribution in persistent kernels ([#9361](https://github.com/triton-lang/triton/pull/9361))
- **TMA im2col:** Added TMA im2col support to the NVIDIA backend and Gluon API, with a convolution tutorial ([#9322](https://github.com/triton-lang/triton/pull/9322), [#9391](https://github.com/triton-lang/triton/pull/9391), [#9406](https://github.com/triton-lang/triton/pull/9406))
- **TMA and TMEM layouts:** Added swizzle-zero TMA+MMA support on Hopper and Blackwell and combined TMEM loads with row reductions on SM103+ ([#10148](https://github.com/triton-lang/triton/pull/10148), [#10551](https://github.com/triton-lang/triton/pull/10551))

### Bug Fixes

- **Synchronization and layout lowering:** Fixed Hopper WGMMA synchronization, Blackwell load-wait placement, multi-CTA atomics, and cluster-barrier lowering in warp-specialized regions ([#9514](https://github.com/triton-lang/triton/pull/9514), [#9636](https://github.com/triton-lang/triton/pull/9636), [#10477](https://github.com/triton-lang/triton/pull/10477), [#10992](https://github.com/triton-lang/triton/pull/10992))
- **SM90 BF16 reductions:** Added a workaround for incorrect reduction vectorization ([#10779](https://github.com/triton-lang/triton/pull/10779))

---

## Gluon & Layout Improvements<a id="gluon--layout-improvements-38"></a>

### New Features

- **Target APIs:** Gluon added 3D dot FMA, shared-memory atomic add, generalized local atomic scatter operations, TMA atomics, NVIDIA asynchronous local stores, and AMD scaled-upcast operations ([#9501](https://github.com/triton-lang/triton/pull/9501), [#10100](https://github.com/triton-lang/triton/pull/10100), [#10183](https://github.com/triton-lang/triton/pull/10183), [#10040](https://github.com/triton-lang/triton/pull/10040), [#10357](https://github.com/triton-lang/triton/pull/10357), [#10111](https://github.com/triton-lang/triton/pull/10111))
- **Generic linear layouts:** Added `GenericLinearEncodingAttr` for supported swizzled and non-injective layouts, including local load/store lowering and permutation-matrix inference ([#9765](https://github.com/triton-lang/triton/pull/9765), [#10122](https://github.com/triton-lang/triton/pull/10122), [#10515](https://github.com/triton-lang/triton/pull/10515))
- **JIT functions:** Gluon now supports variadic JIT functions and exposes `GluonJITFunction` ([#9863](https://github.com/triton-lang/triton/pull/9863))
- **Triton-to-Gluon translator:** Reworked the experimental translator and added AMD and Hopper target support ([#9570](https://github.com/triton-lang/triton/pull/9570), [#9717](https://github.com/triton-lang/triton/pull/9717), [#10089](https://github.com/triton-lang/triton/pull/10089))
- **Async-copy APIs:** Added the shorter `async_load` and `async_store` names; the previous `async_copy_*` names remain available as aliases ([#10083](https://github.com/triton-lang/triton/pull/10083))

### Examples

- **New examples and tutorials:** Added coverage for Cluster Launch Control, TMA im2col convolution, multi-CTA kernels, two-CTA block-scaled matmul, and mixture-of-experts kernels ([#9361](https://github.com/triton-lang/triton/pull/9361), [#9406](https://github.com/triton-lang/triton/pull/9406), [#9654](https://github.com/triton-lang/triton/pull/9654), [#9697](https://github.com/triton-lang/triton/pull/9697), [#10047](https://github.com/triton-lang/triton/pull/10047), [#10204](https://github.com/triton-lang/triton/pull/10204))

---

## Kernels & Benchmarks<a id="kernels--benchmarks-38"></a>

### Low-Precision Matmul

- **New input combinations:** Added NVFP4-by-NVFP4 and MXFP4-by-MXFP4 inputs, tensor-valued scales and scaled NVFP4 outputs, MXFP8 activations with Hopper-swizzled MXFP4 weights, and microscaled activations with dense FP16/BF16 weights ([#9745](https://github.com/triton-lang/triton/pull/9745), [#10650](https://github.com/triton-lang/triton/pull/10650), [#9854](https://github.com/triton-lang/triton/pull/9854), [#10214](https://github.com/triton-lang/triton/pull/10214), [#10316](https://github.com/triton-lang/triton/pull/10316))
- **Split-K accumulation:** Added a configurable intermediate dtype for split-K matmul scratch outputs ([#10236](https://github.com/triton-lang/triton/pull/10236))

### Other Matmul Updates

- **FP32 and FP64:** Persistent matmul supports FP32 inputs, and `triton_kernels` adds an FP64 matmul path ([#9393](https://github.com/triton-lang/triton/pull/9393), [#10634](https://github.com/triton-lang/triton/pull/10634))

### Layouts and Empty Shapes

- **Storage shape queries:** Added `Layout.storage_shape()` for querying a layout's physical storage shape without materializing a conversion ([#10554](https://github.com/triton-lang/triton/pull/10554))
- **Empty tensors:** Matmul, reduction, and MX layout conversion now handle zero-sized dimensions and empty custom-layout outputs ([#10427](https://github.com/triton-lang/triton/pull/10427), [#10432](https://github.com/triton-lang/triton/pull/10432), [#10463](https://github.com/triton-lang/triton/pull/10463), [#10464](https://github.com/triton-lang/triton/pull/10464))

### Performance

- **Matmul and reduction tuning:** Improved memory-bound MX4 MoE kernels, small-batch MXFP4 split-K matmuls, broadcast-masked reductions, and Blackwell MX scale swizzling ([#9698](https://github.com/triton-lang/triton/pull/9698), [#9980](https://github.com/triton-lang/triton/pull/9980), [#10317](https://github.com/triton-lang/triton/pull/10317), [#10361](https://github.com/triton-lang/triton/pull/10361), [#10491](https://github.com/triton-lang/triton/pull/10491))

---

## Proton Profiling<a id="proton-profiling-38"></a>

### Highlights

- **CUDA graph profiling:** Reduced launch and serialization overhead, retained graph executables across replays, and added graph scopes and metadata to traces ([#9405](https://github.com/triton-lang/triton/pull/9405), [#9768](https://github.com/triton-lang/triton/pull/9768), [#9930](https://github.com/triton-lang/triton/pull/9930), [#10326](https://github.com/triton-lang/triton/pull/10326), [#10393](https://github.com/triton-lang/triton/pull/10393), [#10394](https://github.com/triton-lang/triton/pull/10394), [#10395](https://github.com/triton-lang/triton/pull/10395), [#10396](https://github.com/triton-lang/triton/pull/10396), [#10397](https://github.com/triton-lang/triton/pull/10397))
- **Benchmark helpers:** Added `do_bench_proton` and `do_bench_cudagraph_proton` to reduce benchmark bias from CPU launch overhead and an unflushed L2 cache ([#10149](https://github.com/triton-lang/triton/pull/10149))

### API & Extensions

- **Metric buffers:** Added a setting for the Proton metric-buffer size ([#9981](https://github.com/triton-lang/triton/pull/9981))
- **ROCm profiler:** Migrated the ROCm profiler from `roctracer` to `rocprofiler-sdk` ([#9704](https://github.com/triton-lang/triton/pull/9704))
- **Out-of-tree backends:** Backends can register Proton devices, runtimes, and profilers ([#10246](https://github.com/triton-lang/triton/pull/10246))
- **Trace organization:** Fixed multi-stream tracing and grouped metadata helper kernels under their owning Triton operator ([#9796](https://github.com/triton-lang/triton/pull/9796), [#10271](https://github.com/triton-lang/triton/pull/10271))

---

## Testing & CI<a id="testing--ci-38"></a>

- **Test coverage:** Added tests for cross-CTA local loads and stores and two-CTA `tcgen05` code generation. CI now runs the full test suite ([#10344](https://github.com/triton-lang/triton/pull/10344), [#10345](https://github.com/triton-lang/triton/pull/10345), [#10027](https://github.com/triton-lang/triton/pull/10027))
- **Tutorial validation:** Gluon tutorials now run in CI ([#10565](https://github.com/triton-lang/triton/pull/10565))

---

## Build & Infrastructure<a id="build--infrastructure-38"></a>

- **Standalone CUDA backend:** The CUDA backend can run without PyTorch installed ([#9578](https://github.com/triton-lang/triton/pull/9578))
- **Extension development:** Install artifacts now include Triton's C++ libraries, headers, and generated TableGen headers ([#9534](https://github.com/triton-lang/triton/pull/9534), [#9681](https://github.com/triton-lang/triton/pull/9681))
- **Dependency downloads:** Moved third-party package downloads from `setup.py` to CMake, added progress and resume support, and avoided repeated CUDA tool downloads ([#9458](https://github.com/triton-lang/triton/pull/9458), [#9722](https://github.com/triton-lang/triton/pull/9722), [#10418](https://github.com/triton-lang/triton/pull/10418), [#10540](https://github.com/triton-lang/triton/pull/10540))
- **Runtime resources and caches:** Added explicit compiled-kernel unloading with a `kernel_unload_hook` and treated incomplete cache entries as misses ([#9444](https://github.com/triton-lang/triton/pull/9444), [#9542](https://github.com/triton-lang/triton/pull/9542), [#10411](https://github.com/triton-lang/triton/pull/10411))

---

## Documentation<a id="documentation-38"></a>

- **Gluon:** Added a rendered Gluon overview, tutorials, examples, and API reference ([#10101](https://github.com/triton-lang/triton/pull/10101))
- **FpSan:** Added the FpSan programming guide and follow-up documentation fixes ([#10177](https://github.com/triton-lang/triton/pull/10177), [#10228](https://github.com/triton-lang/triton/pull/10228))
- **Language reference:** Documented `map_elementwise`, `tl.expect_zero`, out-of-range float-to-integer casts, control-flow scoping, load semantics, and compiler-hint semantics ([#10695](https://github.com/triton-lang/triton/pull/10695), [#10692](https://github.com/triton-lang/triton/pull/10692), [#10678](https://github.com/triton-lang/triton/pull/10678), [#10119](https://github.com/triton-lang/triton/pull/10119), [#10356](https://github.com/triton-lang/triton/pull/10356))

---

## Breaking Changes<a id="breaking-changes-38"></a>

- **Tensor descriptor IR:** The tensor descriptor type now stores its shape, element type, and optional shared-memory layout directly. Out-of-tree MLIR using `!tt.tensordesc<tensor<...>>` must use the new `!tt.tensordesc<..., #layout>` form ([#9851](https://github.com/triton-lang/triton/pull/9851), [#9984](https://github.com/triton-lang/triton/pull/9984))
- **Block-pointer IR:** Block pointers are now implemented in the Python frontend, and the `tt.make_tensor_ptr` and `tt.advance` IR operations were removed. Out-of-tree MLIR using those operations must migrate to regular pointer operations or tensor descriptors ([#9668](https://github.com/triton-lang/triton/pull/9668))
- **Gluon tensor-memory loads:** Register layouts are now inferred automatically. Callers that passed an explicit layout to `buffer.load()` should omit that argument; use `buffer.get_reg_layout()` when the layout is needed separately ([#9594](https://github.com/triton-lang/triton/pull/9594))
- **Gluon packed arithmetic:** The Blackwell `float2` module was replaced by native packed operations such as `add2`, `sub2`, `mul2`, and `fma2`. Call these operations directly from the Blackwell module instead of importing `blackwell.float2` ([#11002](https://github.com/triton-lang/triton/pull/11002), [#11084](https://github.com/triton-lang/triton/pull/11084))
- **Triton-to-Gluon translator package:** The experimental package moved from `triton.tools.triton_to_gluon_translater` to `triton.tools.triton_to_gluon_translator`. Update imports to use the corrected spelling ([#9570](https://github.com/triton-lang/triton/pull/9570))
- **`tl.dot` output type:** When a non-FP32 accumulator is supplied and `out_dtype` is omitted, the output now defaults to the accumulator dtype instead of `tl.float32`. Set `out_dtype=tl.float32` explicitly to preserve the previous behavior ([#10353](https://github.com/triton-lang/triton/pull/10353))
- **Blackwell TMEM layout override:** Removed the `TRITON_PREFER_TMEM_16x256_LAYOUT` environment variable ([#10664](https://github.com/triton-lang/triton/pull/10664))

---

## Contributors<a id="contributors-38"></a>

This release includes contributions from engineers at:

- Meta
- AMD
- NVIDIA
- OpenAI
- Intel
- Google
- And many individual contributors

Special thanks to all contributors who submitted bug reports, feature requests, and code improvements!

