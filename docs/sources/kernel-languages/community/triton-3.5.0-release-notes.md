# Triton Release Notes

## Table of Contents
- [Dialect & Frontend](#dialect--frontend)
- [Backend & Compiler](#backend--compiler)
- [AMD/HIP Backend](#amdhip-backend)
- [NVIDIA Backend](#nvidia-backend)
- [Gluon & Layout Improvements](#gluon--layout-improvements)
- [Kernels & Benchmarks](#kernels--benchmarks)
- [Testing & CI](#testing--ci)
- [Build & Infrastructure](#build--infrastructure)
- [Documentation](#documentation)
- [Breaking Changes](#breaking-changes)

---

## Dialect & Frontend

### New Features
- **Warp Specialization Enhancements** (#8005): Made warp specialization require at least 4 warps with proper error messaging to prevent compiler crashes
- **Ragged TMA Support** (#7792, #7783): Added support for write-only and general ragged TMAs with automatic bounds checking using higher-dimensional TMA descriptors
- **Device Assert Mask Support** (#7905): Added `mask` parameter to `tl.device_assert` for easier debugging with masked operations
- **Padding Option for TMA Loads** (#7993): Added support for padding option (including NaN) in TMA descriptor creation and fallback paths
- **Implicit Downcast in TMA Descriptor Store** (#6236): Fixed missing implicit downcast when storing blocks through TMA descriptors
- **Mutations Disallowed** (#7762): Disabled all mutations to address semantic issues in the language
- **Specialized Recursion** (#7468): Enabled functions to recurse on specialized versions of themselves
- **Constexpr Function Cache Invalidation** (#7802): Reworked `constexpr_function` to support cache invalidation and capability checks

### Bug Fixes
- **Floating Point Argument Passing** (#7439): Fixed floating point argument passing for `tl.float16` and other FP types
- **Non-Associative Reduce Rematerialization** (#7272): Avoided rematerialization for non-associative reduce operations to prevent data consistency issues
- **PDL Issue Fix** (#7379): Fixed PDL-related issues in the frontend
- **Constexpr in Tuples** (#7442): Improved handling of constexpr in tuples, fixing type mismatches and in-place mutations
- **Loop Carry Detection** (#7200): Improved detection of loop carries when `@builtin` or `@core.extern` functions modify their arguments
- **Liveouts in Conditionals** (#7318): Fixed detection of liveouts in conditional blocks

### Improvements
- **MLIR Verifier After Parsing** (#7999): Run MLIR verifier after parsing to catch errors early
- **Better Error for num_cta > 1 on sm < 90** (#7812): Improved error messaging for unsupported configurations
- **Extern Elementwise Type Handling** (#7930): Fixed mismatched type handling for `core.extern_elementwise`
- **Libdevice Exposure in Gluon** (#7890): Exposed libdevice functions with improved layout propagation

---

## Backend & Compiler

### LLVM Updates
- **LLVM Bump** (#7881): Updated to llvm/llvm-project@bc773632355b with multiple API changes including:
  - Switched `Constant{Int|Float}Op` type and value order
  - Provided triple for `TargetLibraryInfoImpl`
  - Fixed atomic sync scope for NVIDIA
  - Updated MLIR lib names and ops

### Code Generation
- **Generic Swizzling for convert_layout** (#6982, #7565): Implemented generalized swizzling algorithm for `convert_layout` that:
  - Finds optimal shared memory layout maximizing read/write vectorization
  - Minimizes bank conflicts
  - Supports `ldmatrix/stmatrix` and transpose versions
  - Uses columns and diagonals for better performance
- **Warp-Local Layout Conversion** (#7558): Improved warp-local layout conversion algorithm using shuffles with:
  - Better handling of broadcasting in layouts
  - Fewer `select` and `shuffle` instructions
  - Register packing for sub-32-bit data types
- **Byte Permutes in Intra-Warp Conversion** (#7809): Used byte permute instructions for better performance in layout conversions
- **Tmem Alloc Hoisting** (#7568): Hoisted tmem alloc outside of if statements to reduce register pressure
- **CP.Async Lowering Improvements** (#7314): Moved cp.async to better lowering sequence reusing previous optimizations

### Optimizations
- **Simpler Codegen for Linear Layouts** (#7201): Simplified code generation for linear layouts
- **Vectorization Fixes** (#7845): Fixed vectorization for `PaddedSharedEncoding` with non-default order
- **XOR Trick Refactoring** (#7397): Refactored XOR trick into helper function for better code reuse
- **Shared Memory Offset Fixes** (#7949): Fixed various issues with smem base offsets
- **Min/Max Redux Optimization for Blackwell** (#7465): Implemented new redux.sync optimization

### Bug Fixes
- **Atomic RMW Broadcasting** (#7460): Fixed atomic rmw ops to broadcast results when necessary
- **TMA Load with Multiple Users** (#7398): Fixed lowering of TMA load when users have differing encodings
- **Subview Padding** (#7404): Fixed subview padding for PaddedSharedEncoding
- **Memdesc Subview Fixes** (#7480, #7515): Properly handled memdesc_subview with slicing and offsets
- **FP16 to FP32 Conversion** (#7585): Fixed fp16 to fp32 conversion issues
- **Barrier Synchronization** (#7993): Added bar.sync before deallocating tmem to prevent race conditions

---

## AMD/HIP Backend

### New Features
- **GFX950 (MI350) Support**: Added comprehensive support for AMD's latest architecture including:
  - MFMA scale support (#7799)
  - Scale preshuffling (#7603, #7836)
  - OpSel implementation for scaled MFMA
  - Buffer load/store operations (#7738)
  - Improved register usage in Float8 conversions (#7527)
- **ChainedDot Schedule** (#7601, #7638): Added new scheduling variant for loops with 2 chained dots
- **Ping-Pong Transformation** (#7638, #7458): Added ping-pong support for:
  - Chained dot schedules
  - Async load with num_stages=3
  - MXFP types
- **Buffer Atomic CAS** (#7292): Added support for buffer atomic compare-and-swap
- **FP64 MFMA Support** (#7461): Added support for fp64 dot operations using MFMA intrinsics

### Layout & Memory Optimizations
- **General Swizzling Support** (#7482, #7606): Enabled ConvertLayoutOp general swizzling
- **Padded vs Swizzled Allocation** (#7328, #7750): Introduced specialized allocation pass with proper layout selection strategy
- **Improved LDS Usage** (#7750, #7813): Optimized LDS usage by:
  - Preferring swizzle layouts when LDS limits allow
  - Using single LDS for both transposed and non-transposed access
  - Better layout selection in optimize-lds-usage pass
- **TilesPerWarp Parameter** (#7283): Added tilesPerWarp parameter to MFMA layout for contiguous tile computation
- **Extract Slice Rewrite** (#7128): Refactored extract_slice to support:
  - Arbitrary tensor ranks
  - Relaxed layout constraints
  - CTA tile boundary alignment

### Code Generation Improvements
- **PermlaneSwap Pattern** (#7825, #7861): Added general permlane_swap pattern for ConvertLayoutOp
- **Register Broadcast** (#7407): Added support for register broadcast in slice/concat ops
- **Shared Memory Ops for FP4** (#7626): Added support for M/N packed FP4 with transposition
- **Direct-to-LDS Loads** (#7829): Refactored lowering via common `lowerLdSt` path
- **Local Load/Store Lowering** (#7355): Enabled common code path for local_load/store operations

### FP8 & Numeric Support
- **FP8 Variant Support**:
  - Software emulation for non-gfx942 architectures (#7401)
  - Improved conversions with proper clamping (#7337, #7361, #7363)
  - BF16 to OCP FP8 conversion on CDNA3 (#7469)
  - Float8E4M3FN emulation on CDNA3 and below (#7186)
- **Dot Scaled Support**: Enabled on gfx11 (#7954) and gfx12 (#7644) with emulation via decomposition
- **True16 Handling**: Disabled on gfx11 due to test failures (#7953)

### Stream Pipeliner Enhancements
- **Refactoring** (#7526, #7556): Refactored to use more common pipeliner functionality
- **Async Wait Handling** (#7577): Restricted merging async_wait when pipelining with num_stages=3
- **Mask Operation Support** (#7620): Added ttg.mask handling in stream pipeliner

### Build & Driver
- **LLD Library API** (#7548): Replaced shell-out to lld with direct library API calls
- **hipGetProcAddress** (#7350): Switched to using hipGetProcAddress for querying HIP symbols
- **Driver Version Check** (#7501): Added runtime driver version check with descriptive errors
- **AOT Compilation** (#7007): Added HIP AOT compilation support to compile.py tool

### Bug Fixes
- **Pointer Canonicalizer** (#7242): Fixed attribute propagation when ranks don't match
- **Global Atomic Optimization** (#7496): Optimized global atomic operations following memory model semantics
- **FP32/FP16 to OCP FP8** (#7382): Fixed conversion for subnormal numbers
- **Async Copy Vectorization** (#7250): Fixed async load pipeline for less than 32-bit loads
- **OptimizeLDSUtility Crash** (#7434): Fixed nullptr crash in createTmpLayout
- **Memrealtime on GFX11/12** (#7357): Added proper support using s_sendmsg_rtn_b64

---

## NVIDIA Backend

### Hopper/Blackwell Features
- **Warp Specialization**:
  - Enable for persistent matmul and FA (#7642, #7623)
  - Assign final try_wait to partition (#7757)
  - Tightened user critical section with accumulator (#7509)
  - Fixed rematerialization bug in partitioner (#7427)
  - Optimized partitioning by hoisting above broadcasts (#7692)
  - Enabled 1 buffer for SSA partition dependencies (#7686)
  - Control flow support in TMEM allocation (#7698)
- **WGMMA Support in Gluon** (#7300, #7313): Added Hopper WGMMA with async wait support
- **Aref Operations** (#7479, #7561, #7645): Updated aref ops and lower_aref pass with:
  - Multi-consumer support
  - Stage/cluster attribute passing
  - TMA load aref insertion
  - Control flow handling
- **Partition Loops Rewrite** (#7415): Reimplemented supporting general control flow using mutual recursion

### Blackwell-Specific
- **TMEM Support**:
  - Fixed codegen for Nx1xf32 (#7234)
  - Fixed tmem_subslice for packed layouts (#7207)
  - Allowed splitting block_m=64 along N (#7589)
  - Tcgen05_copy exposure (#7936)
  - Generic lowering for tcgen05.ld/st (#7831, #7874)
- **Tcgen05.commit Op** (#7335): Added separate commit op for better persistent kernel support
- **Subtile QK TMEM Load** (#7655): Improved non-causal fp8 performance by 40-50 TFLOPS

### MMA Improvements
- **FP64 SIMT FMA** (#7310): Added fp64 simt fma support and fp64 mma for SM80/SM90
- **FP8 MMAv2** (#7409): Don't promote fp8 MMAv2 dot inputs for sm120 (~1.9x speedup)
- **Min Dot Sizes Update** (#7411): Reduced minimum dot sizes (e.g., N=16 to lower values)
- **NVVM Op Migration** (#7420, #7471, #7512): Replaced inline assembly with NVVM ops for:
  - WGMMAFenceOp, WGMMACommitGroupOp, ClusterWaitOp
  - ClusterCTAIdOp conversion
  - Better optimization opportunities

### Other Enhancements
- **Bar.warp.sync for 1 Warp** (#7336): Emit more efficient bar.warp.sync for single-warp barriers
- **L2 Cache Hints** (#7219): Limited L2 cache hints to sm >= 80
- **Cublas.gemm Exposure** (#7656): Exposed cublas.gemm for performance testing
- **PTX Workarounds**:
  - Matrix descriptor arithmetic (#7197)
  - TMA device-side descriptor race condition (#7293)
  - Byte permutes ptxas bug (#7933)

---

## Gluon & Layout Improvements

### Gluon Language Features
- **AutoLayout** (#7447, #7466): Added AutoLayout for backward layout inference with:
  - Custom layout inference interface
  - Propagation through operations
  - Conflict detection and error reporting
  - `assert_trivial` flag for performance validation
- **Docstrings** (#7323): Added comprehensive documentation for public Gluon API
- **API Improvements**:
  - Added `numel` and `nbytes` properties (#7507)
  - Added `map_elementwise` (#7564)
  - Fixed `tensor.sum` (#7617)
  - Fixed `splat` returning auto encoding (#7490)
  - Fixed auto encoding inconsistencies (#7726)
  - Added constexpr_function and static_range (#7531)

### Layout System
- **Padded Shared Layout** (#7212): Added new shared memory layout for padding
- **Slice Encoding for SplitOp** (#7247): Improved slice encoding inference
- **LinearLayout Improvements**:
  - Implemented toLinearLayout for TensorMemoryEncodingAttr (#7748)
  - Fixed split op backward propagation (#7340)
  - Generalized getShapePerCTA (#7580)
  - Fixed memdesc reshape encoding inference (#7544)
- **NVIDIA Shared Layout Improvements**:
  - Added NVMMASharedLayout constructor with default swizzle (#7534)
  - Fixed handling of non-default order (#7845)
- **DotOperandLayout Exposure** (#7730): Exposed for WGMMA with LHS in registers

### Tutorial & Examples
- **Attention Kernels** (#7009, #7298, #7488): Implemented complete attention for d64 and d128 with:
  - Persistent kernel support
  - Causal masking optimization
  - FADD2 for row_sum computation (D64)
  - FFMA2 for QK scale
  - Turnstile over exp2 to control MFU access
  - Subtiling optimizations
  - 100-120 TFLOPS improvement for D64
- **Tutorials Added** (#7657): Comprehensive set covering basic to advanced optimized techniques

---

## Kernels & Benchmarks

### MXFP Support
- **Naming Fixes** (#7870): Changed dequantize to quantize, matched arg names
- **FP32 Support** (#7672): Added quant/dequant from/to fp32
- **Transposed Weight Support** (#7795): Handle both transposed and non-transposed mxfp weights
- **Act In/Out Matmul** (#7598): Added mxfp act input and output support
- **Blackwell Value Padding** (#7958): Fixed mxfp value padding for Blackwell
- **Test Coverage** (#7591): Added missing mxfp4 tests
- **MXFP_BLOCK_SIZE Constant** (#7567): Added constant for better readability
- **Empty Tensor Handling** (#7579): Fixed handling of empty tensors in downcast_to_mxfp

### Matmul Optimizations
- **Significant Cleanup** (#7882): Major refactoring of matmul_ogs.py with:
  - More efficient post-processing
  - Better intelligibility
  - Improved documentation
- **Heuristics Improvements** (#7664, #7632): Tweaked block sizes and heuristics
- **Block Size Improvements** (#7897): Better block sizes for batched matmul_ogs with small m/n/k
- **Swizzling Fixes** (#7582, #7587): Fixed swizzling numerics and contiguity
- **Host TMA Usage** (#7182): Increased use of host TMA for X, W, Mx scales
- **Bias Subtiling** (#7232): Added then reverted bias subtiling changes due to regression
- **Zero Elements Support** (#7808): Added support for inputs with 0 elements
- **Index Casting** (#7794): Cast index to int64 to avoid overflow

### MoE & Multi-GPU
- **Routing Improvements** (#7369): 30% performance improvement through:
  - Liberal kernel fusion (7 launches → 4 launches)
  - Specific kernel optimizations
  - FP32 logits: 22.8us → 18.0us
  - FP16 logits: 17.3us → 12.2us
- **BitMatrix for Routing** (#7789): Used bitmatrix for distributed routing
- **Simple Multi-GPU MoE** (#7352): Initialized baseline implementation

### Benchmarks & Tests
- **Launch Overhead** (#7849): Added microbenchmark to track dispatch overhead
- **Total Time Computation** (#7752): Added total kernel time computation
- **Roofline Fixes** (#7670): Fixed roofline plots for compute-bound kernels
- **MLP Fixes** (#7926): Fixed bench_mlp.py for various issues

---

## Testing & CI

### Test Infrastructure
- **Fresh Knobs Usage** (#7687): Use fresh_knobs when touching triton.knobs
- **Environment Variable Restore** (#7807): Fixed monkey patching for proper cleanup
- **Test Cleanup** (#7801): Various cleanups on test_core.py including:
  - Changed kernel launch to warmup
  - Moved tuple tests to test_tuple.py
  - Improved error handling
- **Reduce Test Overflow Fix** (#7470): Limited integer range to avoid overflow
- **Input Generation Consolidation** (#7477): Consolidated input generation for reduce tests

### AMD Testing
- **GFX950 CI** (#7189): Enabled CI for GFX950
- **GPU Isolation** (#7650): Added env-file for better GPU isolation in CI
- **Passing Tests** (#7363, #7365, #7236, #7183): Enabled many passing tests for AMD GFX942 and GFX12
- **Test Skipping**: Properly skipped flaky or unsupported tests (globaltimer, True16, etc.)

### Lit Tests
- **Subfolder Test Fixes** (#7966): Fixed lit tests failing when run via ninja check-triton-lit-tests-<folder>
- **LLD Configuration** (#7992): Added llc to lit tool configuration
- **Test Updates**: Updated numerous lit tests for new features and bug fixes

### NVIDIA Testing
- **GB200 Error Handling** (#7537): Continue running CI when GB200 errors out
- **Warp Specialization Tests** (#7623, #7642): Enabled WS tests for Hopper

---

## Build & Infrastructure

### Build System
- **Out-of-Tree Build** (#7347, #7871): Enabled complete out-of-tree build with TRITON_BUILD_DIR
- **Compile Commands Symlink** (#7305, #7341): Symlink compile_commands.json to root for better IDE support
- **Elapsed Time for MacOS** (#7559): Added elapsed time logging to MacOS builds
- **LLD for MacOS** (#7559): Enabled LLD for macOS build to reduce time
- **Clang Warning Fixes** (#7868): Fixed warnings to build triton with clang
- **Debug Build Default** (#7872): Changed default LLVM build to release

### Dependencies & Environment
- **Custom LLVM Build** (#7279, #6709): Mentioned `make dev-install-llvm` in README
- **Python 3.14 Wheels** (#7695): Added Python 3.14 wheel build support
- **Setuptools Removal** (#7983): Removed setuptools requirement from setup.py
- **New CUDA Versions** (#7384): Automatically handle newer CUDA versions
- **LLVM System Suffix** (#7430): Added TRITON_LLVM_SYSTEM_SUFFIX for user-specified prebuilt LLVM

### Runtime & Compilation
- **Async Compile Mode** (#7306): Added AsyncCompileMode to build multiple kernels in parallel
- **Thread-Safe Allocator** (#7685): Made set_allocator thread-safe using ContextVar
- **AsyncCompileMode Thread Safety** (#7701): Made AsyncCompileMode thread safe
- **Kernel Caching**:
  - Cache invalidation for constexpr_function (#7802)
  - Fixed cache key computation thread safety (#7974)
  - Include constexprs in cache keys (#7348)
  - NvidiaTool.from_path caching (#7569)

### Driver & Backend
- **Host Compiler Flags** (#7659): Allow backend-provided runtime host compiler flags
- **NVIDIA Driver Improvements** (#7769): Slightly improved NVIDIA driver backend with string caching
- **HIP Driver Updates**:
  - Fixed hipError discard (#7832)
  - Fixed multiple compiler warnings (#7838)
  - Fixed undefined behavior (#7806)

---

## Documentation

### New Documentation
- **Community Meetup Notes**:
  - 2025-03-12 (#7255)
  - 2025-05-01 (#7256)
  - 2025-07-09 (#7788)
- **Moderators Guide** (#7787): Updated with YouTube info and event creation
- **Installation Instructions** (#7572): Updated install instructions in docs
- **README Improvements** (#7368): Improved readability and fixed minor issues
- **Running Meetups** (#7103): Added documentation for running Triton Community Meetups

### Code Quality
- **CODEOWNERS Updates**:
  - Linear Layouts (#7754)
  - Gluon section (#7744)
  - Proton Backend (#7782)
- **NFC Refactorings**: Multiple no-functional-change refactorings for better code organization
- **Unused Code Removal** (#7703): Killed unused functions throughout codebase

---

## Breaking Changes

### API Changes
- **Mutations Disallowed** (#7762): All mutations are now disabled in the language
- **Min Dot Sizes** (#7411, #7451): Relaxed minimum dot size requirements (may affect autotuning)
- **Constexpr Handling**: Changed how constexprs are included in cache keys (#7348)
- **Environment Variables**:
  - Only check `TRITON_DEBUG` at import time (#7767)
  - Removed getattr overhead from DriverConfig and CompiledKernel (#7770)
- **Hook System Changes**:
  - Renamed and changed signature for kernel load hooks (#7834)
  - More generalized hooking system (#7866)

### Removed Features
- **Python 3.9 Support** (#8222, #8287): Cleaned up Python 3.9 related code
- **Nightly Installation**: Removed from documentation
- **Local Prefetch Schedule** (#7395): Retired AMD local prefetch schedule hint variant

### Deprecations
- **Warp Size**: Removed hardcoded warp size assumptions (#7253)
- **GetShapePerCTA**: Moving toward elimination in AMD backend (#7740)

---

## Performance Improvements

### Measured Improvements
- **Attention Kernels**: Up to 785 TFLOPS for D64, 1230 TFLOPS for D128
- **MoE Routing**: 30% faster (17.3us → 12.2us for fp16)
- **FP8 on Blackwell**: ~1.9x speedup for large matmuls
- **Launch Overhead**: Reduced by various optimizations (DriverConfig cleanup, etc.)
- **Compile Time**: ~20% savings by skipping link_extern_libs when unnecessary (#7570)

### Optimization Techniques
- **Register Pressure**: Better management through tmem alloc hoisting
- **Vectorization**: Improved through generic swizzling and layout optimizations
- **Bank Conflicts**: Minimized through optimized shared memory layouts
- **Instruction Scheduling**: Better code generation for linear layouts

---

## Notable Bug Fixes

### Correctness Issues
- **Non-Associative Reduce** (#7272): Fixed rematerialization causing incorrect results
- **Atomic Operations** (#7460): Fixed broadcasting for atomic_cas and rmw operations
- **Memory Model**: Multiple fixes for proper fence insertion and synchronization
- **FP8 Conversions**: Fixed numerous rounding and clamping issues
- **TMA Operations**: Fixed various edge cases in TMA load/store

### Crash Fixes
- **Warp Specialization**: Fixed iterator invalidation and use-after-free issues
- **AMD OptimizeLDS**: Fixed nullptr crash
- **Memory Leaks**: Fixed in TritonNvidiaGPU InterleaveTMem.cpp (#7924)
- **Nullptr Access**: Fixed in AMD pingpong ChainedDot (#7694)

### Regression Fixes
- **Block Size Logic Revert** (#7971): Reverted fp8 matmul issues
- **Byte Permutes Revert** (#7899): Reverted due to functional regression, then relanded with fix (#7933)
- **Diagonal Iteration Partial Revert** (#7245): Addressed internal regressions

---

## Contributors

This release includes contributions from engineers at:
- OpenAI
- Meta
- AMD
- NVIDIA
- Intel
- Google
- And many individual contributors

Special thanks to all contributors who submitted bug reports, feature requests, and code improvements!
