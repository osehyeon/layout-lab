# Triton 3.6 Release Notes

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

---

## Dialect & Frontend

### New Features
- **Multidimensional Batch Support** (#8542): Added support for multidimensional batches in `tl.trans` and `tl.dot` operations
- **Ragged TMA Atomic Add** (#8238): Added atomic add support for ragged TMA operations
- **Integer Range Utility** (#8753): Exposed an integer-range utility from AMD range analysis code for broader use
- **Constexpr Through Min/Max** (#8733): Propagate constexpr through builtin min/max functions (BC-breaking)
- **Scales Dimension Checks** (#8564): Added dimension checks for scales in `dot_scaled` operations
- **Loop Bounds Verification** (#8243): Added verification that loop bounds are scalars

### Bug Fixes
- **For Loop Induction Variable** (#8750): Fixed modification of for loop induction variable handling
- **Store Broadcasting** (#8661): Fixed broadcasting issues in store operations
- **Missing `dot_scaled` Handling** (#8658): Fixed missing handling for None acc in `dot_scaled`
- **AugAssign Line Information** (#8703): Attached proper line number information to AugAssign nodes
- **Starred Argument Handling** (#8686): Made starred argument handling more robust
- **Saved Exception Cloning** (#8115): Fixed clone of saved exception before raising
- **Tuple Mangling** (#8060): Fixed mangling for tuples in JIT compilation

### Improvements
- **Optimized `tl.cdiv`** (#8669): Optimized `tl.cdiv` for common case of 32-bit divisors
- **Un-deprecated min/max** (#8734): Un-deprecated min/max on scalar tensors
- **Warmup in KernelInterface** (#8757): Moved warmup functionality into KernelInterface
- **Verification with Diagnostics** (#8074): Frontend always verifies with diagnostics enabled
- **Constexpr with do_not_specialize Error** (#8275): Added error when constexpr is combined with do_not_specialize
- **Deprecated ast.Num Replacement** (#8698): Replaced usage of deprecated `ast.Num`

---

## Backend & Compiler

### LLVM Updates
- **LLVM Bump** (#8299): Bumped to llvm/llvm-project@f6ded0be897e
- **LLVM Head Merge** (#8612): Merged back changes from llvm-head with updated APIs
- **Inliner Import** (#8152): Import inliner in triton-opt for better optimization

### Code Generation
- **CTALayout as LinearLayout** (#8770): Made CTALayout an honest-to-goodness LinearLayout for better representation
- **Shared Layout Rank Check** (#8772): Added check that Shared layouts have rank equal to the tensor or one less
- **Backward Propagation Fix Point** (#8776): Run remove backward prop until fix point for correctness
- **Generic `tcgen05.cp` Lowering** (#8225): Implemented generic lowering for `tcgen05.cp`
- **Generic Matrix Descriptors** (#8321): Implemented shmem matrix descriptors generically
- **LinearSharedEncoding Support** (#8116): Added support for LinearSharedEncoding
- **BF16x3 Trick** (#7592): Implemented BF16x3 trick for improved performance
- **Padded Shared Linear Remapping** (#7929): Added linear remapping to padded shared layout

### Optimizations
- **Compilation Time Improvement** (#8689): Improved compilation time in constant sanitizer pass
- **AxisInfo Loop Removal** (#8679): Removed unnecessary loop over roots in AxisInfo analysis
- **Constant Analysis** (#8502): Improved constant analysis in AxisInfo
- **Combinatory Explosion Prevention** (#8477): Prevented combinatory explosion when checking tmem_load uses
- **Layout Conversion Vectorization** (#8655): Fixed vectorization for convert_layout with ldmatrix and stmatrix
- **Maybeduplicate Generalization** (#8492): Generalized maybeDeduplicate to all layouts

### Bug Fixes
- **cp_async Alignment** (#8752): Fixed cp_async used in pipeliner when alignment info gets lost
- **While Op Layout Propagation** (#8751): Prevented backward layout propagation through while op
- **AxisInfo Handling** (#8723, #8754): Fixed handling of unvisited operands in AxisInfoAnalysis
- **64-bit Atomic CAS** (#8105): Fixed 64-bit `atomic_cas` operation
- **Memdesc of Pointers** (#8515): Fixed memdesc handling for pointer types
- **Alloc Shape Reset** (#8537): Reset alloc_shape when doing memdesc_index
- **Denorm Flushing** (#8557): Don't flush denorms for precise div/sqrt
- **Local Load Reordering** (#8423): Prevented reordering local_load across side-effecting operations
- **Pattern Reordering** (#8266): Restricted pattern re-ordering of alloc and reshape
- **Poison Op AxisInfo** (#8489): Fixed AxisInfo handling of PoisonOp producing MemDesc

### Analysis Improvements
- **Trans Contiguity** (#8226): Added tt.trans contiguity analysis support
- **Hint Analysis** (#5254): Fixed hint analysis in axis info
- **Topological Sort Deprecation** (#8596): Deprecated triton's custom topological sort in favor of MLIR's

---

## AMD/HIP Backend

### GFX1250 (RDNA4) Support
- **Initial Skeleton** (#8131): Added gfx1250 skeleton support
- **WMMA Support** (#8174, #8283, #8312): Added initial and scaled WMMA support for gfx1250
- **TDM Support** (#8333, #8392, #8479): Added Tensor Data Movement (TDM) load/store support
- **Async Copy** (#8509, #8510, #8621, #8622): Added async copy and async wait support
- **Buffer Ops** (#8130, #8532): Enabled buffer atomics and exposed buffer ops
- **Multicast Loads** (#8719, #8759): Added async load to LDS multicast and multicast in `tt.LoadOp`
- **ds_read_tr** (#8461): Added gfx1250 support for ds_read_tr
- **LDS Memory Barriers** (#8681): Added support for LDS memory barriers
- **Shared Memory Size** (#8517): Updated shared memory size from TargetInfo
- **num_cta > 1** (#8718): Support launches with num_cta > 1 on gfx1250
- **Scale Preshuffling** (#8576): Implemented scale preshuffling and opSel

### MXFP & Scaled Dot
- **Scale Preshuffling in Decomposed Dot** (#8170): Support scale preshuffling in decomposed scaled dot
- **Pipeline Scale via LDS** (#8258): Pipeline scale in decomposed scaled dot via LDS
- **Scaled Upcast Ops** (#8088): Introduced scaled upcast ops for hardware upcasting
- **FP4->BF16 Optimized Conversion** (#8145): Added optimized fp4->bf16 conversion for MI300
- **Scaled Dot Decomposition for GFX950** (#7839): Enabled f16 * mxfp scaled dot decomposition

### Layout & Memory Optimizations
- **Permlane Swap** (#7947): Use permlane_swap for layout conversions between dot operations
- **Padded Shared with AsyncCopy** (#8365): Use PaddedLayout with AsyncCopy on gfx950 when pipelining
- **LDS Layout Selection Redesign** (#8053): Redesigned stream pipeliner LDS layout selection logic
- **Padded Encoding Restrictions** (#8583): Relaxed padded encoding block size restrictions
- **Direct-to-LDS with Padded** (#8185): Coalesce direct-to-lds loads with padded encodings
- **Contiguity Hint for Direct-to-LDS** (#8761): Use contiguity hint for direct-to-lds ops
- **BypassLDS Feature** (#7968): Added bypassLDS feature to StreamPipeline

### Code Generation
- **ds_read_tr with Linear Layout** (#8235): Use linear layout to infer and emit ds_read_tr
- **ds_read_tr Restrictions Lifted** (#8442): Lift unneeded ds_read_tr lowering restrictions
- **ds_read_tr Vec Size Limit** (#8377): Limit vec size for ds_read_tr + padded layouts by min interval
- **Wave ID Optimization** (#8601): Optimized gfx9 wave id code generation
- **MFMA Layout Refactor** (#8213): Refactored MFMA layout implementation
- **MFMA Select Replacement** (#8320): Replaced mfma select in LLVM conversion
- **FP8/BF8 WMMA Instruction Selection** (#8649): Fixed instruction selection for fp8/bf8 wmma
- **Chained WMMA Optimization** (#7374): Optimized chained multiplications for WMMA
- **BF16 v_dot** (#8444): Use v_dot for bf16 multiplication on gfx11/gfx12

### Build & Driver
- **ROCm 7 Docker Image** (#8224): Switched to use official ROCm 7 docker image
- **HIP v6 Requirement** (#8748): Only require HIP v6 which is necessary
- **HIP Header Update** (#8709): Updated HIP header files to 7.1
- **Optional Symbols Support** (#8729): Support optional symbols in driver.py
- **Uniform Workgroup Size** (#8720): Indicate uniform workgroup size to LLVM
- **MIR Dump Option** (#8663): Added option to dump MIR
- **Custom LLVM Scheduler** (#8326, #8700): Added schedule hint for custom LLVM scheduler

### Bug Fixes
- **Pointer Canonicalization** (#8465, #8276): Fixed ptr-canonicalization segfault and assertion
- **Large Tensor Pointer Canonicalization** (#8359): Disabled pointer-canonicalization for large tensors
- **Padded Shared Local Load** (#8683): Fixed padded shared when lowering local load
- **Nondeterministic Atomic Tests** (#8633): Fixed nondeterministic atomic tests failure on RDNA
- **Buffer Cache Swizzling** (#8264): Turned off buffer op cache swizzling temporarily
- **Direct-to-LDS on CDNA1/2** (#8280): Disabled direct-to-lds loads on CDNA1 and CDNA2
- **Floating-point Upcasting Rounding** (#8268): Skip rounding mode for floating-point upcasting
- **TilesPerWarp Boundary Cases** (#8467): Fixed deduceTilesPerWarp boundary cases
- **fast_tanhf Overflow** (#8551): Reimplemented fast_tanhf() to avoid overflow
- **MFMA Small K Selection** (#8278): Avoid selecting MFMA with smaller K than problem size

---

## NVIDIA Backend

### Blackwell Features
- **TMEM Bitwidth** (#8136): Added bitwidth to TMEM encoding for better representation
- **TMEM Layout Broadcasting** (#8148): Represent broadcasting in TensorMemoryLayouts
- **TMEM Layout Construction** (#8202): Simplified TMEM layout construction and row/col computation
- **Generic tcgen05.ld/st Layouts** (#8421, #8495): Generate distributed layouts for `tcgen05.ld/st` generically
- **tcgen05.mma Generalization** (#8386): Generalized `tcgen05.mma` to accept `SharedLinearEncodingAttr`
- **tcgen05.cp Generic Lowering** (#8102, #8338): Towards a generic tcgen05.cp lowering via matrix descriptors
- **tcgen05.mma Verifier** (#8725): Fixed missing case in tcgen05.mma verifier
- **Explicit Commit Merge** (#8026): Added rewrite pattern to merge explicit commit ops into MMAv5
- **2CTA Mode Support** (#8644, #8653): Initial support for 2CTA mode in Gluon with global flag
- **reqnctapercluster Emission** (#8645): Emit reqnctapercluster for better cluster sizing

### SM120 Features
- **Native FP4 Scaled Dot** (#8494): Added native FP4 scaled_dot for SM120
- **Native MXFP FP8 Scaled Dot** (#7918, #8029, #8129): Added native MXFP FP8 scaled_dot for SM120
- **TMA Gather4** (#8498): Enabled TMA gather4 on sm_120 and sm_121
- **DotScaledScaleLayout Rewrite** (#8482): Rewrote getSM120DotScaledScaleLayout and refactored MMAv2

### Warp Specialization
- **E2E Aref** (#8262): Enabled end-to-end aref for warp specialization
- **TMA Load Aref Insertion** (#7826): Use aref for TMA load pipelining and lowering
- **TMEM Aref Insertion Pass** (#8009): Added aref tmem insertion pass
- **Partition Representation Rework** (#8123): Reworked partition representation
- **Assign Partitions to All Ops** (#8534): Assign partitions to all ops for consistency
- **Nested Loop Recognition** (#8451): Recognize warp-specialized nested loops in AssignLatencies
- **Scalar Ops Across Partition** (#8061): Support scalar ops across partition boundaries
- **Stage/Phase Assignment** (#8329): Assign stage-phase only to partitions that need it
- **Partition Scheduler Annotations** (#8215): Partition-scheduler annotates all ops with fixes
- **Control Flow Support** (#8651): Support ops annotations outside tt.ws loops
- **Then/Else Heuristic Patch** (#8656): Patched partitioner then/else heuristic
- **Fence After Local Store** (#8317): Added missing fence after local_store for MMAv5 consumers

### Other Enhancements
- **Descriptor Bit 46** (#8032): Turn on bit 46 for descriptors in mmav5
- **Matrix Descriptor No-Swizzle** (#8027): Fixed matrix descriptor for no-swizzle case
- **WGMMA Wait Op CVT** (#8579): Fixed unnecessary cvt caused by wgmma wait op
- **Enable Reflect FTZ Flag** (#8762): Added enable_reflect_ftz flag to NVIDIA backend
- **libcuda.so.1 Usage** (#8668): Modified NVIDIA backend driver to use libcuda.so.1
- **Padded Shared in MemDescSubslice** (#7944): Support padded shared in MemDescSubsliceOp
- **Ptxas Workaround** (#8155): Fixed ptxas workaround in convert_layout
- **ldmatrix/stmatrix.b8.trans** (#7542): Added support in local_load/store for ldmatrix/stmatrix.b8.trans

---

## Gluon & Layout Improvements

### Gluon Language Features
- **Warp Specialize API Change** (#8527): Changed `gl.warp_specialize` API for better usability
- **Multi-CTA Support** (#8468, #8587, #8644): Basic multi-cta support with initial implementation
- **num_ctas Implementation** (#8602): Implemented `num_ctas` in Gluon
- **Device-Side TMA** (#8505): Added device-side TMA support
- **Coalesced Layout** (#8604): Added coalesced layout support
- **get_num_warps** (#8133): Added `ttgl.get_num_warps` metafunction
- **gather Integration** (#8018): Integrated `gather` and its layout tests
- **reduce with No Axis** (#8396): Added support for reduce with no axis
- **assume Support** (#8394): Added support for assume operation
- **cat Remapping** (#8715): Remap more `tl` functions into gluon and expose `cat`
- **Type Verifiers** (#8007): Added type verifiers for many methods

### Layout System
- **bank_conflicts Exposure** (#8181): Exposed bank_conflicts and to_linear_layout
- **Linear Layout Python Interface** (#8521): Added LL Python Interface
- **Layout Check Message** (#8456): Improved layout check error messages
- **Tensor Rank Verification** (#8242): Verify tensor rank and layout rank match
- **MemDesc Trans/Reshape** (#8251): Have MemDesc{Trans,Reshape} accept equivalent layouts
- **Fp4ToFp Backward Propagation** (#8438): Fixed backwards propagation for Fp4ToFp
- **ResolveAutoEncodings Print** (#8228): Print encoding in ResolveAutoEncodings

### Gluon AMD Support
- **Host-Side TDM Descriptor** (#8722): Initial support for host-side tdm descriptor
- **TDM 1D-5D Support** (#8743): Support TDM load/store for 1D-5D tensors
- **TDM Pred Exposure** (#8767): Expose pred for TDM load
- **Scale Layout Selection** (#8673): Turn select scale layout into constexpr function
- **WMMA/MFMA Scale Layout** (#8496): Expose get wmma/mfma scale layout
- **AMDWMMALayout Exposure** (#8090): Exposed AMDWMMALayout
- **WMMA for RDNA3/RDNA4** (#8111): Exposed WMMA for RDNA3 and RDNA4
- **Buffer Ops Exposure** (#8532): Expose buffer ops to gfx1250
- **buffer_atomic_rmw API** (#8325): Refactored buffer_atomic_rmw API
- **async_copy for gfx1250** (#8622): Added `async_copy` to Gluon for gfx1250
- **Async Wait Groups** (#8605): Wait outstanding async commit groups instead of instructions

### Gluon NVIDIA Support
- **tcgen05 mma scaled** (#8393): Added tcgen05 mma scaled support
- **MMAv2 and Dot FMA** (#8227): Exposed MMAv2 and Dot FMA
- **Float2 API** (#8209): Added proper float2 API for Blackwell
- **warp_specialize Docs** (#8553): Updated gl.warp_specialize docs

### Bug Fixes
- **Translator Fixes** (#8569): Fixed several things in the translator
- **SwizzledSharedLayout** (#8003): Fixed getting layout from a SwizzledSharedLayout
- **Bank Conflict Computation** (#8200): Fixed bank_conflict computation with shmem broadcasting
- **Trans Alloc Optimization** (#8193): Simplified and fixed trans(alloc) optimization
- **TMem Alloc/Store Pattern** (#8192): Fixed pattern combining tmem_alloc and store
- **Constant CSE** (#8323): Disabled constant CSE before auto layout propagation

---

## Kernels & Benchmarks

### MXFP Improvements
- **MXFP Conversions Speedup** (#8610): Significant speedup for mxfp conversions
- **FP32 MXFP Support** (#8672 from 3.5): Added quant/dequant from/to fp32
- **MXFP4 Hopper Layout on A100** (#8474): Apply MXFP4 Hopper layout on A100
- **A100 MXFP4 Upcasting** (#8428): Support A100 upcasting for mxfp4
- **MXFP8 X Support** (#8062): Support mxfp8 `x` in triton_kernels
- **BF16 x MXFP4 Bug Fix** (#8478): Fixed bf16 x mxfp4 bug with SUBTILE_FACTOR > 1
- **EXPT_IS_INNER Support** (#8385): Support EXPT_IS_INNER for MX
- **w_scale Swizzle Handling** (#8652): Handle w_scale without swizzle correctly
- **Max Value Handling** (#8356): Handle values close to max correctly without overflow
- **x_scale OOB Fix** (#8369): Fixed x_scale out-of-bounds access
- **Round-to-Nearest-Even** (#8110): Use round-to-nearest-even mxfp4 quant for consistency

### Matmul Optimizations
- **Batched Block Sizes** (#7897, #8084): Improved block sizes for batched matmul_ogs with small m/n/k
- **Ragged Matmul DW** (#8256): Added support for ragged matmul dw
- **Split-K Fixes** (#8252): Two small split-k fixes
- **Batched Split-K** (#8327): Fixed and enabled batched matmul with split-k
- **Split-K Constraint** (#8404): Added constraint on `split_k` on `m * n`
- **Launch Metadata** (#8429): Fixed launch metadata computations for matmul_ogs
- **Transposed X Fix** (#8156): Fixed _p_matmul_ogs when x is transposed
- **MX Scale Mask** (#8161): Fixed mx scale mask update

### Expert Parallelism & MoE
- **Basic Expert Parallelism** (#8448): Basic expert parallelism implementation
- **EP Sharding** (#8493): Incorporated EP sharding and deprecated legacy communication
- **CUDA Graph Tracing** (#8563): vllm compatible version of CUDA Graph tracing for expert parallelism
- **Fused Matmul + Comms** (#8340): Fused matmul_ogs + communications
- **Split-K Decoupling** (#8483): Decoupled split-k reduction from inter-expert reductions
- **Small Batch MoE Tuning** (#8206): Tuning for small batch MoE
- **BitmatrixMetadata** (#8375): Added `BitmatrixMetadata` and `RaggedTensorMetadata`; deprecated triton_kernels.routing
- **BitMatrix Fix** (#8599): Fixed BitmatrixMetadata col/row_sorted_indx
- **y_indx Support** (#8472): Support `y_indx` and uniform distribution

### Benchmarks
- **Roofline Plotting** (#8244): Fixed roofline plotting
- **HipBlas Roofline** (#8216): Integrated hipblas in roofline measurement
- **GFX950 BF16 x MXFP4 MoE** (#8176): Updated parameters for bf16 x mxfp4 MoE kernel
- **MLP Benchmark Fix** (#8699): Added missing `reduction_n=2` to `bench_mlp.py`
- **tl.clamp Usage** (#8728): Use tl.clamp whenever possible in triton_kernels

### Other Improvements
- **Redundant Reduce Removal** (#8647): Removed redundant reduce for topk=1
- **Split-K with Fused Scatter** (#8618): Forbid use of `split_k > 1` with fused scatter
- **Layout Dataclasses** (#8690): Made layout classes dataclasses (NFC)
- **HopperValue Padding** (#8677): Pad tensors in `HopperValue` layout
- **A100 Default Layout Revert** (#8549): Reverted a100 default layout change
- **opt_flags Reset** (#8453): Added function to reset opt_flags

---

## Proton Profiling

### New Features
- **Global Memory Support** (#8641): Global memory support for proton intra kernel profiler
- **Global Timestamps** (#7729): Capture global timestamps for consistent cross-CTA timeline
- **Intra Kernel Call Stack** (#8071): Added kernel call stack to intra kernel events
- **NVTX/ROCTX Support** (#8095): Init NVTX/ROCTX support for external profilers
- **Graph Profiling** (#8676): Improved graph profiling part-1
- **Disable Flag** (#8293): Added flag to disable proton to use other profilers

### Improvements
- **Scope ID Allocation Refactor** (#8613): Refactored scope id allocation to allow flexible annotations
- **Concrete Line Info** (#8614): Attached concrete line info to proton operations
- **FinalizeOp Refactor** (#8635): Refactored finalizeOp to reduce buffer write overhead
- **Buffer Size Description** (#8650): Improved default buffer size description
- **Profile Allocator** (#8730): Made profile allocator a global var
- **Backend Lib Settings** (#8246): Simplified backend lib settings
- **Python Frame Representation** (#8241): Unified python frame representation

### Bug Fixes
- **Dominance Analysis** (#8712): Fixed dominance analysis in Proton
- **Function Metadata Cleanup** (#8713): Do not clean up function metadata at finalize
- **Memory Leak Fix** (#8692): Fixed memory leak and removed unused variables
- **Buffer Overflow Warning** (#8109): Fixed proton intra kernel profiling buffer overflow warning
- **Concurrent Profiling** (#8210): Do not allow concurrent profiling with different modes
- **Triton Function Filtering** (#8021): Filter out all intrinsics when counting triton functions
- **Global Time Trace Precision** (#8309): Fixed global time trace precision

### Testing
- **Internal Testing Utility** (#8204): Use more internal testing utility
- **Proton Tests Conditional** (#8237): Conditionally include Proton tests
- **AMD Proton Tests** (#8388): Simplified proton tests on AMD
- **Skip AMD Overhead Tests** (#8665): Skip hip overhead tests
- **Globaltime GFX950** (#8627): Disabled test_globaltime on gfx950

---

## Concurrency Sanitizer (ConSan)

### New Features
- **Deadlock Detection** (#8285): Added deadlock detection capability
- **Warp Specialization Support** (#8189, #8265): Added support for WarpSpecialization with fixes
- **TMA Store Validation** (#8672): Support for TMA store validation

### Improvements
- **Function Call Opcodes** (#8559): Converted consan instrumentation opcodes to function calls
- **Compilation Time** (#8689): Improved compilation time
- **Cache Invalidation** (#8332, #8342): ConSan env var should be cache invalidating

---

## Testing & CI

### Test Infrastructure
- **Frontend Tests for test-nogpu** (#8771): Added frontend tests to make test-nogpu
- **Device Fixture Usage** (#8512): Using device fixture instead of cuda in tensor descriptor tests
- **tb=short in CI** (#8440): Added tb=short to CI for shorter tracebacks
- **Subprocess Removal** (#8350): Removed subprocess usage from test_triton_debuginfo_on
- **SmallVector Crash Fix** (#8544): Fixed SmallVector crash issue of AxisInfoAnalysis

### AMD Testing
- **GFX950 CI Fixes** (#8741, #8760): Avoid gfx950 runner failing others, fix continue-on-error
- **GFX1250 Tests** (#8680): Updated gfx1250 Gluon tests
- **Padded Layout Lit Tests** (#8399): Added lit tests for pipelining with padded layouts on gfx950
- **CDNA2 Atomic CAS** (#8376): Disabled flaky atomic cas test on CDNA2

### NVIDIA Testing
- **Warp Specialization Tests**: Enabled WS tests for various features
- **GB200 Error Handling**: Continue running CI when GB200 errors out

### Lit Tests
- **Redundant CTALayout Removal** (#8704): Removed all redundant CTALayout information from LIT tests
- **ASAN Fix** (#8117): Fixed ASAN initialization-order-fiasco issue in tensor_layout_print.mlir test
- **MMA Support Check** (#8640): Perform supportMMA check during IR verification

---

## Build & Infrastructure

### Build System
- **Python 3.9 Support Removal** (#8222): Cleaned up Python 3.9 related code/docs
- **Python 3.10 Minimum** (#8167): Updated MIN_PYTHON version to 3.10
- **Python 3.14 Wheels** (#7695 from 3.5): Python 3.14 wheel build support
- **Python 3.13 Fix** (#8403): Fixed Python 3.13 compatibility issues
- **CentOS 7 Removal** (#8191): Removed CentOS 7 build
- **Actions Updates** (#8347, #8361, #8187): Bumped actions/setup-python to v6, tj-actions/changed-files to v47
- **TarFile Deprecation** (#8337): Fixed deprecation warning from TarFile.extractall
- **Unused CMake Removal** (#8408, #8362): Removed unused include(ExternalProject) and find_library

### Compilation & Runtime
- **Native Specialize** (#7771): Native specialize for improved launch latency
- **AsyncCompile Error Option** (#8756): Added option to ignore errors in AsyncCompile
- **JIT Functions to Kernels** (#8721): Added test that jit functions can be passed to kernels safely
- **JIT Specialization Serialization** (#8639): Fixed JIT specialization data (de)serialization for tuples and constexprs
- **Aggregate Cache Keys** (#8528, #8568): Made sure aggregate members are added to the cache key
- **Interpreter Mode Cache** (#8499): Disabled cache when interpreter is enabled
- **Backend Detection** (#8046): Added env var to speed up backend detection in tree

### Compiler Pipeline
- **Configurable Pass Pipeline** (#8137): Added hook for configurable/overridable compiler pass pipeline
- **MLIR Reproducer Retention** (#8113): Retain mlir reproducer temporaries from prior run pass pipelines
- **MLIR Multithreading Disable** (#8255): Disabled MLIR multithreading
- **SCF to CF Inliner** (#8017): Run the inliner after scf-to-cf

### CUDA Updates
- **PTXAS Upgrade** (#8476): Upgraded ptxas to 12.9.86 for Blackwell
- **CUDA 13 CRT Headers** (#8336): Fixed crt header download location for CUDA >= 13
- **ptxas_options Knobs** (#8121): Updated ptxas_options knobs default value

### AOT Compilation
- **Gluon Kernel Compilation** (#8660): Support compile gluon kernels in compile.py

### Interpreter
- **TRITON_INTERPRET Cleanup** (#8735, #8736): Made TRITON_INTERPRET cleanup after itself with improvements
- **Tensor Descriptor Stride Validation** (#8670): Fixed tensor descriptor stride validation
- **Histogram Silent Corruption** (#8550): Fixed silent data corruption in histogram
- **TensorHandle Dtype Validation** (#8594): Validated TensorHandle np/tl dtypes size
- **Pre-run Hooks** (#8573): Enabled pre-run hooks in interpreter mode

---

## Documentation

### Community Meetup Notes
- **2025-09-03** (#8178): Adding meeting notes for 2025-09-03 community meetup
- **2025-11-05** (#8727): Added meeting notes for 2025-11-05 community meetup

### Technical Documentation
- **dot_scaled Requirements** (#8433): Clarified lhs_scale and rhs_scale requirements in dot_scaled
- **Install Command Fix** (#8271): Fixed install command in tutorials README.rst
- **Gluon Tutorial Fix** (#8593): Fixed gluon tutorial example
- **Gluon Layout Explanation** (#8020): Fixed description in layout explanation in gluon tutorial
- **Proton README** (#8319): Updated Proton README
- **Proton Tutorial** (#8334): Intra kernel profiling tutorial and examples
- **Tutorial Units** (#8631): Added units to result tables in tutorials
- **AMD Scaled Matmul Tutorial** (#8099): Added AMD GPUs in scaled matmul tutorial

### README Updates
- **Triton Conference 2025** (#8186): Added Triton Conference 2025 details to README
- **Conference Registration** (#8114): Added conference registration link

---

## Breaking Changes

### API Changes
- **Constexpr Through min/max** (#8733): BC-breaking propagation of constexpr through builtin min/max
- **Aggregate Cache Keys** (#8568): Aggregate members are now added to the cache key
- **warp_specialize Argument Tuples** (#8368): Required warp_specialize default_args and worker_args to be tuples
- **warp_specialize API Change** (#8527): Changed `gl.warp_specialize` API

### Proton Changes
- **Metric ValueId Types** (#7979): BC-break - Prevent updating the same metric valueId with different types

### Removed Features
- **Python 3.9 Support** (#8222): Removed Python 3.9 support, minimum is now 3.10
- **CentOS 7 Build** (#8191): Removed CentOS 7 build support
- **GlobalPrefetch/LocalPrefetch Knobs** (#8295): Removed GlobalPrefetch and LocalPrefetch Knobs for AMD

### Deprecations
- **triton_kernels.routing** (#8375): Deprecated triton_kernels.routing in favor of BitmatrixMetadata
- **Custom Topological Sort** (#8596): Deprecated triton's custom topological sort

---

## Performance Improvements

### Compilation Performance
- **Native Specialization** (#7771): Significant launch latency improvements through native specialize
- **ConSan Compilation Time** (#8689): Improved compilation time in constant sanitizer

### Runtime Performance
- **MXFP Conversions** (#8610): Speedup for mxfp conversions
- **FP4->BF16 Conversion** (#8145): Optimized fp4->bf16 conversion for MI300
- **Permlane Swap** (#7947): Use permlane_swap for efficient layout conversions
- **Chained WMMA** (#7374): Optimized chained multiplications for WMMA
- **Expert Parallelism** (#8448): New expert parallelism implementation

### Memory Optimizations
- **BypassLDS** (#7968): Added bypassLDS feature to skip LDS when possible
- **Padded Layout Selection** (#8053): Redesigned stream pipeliner LDS layout selection

---

## Notable Bug Fixes

### Correctness Issues
- **Loop Induction Variable** (#8750): Fixed modification of for loop induction variable
- **Store Broadcasting** (#8661): Fixed broadcasting in store operations
- **64-bit Atomic CAS** (#8105): Fixed 64-bit atomic_cas
- **Histogram Corruption** (#8550): Fixed silent data corruption in histogram
- **MXFP Overflow** (#8356): Handle values close to max correctly without overflow

### Crash Fixes
- **Pointer Canonicalization** (#8465): Fixed ptr-canonicalization segmentation fault
- **SmallVector Crash** (#8544): Fixed SmallVector crash issue in AxisInfoAnalysis
- **ASAN Issues** (#8117): Fixed ASAN initialization-order-fiasco

### Regression Fixes
- **Batched Block Sizes Reapply** (#8084): Reapplied improved block sizes after fixes
- **Native MXFP FP8 Reapply** (#8129): Reapplied native MXFP FP8 scaled_dot for SM120

---

## Experimental Triton to Gluon Translator

- **Translator Tool** (#8417): Added experimental translator from Triton to Gluon for easier migration

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

