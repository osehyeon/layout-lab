# 05. 연구 가능 주제 — 3개 GPU 통합 정리

> 개별 GPU 문서의 "연구 가능 주제" 절([02 §6](02-b200.md#6-연구-가능-주제), [03 §6](03-rtx-pro-6000.md#6-연구-가능-주제), [04 §6](04-rtx-5090.md#6-연구-가능-주제))과 [01 §7](01-architecture.md#7-연구-관점-핵심-포인트)을 모아 GPU별 적합도로 다시 정리했다. 근거와 SW 목록은 원 문서를 참조한다.
>
> 적합도: ◎ 최적 / ○ 가능 / △ 제약 큼 / × 불가

## 1. 주제 × GPU 적합도 매트릭스

| # | 주제 | B200 | PRO 6000 | 5090 | 결정 요인 |
|---|---|:-:|:-:|:-:|---|
| **저정밀 수치 포맷** |||||
| 1 | NVFP4 / MXFP8 **사전학습 레시피** (RHT, 2D weight scaling, stochastic rounding) | ◎ | ○ | △ | HW SR(`cvt.rs`)은 sm_100a 전용, TE NVFP4 주 타깃은 sm_100. PRO/5090은 소규모 재현만 |
| 2 | NVFP4 vs MXFP4 vs FP8 vs W4A16 **PTQ/QAT 추론 정확도·속도** | ◎ (671B급 MoE) | ◎ (70B급 단일 GPU) | ○ (7B–32B) | 메모리 용량이 모델 규모를 결정 |
| 3 | FP8/FP4 **KV cache 양자화**, long-context 품질 | ◎ | ◎ | ○ | 32 GB면 KV 여유가 작다 |
| 4 | **2:4 구조적 희소성**의 실효 이득 검증 (FP4/FP8 sparse) | ○ | ○ | ○ | 스펙 수치는 sparse인데 실측 검증은 드묾 |
| **커널 / 아키텍처** |||||
| 5 | **`tcgen05` / TMEM / 2-CTA MMA** 기반 GEMM·grouped GEMM | ◎ | × | × | sm_100 전용 ISA |
| 6 | **sm_120 block-scaled `mma.sync`** NVFP4/MXFP8 GEMM·attention | × | ◎ | ◎ | 공개 튜토리얼 커널이 peak의 약 60% → 개선 여지 |
| 7 | **Long-context attention** (FA4 설계, softmax/SFU 병목) | ◎ | ○ | ○ | FA4는 sm_100 전용 → sm_120용 attention은 공백 영역 |
| 8 | **sm_100 ↔ sm_120 이식성** (family target, fatbin dispatch, Triton/CuTe DSL의 계열별 성능) | ◎ | ◎ | ◎ | **세 GPU를 모두 가진 환경만의 강점** |
| 9 | Microbenchmark로 아키텍처 해부 (latency, SMEM/L2, Tensor 처리량) | ○ | ○ | ○ | 참고: arXiv 2507.10789 |
| **시스템 / 멀티 GPU** |||||
| 10 | **NVLink vs PCIe** 통신 병목 비교 (TP/PP/EP 스케일링, 통신 압축) | ◎ | ○ | ○ (P2P 없음) | 동일 모델을 세 환경에서 돌리는 비교 연구 |
| 11 | **MoE expert parallelism** (all-to-all, wide-EP) | ◎ | △ | △ | NVSwitch 도메인 필요 |
| 12 | 통신-연산 overlap (TP GEMM + collective 융합) | ◎ | ○ | △ | |
| **서빙 / 운영** |||||
| 13 | **MIG 멀티테넌트 서빙** (MIG vs MPS vs time-slicing, tail latency) | ◎ (7) | ◎ (4, 그래픽 혼합 가능) | × | |
| 14 | 로컬/소규모 **LLM 서빙 스케줄러**, speculative decoding, multi-LoRA | ○ | ◎ | ◎ | 메모리 제약이 뚜렷해야 trade-off가 잘 보임 |
| 15 | **Confidential AI** 오버헤드 (TDX/SEV-SNP, bounce buffer, NVLink 암호화) | ◎ (멀티 GPU) | ○ (Server, 단일 GPU) | × | |
| 16 | vGPU 밀도·격리 | (미확인) | ○ (Server) | × | |
| 17 | **전력 효율 / power capping** (decode vs prefill의 perf/W) | ○ | ◎ (300/600 W 에디션) | ◎ | `nvidia-smi -pl` |
| **학습** |||||
| 18 | 대형 모델 **LoRA / QLoRA** | ◎ | ◎ | ○ | 5090은 FP32 누산 반속 |
| 19 | **GeForce FP32 누산 반속**이 실제 학습/추론에 주는 영향 (5090 vs PRO 6000, 같은 다이) | — | ◎ | ◎ | 두 카드를 모두 가진 환경에서만 가능한 통제 실험 |
| **HPC / 기타** |||||
| 20 | **FP64 에뮬레이션** (Ozaki scheme, INT8 Tensor Core) | ◎ (native FP64 37 TF를 기준선으로) | ○ | ○ | PRO/5090 FP64는 1/64이지만 cuBLAS FP64 에뮬레이션(fixed-point)이 CC 12.x를 CUDA 13.0u2+부터 지원. B300은 tcgen05 INT8 MMA와 FP64 Tensor Core가 없음(FP32:FP64 64:1) → B200이 기준선으로 유리 |
| 21 | **Decompression Engine** 데이터 파이프라인 (nvCOMP, RAPIDS) | ◎ | × | × | |
| 22 | **Neural rendering** (Cooperative Vectors, neural texture/material) | × | ◎ | ◎ | RT core/그래픽 파이프라인 필요 |
| 23 | **비디오 디코드 → AI → 인코드** 파이프라인 | (미확인) | ◎ (NVENC 4) | ○ (NVENC 3) | |

## 2. 세 GPU를 함께 가진 환경에서 특히 가치 있는 주제

이 lab처럼 B200, RTX PRO 6000, RTX 5090을 모두 쓸 수 있다면 단일 GPU 연구보다 **통제된 비교 연구**가 차별점이 된다.

1. **Blackwell 두 계열 간 커널 이식성 연구** (#8)
   - 같은 NVFP4 GEMM/attention을 `sm_100`(tcgen05+TMEM)과 `sm_120`(`mma.sync`)으로 각각 구현하고, peak 대비 효율, 코드 복잡도, CuTe DSL/Triton 같은 고수준 도구가 두 계열에서 내는 효율 차이를 정량화한다.
   - B200과 PRO 6000은 연산/대역폭 비율(ridge point)이 거의 같다([00 §2.1](00-comparison.md#21-roofline-관점-ridge-point--dense-flops--대역폭-flopbyte-계산값)). 따라서 효율 차이를 **ISA·마이크로아키텍처 차이로 귀속**시키기 좋다.
2. **같은 다이, 다른 제품 정책** (#19)
   - RTX 5090 vs RTX PRO 6000은 같은 GB202다. FP32 누산 반속, SM 수(170 vs 188), 메모리 용량만 다르다. BF16 학습·FP8 추론·FP4 추론에서 각각 실측 격차가 이론치(2.4배 / 2.4배 / 1.2배)를 따르는지 검증한다.
3. **인터커넥트 스펙트럼 비교** (#10)
   - NVLink 5 + NVSwitch(B200) → PCIe P2P(PRO 6000, 가능 여부 확인 필요) → PCIe host 경유(5090). 동일 모델에서 TP/PP/EP 전략별 스케일링 곡선을 비교한다. 통신 압축(FP8/FP4 collective)이 어느 대역폭 구간에서 이득인지 경계를 찾는다.
4. **저정밀 학습의 하드웨어 의존성** (#1)
   - stochastic rounding을 HW(`cvt.rs`, B200)와 SW 에뮬레이션(sm_120)으로 수행했을 때 수렴 품질과 속도 차이를 비교한다.

## 3. 권장 개발 흐름 (프로토타입 → 확장)

```
RTX 5090 (sm_120, 32 GB)          → 알고리즘·수치 포맷·정확도 검증, sm_120 커널 프로토타입
   ↓ (같은 ISA, 커널 그대로 이식)
RTX PRO 6000 (sm_120, 96 GB)      → 대형 모델 단일 GPU 실험, FP32 누산 full rate 학습, MIG
   ↓ (mainloop 재작성 필요: tcgen05/TMEM/2-CTA, SMEM 227 KB)
B200 (sm_100, 180 GB × 8, NVLink) → 스케일링, 대규모 학습, sm_100 전용 최적화
```

- 5090→PRO 6000 이식은 타일 크기와 SM 수(wave quantization) 조정 정도로 끝난다.
- PRO 6000→B200 이식에서 이어지는 것은 수치 포맷, scale factor layout, quantize/dequantize epilogue, TMA 로딩, 정확도 결과까지다. **GEMM mainloop는 새로 설계해야 한다.**

## 4. 착수 전 확인할 것 (미확인 항목)

- RTX PRO 6000 Workstation 여러 장 사이의 **PCIe P2P 지원 여부** → `nvidia-smi topo -m`, `p2pBandwidthLatencyTest`로 직접 확인 (#10).
- **Transformer Engine NVFP4 학습**이 sm_120에서 공식 지원되는지 → TE 릴리스 노트 확인 (#1).
- sm_120에서 **FA4 / DeepGEMM 대체 경로**의 현황 (FlashInfer SM120 경로, vLLM SM120 NVFP4 PR) → 각 GitHub 이슈 추적 (#6, #7).
- B200의 **vGPU / NVENC** 지원 여부 (#16, #23).
