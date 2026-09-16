# LinearLayout 1차 출처 목록

- 수집일(retrieval date): **2026-09-16**
- 수집 방법: 사용자가 직접 넣은 PDF (저장소 루트 → 이 폴더로 이동). 원본 URL 은 확인하지 못함.
- 텍스트 추출: `pdftotext -layout` → 같은 이름의 `.txt`. 슬라이드라 글이 적고 그림이 많다 — 수치·행렬은 PDF 를 직접 볼 것.
- 원본 PDF 는 `.gitignore` 로 제외된다. 커밋되는 것은 `.txt` 와 이 파일뿐.

## 다운로드된 파일

| 파일 | 원본 URL | 설명 |
|---|---|---|
| `Linear_Layouts_ASPLOS26.pdf` (+`.txt`) | 미확인 (사용자 제공) | *Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F₂* — ASPLOS'26 **발표 슬라이드** 41장. Keren Zhou, Mario Lezcano, Adam Goucher 외. 구성: Background · Linear Layouts · Code Generation · Experiments · Conclusions. 마지막 장들이 더 읽을 것(블로그, 시각화 도구, 코드 링크), CuTe 비교, Related Work. `docs/16-linearlayout-talk-primer.md` 가 이것을 재구성한 문서 |

## 블로그 — ita9naiwa (Hyunsung Lee), 수집일 2026-09-16, `curl -L` → HTML 태그 제거로 `.txt`

`docs/17-linearlayout-blog-ita9naiwa.md` 가 이 다섯 편을 정리한 문서다. 전부 CC-BY-NC-4.0. HTML 원본은 `.gitignore` 로 제외.

| 파일 | 원본 URL | 설명 |
|---|---|---|
| `ita9naiwa-ll-in-triton-part-1.html` (+`.txt`) | https://ita9naiwa.github.io/mlsys/2025/10/27/ll-in-triton-part-1.html | *Linear Layout in Triton (1): Basic Idea* (2025-10-27). 정의·선형성 규칙·GF(2)·스위즐 예·코드 예. §1 은 `LinearLayout.h` 머리 주석과 거의 같다 |
| `ita9naiwa-ll-in-triton-part-2.html` (+`.txt`) | https://ita9naiwa.github.io/mlsys/2025/10/27/ll-in-triton-part-2.html | *Linear Layout in Triton (2): LL Examples* (2025-10-27). 축 이름, `blocked`/`swizzled_shared` → LL 출력, minor-to-major, `identity1D`/`zeros1D`/`operator*`/`compose`/`invertAndCompose`/`apply`, MMA 누산기 예 |
| `ita9naiwa-triton-first-pr.html` (+`.txt`) | https://ita9naiwa.github.io/mlsys/2025/08/29/triton-first-pr.html | *Adding Scaled Dot Product to Triton* (2025-08-29). PR #7918 — sm_120 `tt.dot_scaled`, `chooseScaledNvidiaScaleLayout`, RTX 5090 vLLM 벤치 3개 (2차 출처) |
| `ita9naiwa-ptx-mental-model.html` (+`.txt`) | https://ita9naiwa.github.io/mlsys/2025/10/05/ptx-mental-model.html | *PTX Mental Model* (2025-10-05). `cp.async.cg` → `ldmatrix.x4/.x2.trans` → `mma.sync.m16n8k16` → scatter 를 inline PTX 로. 전체 코드는 gist 링크 |
| `ita9naiwa-flashattention-cuda.html` (+`.txt`) | https://ita9naiwa.github.io/mlsys/2025/01/31/flashattention-cuda.html | *Implementing FlashAttention V1 naively* (2025-01-31). 16×16 블록, `__shfl_down_sync` 축약, 스스로 적은 한계 |

## 링크만 (다운로드하지 않음)

| 자료 | URL | 비고 |
|---|---|---|
| 논문 (arXiv 판) | https://arxiv.org/abs/2505.23819 | 슬라이드 36장: *"Figure 4 and Figure 5 have been fixed in the Arxiv version"*, ACM DL 판은 갱신 못 함 → **arXiv 판을 볼 것** |
| Lei 블로그 — linear layout 개념 | https://www.lei.chat/posts/triton-linear-layout-concept/ | 슬라이드 36장 |
| Lei 블로그 — Triton bespoke layouts | https://www.lei.chat/posts/triton-bespoke-layouts/ | 슬라이드 36장 |
| Justin Lebar 블로그 | https://jlebar.com | 슬라이드 36장 |
| Thaihoa 의 Layout Visualizer | https://deep-learning-profiling-tools.github.io/linear-layout-viz/ | 슬라이드 37장 |
| Triton bespoke layout 정의 | https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td | 슬라이드 38장 "Code" |
| LinearLayout 연산 | https://github.com/triton-lang/triton/blob/main/include/triton/Tools/LinearLayout.h | 슬라이드 38장 "Code" |
| LinearLayout Python 인터페이스 | https://github.com/triton-lang/triton/blob/main/python/src/linear_layout.cc | 슬라이드 38장 "Code" |
