# LLM Benchmarks - Intel Arc Pro B70 (llama.cpp SYCL)

Full single-card LLM numbers. Dual-card results are in [multi-gpu.md](multi-gpu.md), engine comparisons in [engine-comparison.md](engine-comparison.md).

All numbers below are from `llama-bench`, SYCL backend, F16 accumulation (`-DGGML_SYCL_F16=ON`), commit **`ec6f7a6a5c`** (`b8840-12-gec6f7a6a5-dirty`, 2026-04-21). Every row captured power telemetry via the `xe` driver's hwmon energy counters. See [methodology.md](methodology.md) for build details and [CHANGELOG.md](CHANGELOG.md) for the transition from earlier numbers.

---

## Master Table - All Models (v0.2 / v0.2.1)

Sorted by generation speed (`tg128`). "t/J" is tokens-per-joule (higher = more efficient).

| Model | Type | Quant | Size (GiB) | GPUs | pp512 t/s | tg128 t/s | avg W | t/J |
|-------|------|-------|-----------|------|-----------|-----------|-------|-----|
| Qwen 2.5-1.5B | Dense | Q4_K_M | 1.0 | 1 | 8,048 | **216.4** | 129 | 1.68 |
| Llama 3.1-8B Instruct | Dense | Q4_K_M | 4.6 | 1 | 2,452 | 82.6 | 37* | 2.26 |
| Qwen 3.5-9B | Dense | Q4_K_M | 5.3 | 1 | 2,302 | 60.2 | 168 | 0.36 |
| **Qwen 3.6-35B-A3B** | **MoE** | **UD-Q4_K_M** | **20.6** | **1** | **615** | **54.7** | **114** | **0.48** |
| Qwen 3.5-35B-A3B | MoE | Q4_K_M | 20.5 | 1 | 618 | 54.5 | 92 | 0.59 |
| Gemma 4 26B-A4B | MoE | Q4_K_M | 15.7 | 1 | 1,129 | 52.6 | 102 | 0.52 |
| Qwen 3.5-9B | Dense | Q8_0 | 8.9 | 1 | 2,444 | 48.0 | 149 | 0.32 |
| Phi-4 14B | Dense | Q4_K_M | 8.4 | 1 | 1,424 | 43.7 | 40* | 1.08 |
| Qwen3-Coder-Next 80B-A3B | MoE | Q4_K_M | 45.1 | 2 | 305 | 43.4 | 79 | 0.55 |
| Qwen 3.6-35B-A3B | MoE | Q8_0 | 34.3 | 2 | 458 | 36.5 | 91 | 0.40 |
| Mistral Small 3.2-24B | Dense | Q4_K_M | 13.3 | 1 | 994 | 30.1 | 167 | 0.18 |
| Devstral Small 2-24B | Dense | Q4_K_M | 13.3 | 1 | 987 | 30.0 | 165 | 0.18 |
| Gemma 4 31B | Dense | Q4_K_M | 17.1 | 1 | 601 | 21.7 | 169 | 0.13 |
| Qwen 3.5-27B | Dense | Q4_K_M | 15.6 | 1 | 718 | 20.4 | 178 | 0.11 |
| Qwen 3.5-27B | Dense | Q8_0 | 26.6 | 1 | 776 | **15.3** | 166 | 0.09 |
| Qwen 3.5-27B | Dense | Q6_K | 20.9 | 1 | 785 | 15.1 | 179 | 0.08 |
| Gemma 4 31B | Dense | Q8_0 | 30.4 | 2 | 654 | **14.1** | 139 | 0.10 |
| Gemma 4 31B | Dense | Q6_K | 20.9 | 1 | 673 | 13.1 | 179 | 0.07 |
| DeepSeek-R1 70B Distill | Dense | Q4_K_M | 39.6 | 2 | 336 | 11.5 | 185 | 0.06 |
| Llama 3.3-70B Instruct | Dense | Q4_K_M | 39.6 | 2 | 338 | 11.5 | 186 | 0.06 |

**Bolded rows** are the flagship recommendations: Qwen 3.6-35B-A3B at UD-Q4_K_M is the current best "smart and fast" single-card option.

Asterisked (*) avg-watt rows were loaded from NAS over CIFS (~100 MB/s), so the avg window includes load time where the GPU was near-idle. Peak watts (247, 244 respectively) reflect actual under-load draw.

Q8_0 at 15.3 t/s on Qwen 27B and 14.1 t/s on Gemma 31B dual are the post-fix numbers; previously 4.88 and 4.1 t/s respectively, before PRs [#21527](https://github.com/ggerganov/llama.cpp/pull/21527) and [#21638](https://github.com/ggerganov/llama.cpp/pull/21638) landed. See [upstream-contributions.md](upstream-contributions.md) and the Q8_0 story section below.

---

## The Quantization Sweep - Qwen 3.5-27B

> **v0.1 baseline** (build `25eec6f`, pre-NDEBUG-fix). The master table above has the current Q4_K_M, Q6_K and Q8_0 numbers on Qwen 3.5-27B. The remaining five quants (Q4_0, Q4_K_S, IQ4_XS, Q4_1, Q5_K_M, Q5_K_S) haven't been re-run yet; prefill values here are understated by ~50% relative to the current build (tg128 is less affected).

Same model, same hardware, eight quantizations. Sorted by generation speed:

| Quant | Method | Size (GiB) | pp512 t/s | tg128 t/s | BW util | Verdict |
|-------|--------|-----------|-----------|-----------|---------|---------|
| Q4_0 | Legacy round-to-nearest | 14.63 | 243 | **23.67** | 57% | Fast |
| Q4_K_S | K-quant 4-bit small | 14.68 | 309 | **23.05** | 56% | Fast |
| Q4_K_M | K-quant 4-bit mixed | 15.58 | 302 | **20.56** | 53% | Fast - recommended |
| IQ4_XS | Importance 4-bit XS | 13.94 | 267 | 17.52 | 40% | Medium |
| Q4_1 | Legacy with offset | 15.99 | 259 | 16.78 | 44% | Medium |
| Q6_K | K-quant 6-bit | 20.90 | 304 | 13.83 | 48% | Medium |
| Q5_K_M | K-quant 5-bit mixed | 18.25 | 300 | 13.78 | 41% | Medium |
| Q5_K_S | K-quant 5-bit small | 17.58 | 307 | 13.50 | 39% | Medium |

Q8_0 and IQ4_NL rows were captured pre-fix and are preserved in the Q8_0 story section below, not here, so the sweep shows current Xe2 behavior rather than a snapshot of a now-patched bug.

**Bandwidth utilization** = (theoretical bytes-per-token read) / (608 GB/s × 1s) × (actual tg / theoretical tg). Higher is better - a value of 57% means the kernel is extracting ~57% of the card's rated memory bandwidth, which is excellent for llama.cpp-class workloads.

**Q4_K_M is the sweet spot.** Small enough to fit comfortably on one card with KV headroom, fast dequantization kernel, minimal quality loss vs Q6/Q8 for most uses.

---

## SYCL vs Vulkan - Same Hardware, Same Model

> **v0.1 baseline.** Both sides used the pre-NDEBUG-fix build. The SYCL vs Vulkan *ratio* is the interesting signal here and should still hold on the current build; raw throughput on SYCL side is now ~50% higher (tg128 on 1.5B is 216 vs the 229 shown below).

Qwen 2.5-1.5B Q4_K_M, single B70, layers 99:

| Test | Vulkan | SYCL | SYCL/Vulkan |
|------|--------|------|-------------|
| pp512 | 5,468 t/s | 5,313 t/s | 0.97× (noise) |
| tg128 | 102 t/s | 229 t/s | **2.24×** |

SYCL generation is **2.2× faster** than Vulkan on the same hardware. The entire gap comes from SYCL's MMVQ + reorder path, which Vulkan doesn't have.

**Don't use Vulkan for B70 token generation.** Prompt processing is a wash; decode loses a factor of 2.

### Coopmat control (Vulkan only)

| Quant | Coopmat | pp512 | tg128 |
|-------|---------|-------|-------|
| Q8_0 | Enabled | 418 | 5.37 |
| Q8_0 | Disabled (`GGML_VK_DISABLE_COOPMAT=1`) | 244 | 5.32 |
| Q4_K_M | Enabled | 398 | 10.71 |

Coopmat helps Vulkan prompt processing (~70%) but doesn't move generation. Vulkan Q4_K_M decode (10.7 t/s) is still ~2× slower than SYCL (20.6 t/s) - the gap is SYCL's MMVQ+reorder, not coopmat availability.

---

## F16 Accumulation Mode - Free Prefill Speedup

> **v0.1 baseline.** Comparison of `-DGGML_SYCL_F16=ON` vs OFF on the old build. The conclusion (turn it on, always) still holds; current builds use F16 by default.

Rebuild llama.cpp SYCL with `-DGGML_SYCL_F16=ON`. FP16 halves the accumulator size and doubles throughput on Xe2's XMX engines, which have native FP16 support.

| Model | Quant | FP32 pp512 | **F16 pp512** | Change | FP32 tg128 | F16 tg128 | tg change |
|-------|-------|-----------|---------------|--------|-----------|----------|-----------|
| Qwen 27B | Q8_0 | 296 | **707** | **+139%** | 4.97 | 4.95 | ~0% |
| Qwen 27B | Q4_K_M | 302 | **725** | **+140%** | 20.56 | 19.72 | -4% |
| Gemma 31B | Q4_K_M | 255 | **704** | **+176%** | 22.6 | 21.84 | -3% |
| Qwen 35B-A3B MoE | Q4_K_M | 573 | 585 | +2% | 38.9 | 38.77 | ~0% |

**F16 gives 2.4-2.8× prompt processing speedup on dense models at a <5% decode regression.** MoE barely moves because its prefill is already compute-bound elsewhere.

**Recommendation: always build with F16 on.** The only reason not to is if you specifically need FP32 accumulation for a research-precision workload. For practical inference, keep it on.

---

## Context Length Scaling

Does performance degrade at longer contexts? Tested at pp512 / pp1024 / pp2048 / pp4096 (F16 mode, single GPU):

**Qwen 3.5-27B Q4_K_M (dense, 15.58 GiB):**

| Context | pp t/s | vs pp512 |
|---------|--------|----------|
| 512 | 721 | - |
| 1024 | 725 | +0.6% |
| 2048 | 715 | -0.8% |
| 4096 | 712 | -1.2% |
| tg128 | 19.62 | - |

**Qwen 3.5-35B-A3B MoE Q4_K_M (20.49 GiB):**

| Context | pp t/s | vs pp512 |
|---------|--------|----------|
| 512 | 611 | - |
| 1024 | 593 | -2.9% |
| 2048 | 587 | -3.9% |
| tg128 | 38.41 | - |

Dense model is essentially flat to 4K. MoE drops ~4% at 2K - slightly more KV-cache pressure, still excellent.

---

## The Q8_0 Story (Pre-Fix Baseline)

**Historical.** The numbers in this section are from before PRs [#21527](https://github.com/ggerganov/llama.cpp/pull/21527) and [#21638](https://github.com/ggerganov/llama.cpp/pull/21638) merged upstream. They're preserved as the investigation log that motivated the fix. Post-fix Q8_0 numbers will be slotted back into the Master Table above when re-benchmarked; current behavior on recent llama.cpp master is comparable to Q4_K_M.

Originally Q8_0 ran 4× slower than Q4_K_M on dense models (4.88 vs 20.56 t/s on Qwen 27B). Not a VRAM pressure issue - Qwen 9B Q8_0 fits with 22 GiB free and hit only 16.5 t/s at the time.

### Kernel path comparison

| Quant | Path | How forced | tg128 t/s |
|-------|------|-----------|-----------|
| Qwen 27B Q4_K_M | MMVQ+reorder | Default | 20.56 |
| Qwen 27B Q4_K_M | DMMV | `GGML_SYCL_PRIORITIZE_DMMV=1` | 12.38 |
| Qwen 27B Q8_0 | DMMV | Default (only option upstream) | 4.97 |
| Qwen 27B Q8_0 | MMVQ | Patched `ggml_sycl_supports_dmmv` | 4.33 |

MMVQ is 1.66× faster than DMMV for Q4_K_M. But Q8_0 is slow on **both** paths - the kernel itself is inefficient on Xe2.

### Driver doesn't fix it

Tested Intel Compute Runtime 26.05→26.09 + IGC 2.28→2.30 with clean rebuild:

| Model | Old pp512 | New pp512 | Old tg128 | New tg128 |
|-------|----------|-----------|----------|----------|
| Qwen 27B Q8_0 | 707 | **771 (+9%)** | 4.97 | 4.98 (0%) |
| Qwen 27B Q4_K_M | 725 | 733 (+1%) | 20.56 | 19.88 (-3%) |

Driver update gave Q8_0 a 9% pp bump but **zero tg improvement**. Confirms the Q8_0 bottleneck is in llama.cpp's SYCL kernel, not in the Intel driver or compiler.

### Our upstream fix

PR [#21527](https://github.com/ggerganov/llama.cpp/pull/21527) (merged) added Q8_0 reorder, lifting tg from 4.88 → 15.24 t/s (3.1× speedup). A follow-up GEMM bug found post-merge was fixed in [#21638](https://github.com/ggerganov/llama.cpp/pull/21638). See [upstream-contributions.md](upstream-contributions.md) for the full story.

---

## MoE vs Dense - What Actually Matters

| Model | Type | Active params | Total size | pp512 | tg128 |
|-------|------|---------------|-----------|-------|-------|
| Gemma 4 26B-A4B | MoE | ~4B | 15.7 GiB | 943 | 30.1 |
| Qwen 3.5-35B-A3B | MoE | ~3B | 20.5 GiB | 573 | 38.9 |
| Qwen 3.5-27B | Dense | 27B | 15.6 GiB | 302 | 20.6 |
| Gemma 4 31B | Dense | 31B | 17.1 GiB | 255 | 22.6 |

MoE architectures win on this hardware. Only a fraction of params activate per token, so per-token compute is low. Qwen 35B-A3B is our single-card champion (38.9 t/s) despite being the largest model that fits - it's running like a 3B-dense model in terms of compute-per-token.

**The caveat:** compared to NVIDIA CUDA, Intel SYCL's MoE dispatch is less efficient. On a 3090, Gemma 26B-A4B runs at 134 t/s (18% faster than 9B dense). On the B70, the same MoE runs at 30 t/s - **slower** than 9B dense (54 t/s). MoE still wins on quality-per-second, but the SYCL MoE path has headroom. See [cross-card-comparison.md](cross-card-comparison.md) for the full table.

---

## Other Backend Experiments

### DNNL (oneDNN) path

Built with and without `-DGGML_SYCL_DNN=ON`. DNNL path is currently off for dense-model inference in our builds - no measurable speedup for our configs, and the DNNL INT4 GEMM path is still being investigated. Not recommended to add for B70 users yet.

### Row-split multi-GPU

`--split-mode row` segfaults on model load on our dual-B70 setup with SYCL. Reproduces on pristine master, so it's pre-existing. Workaround: use `--split-mode layer` (the default). Multi-GPU details in [multi-gpu.md](multi-gpu.md).

---

## Quick Recommendations

| Use case | Model | Quant | GPUs | Expected tg | t/J |
|----------|-------|-------|------|-------------|-----|
| **Fast & smart** | Qwen 3.6-35B-A3B | UD-Q4_K_M | 1 | **54.7 t/s** | 0.48 |
| **Most efficient single-card** | Qwen 3.5-35B-A3B | Q4_K_M | 1 | 54.5 t/s | **0.59** |
| **Fastest mid-size dense** | Phi-4 14B | Q4_K_M | 1 | 43.7 t/s | 1.08 |
| **Fastest MoE** | Gemma 4 26B-A4B | Q4_K_M | 1 | 52.6 t/s | 0.52 |
| **Mid-size dense (24B)** | Mistral Small 3.2-24B | Q4_K_M | 1 | 30.1 t/s | 0.18 |
| **Simple 27B dense** | Qwen 3.5-27B | Q4_K_M | 1 | 20.4 t/s | 0.11 |
| **Coding, maximum quality** | Qwen3-Coder-Next 80B-A3B | Q4_K_M | 2 | 43.4 t/s | 0.55 |
| **70B dense** | DeepSeek-R1-70B / Llama 3.3-70B | Q4_K_M | 2 | 11.5 t/s | 0.06 |
| **Small/draft** | Qwen 2.5-1.5B | Q4_K_M | 1 | 216 t/s | 1.68 |

Avoid: IQ4_NL on dense models (v0.1 data showed it broken; not yet re-tested on current build), `--split-mode row` (SYCL segfault), Vulkan for decode on anything bigger than ~2B.
