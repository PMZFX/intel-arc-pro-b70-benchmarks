# Key Findings - Intel Arc Pro B70

Summary of the non-obvious things this repo documents. All numbers reference commit `ec6f7a6a5c` (llama.cpp b8840-12-gec6f7a6a5-dirty, 2026-04-21) with explicit NDEBUG compile flags, F16 SYCL accumulation on, flash attention on, and dual 32 GB B70 workstation. See [CHANGELOG.md](CHANGELOG.md) for the full run log and [data/llm/](data/llm/) for per-benchmark JSON.

---

## 1. Every B70 benchmark on the internet is ~50% low on prefill

The shipped `llama-cpp-daily` build configured for Release but with an **empty** `CMAKE_CXX_FLAGS_RELEASE`, so `NDEBUG` never made it to the C++ compiler. `llama-bench` then ran with GGML_ASSERTs active and printed `warning: asserts enabled, performance may be affected` on every startup. Most readers ignore that line. It costs a lot.

Clean rebuild with explicit `-DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG"` versus the same binary pre-fix, all else equal:

| Model | Quant | pp512 old | pp512 new | Change |
|-------|-------|-----------|-----------|--------|
| Qwen 2.5-1.5B | Q4_K_M | 5,313 | 8,048 | +51% |
| Qwen 3.5-9B | Q4_K_M | 1,038 | 2,302 | +122% |
| Qwen 3.5-27B | Q4_K_M | 302 | 718 | +138% |
| Gemma 4 31B | Q4_K_M | 255 | 601 | +136% |
| Gemma 4 26B-A4B | Q4_K_M | 943 | 1,129 | +20% |
| DeepSeek-R1 70B dual | Q4_K_M | 120 | 336 | +180% |

Token generation moves much less (tg is memory-bandwidth-bound, not assert-sensitive) but prefill gets 50-180% better on dense models. This is the single biggest reason existing public B70 numbers understate the card.

**Action if you're benchmarking a B70:** always check `llama-bench -p 0 -n 0 2>&1 | head` for the `asserts enabled` warning before trusting numbers. If it's there, rebuild.

---

## 2. PR #21527's Q8_0 fix isn't Qwen-specific

Our upstream fix for Q8_0 SYCL reorder (merged as PR #21527, follow-up #21638) was characterized on Qwen 27B. One reasonable criticism was that the speedup might only show on Qwen's specific weight layout.

We now have two independent validations on the post-fix build:

| Model | Config | Before (pre-#21527) | After | Speedup |
|-------|--------|---------------------|-------|---------|
| Qwen 3.5-27B Q8_0 | single-card | 4.88 t/s | 15.28 t/s | **3.13×** |
| Gemma 4 31B Q8_0 | dual-card | 4.13 t/s | 14.07 t/s | **3.40×** |

Different model family, different card count, same magnitude of speedup. The Q8_0 kernel inefficiency was a general issue and the fix is correct. If you had written off Q8_0 on a B70, it's worth revisiting.

---

## 3. First public Qwen 3.6 data on any Intel GPU

| Variant | GPUs | pp512 | tg128 | avg W | t/J |
|---------|------|-------|-------|-------|-----|
| Qwen 3.6-35B-A3B UD-Q4_K_M | 1 | 615 | **54.7** | 114 | 0.48 |
| Qwen 3.6-35B-A3B Q8_0 | 2 | 458 | 36.5 | 91 | 0.40 |

Compared to Qwen 3.5-35B-A3B on the same build:

| Variant | Qwen 3.5 | Qwen 3.6 | Δ |
|---------|----------|----------|---|
| Q4_K_M single-card tg | 54.5 | 54.7 | flat |
| Q8_0 dual-card tg | 28.7 | 36.5 | **+27%** |

Qwen 3.6 matches 3.5 at Q4_K_M but gains materially at dual-card Q8_0. Whatever architectural change Qwen made between 3.5 and 3.6 plays better with our dual-card SYCL path.

Long-context behavior is also excellent: at 64K with Q8_0 KV, tg still holds at 52.4 (see Finding 6).

---

## 4. MoE models are 3-5× more energy-efficient than dense

Tokens per joule of GPU energy consumed, selected rows sorted by class:

| Class | Model | t/J |
|-------|-------|-----|
| Small dense | Llama 3.1-8B Q4_K_M | 2.26 |
| Small dense | Qwen 2.5-1.5B Q4_K_M | 1.68 |
| Mid dense | Phi-4 14B Q4_K_M | 1.08 |
| **35B MoE** | **Qwen 3.5-35B-A3B Q4_K_M** | **0.59** |
| **80B MoE dual** | **Qwen3-Coder 80B-A3B Q4_K_M** | **0.55** |
| 26B MoE | Gemma 4 26B-A4B Q4_K_M | 0.52 |
| 35B MoE | Qwen 3.6-35B-A3B UD-Q4_K_M | 0.48 |
| Mid dense | Mistral Small 3.2-24B Q4_K_M | 0.18 |
| 27B dense | Qwen 3.5-27B Q4_K_M | 0.11 |
| 31B dense | Gemma 4 31B Q4_K_M | 0.13 |
| 70B dense dual | DeepSeek-R1 70B Q4_K_M | 0.06 |
| 70B dense dual | Llama 3.3-70B Instruct Q4_K_M | 0.06 |

Dual-card Qwen3-Coder 80B-A3B runs at **79 W average across both cards combined** while generating 43 t/s. A single-card dense 27B runs at 178 W for 20 t/s. Per token generated, the 80B MoE is **4.5× more energy-efficient** than the 27B dense, while being a significantly more capable model.

For homelab budgets and 24/7 inference loads, MoE is the play.

---

## 5. Two 70B dense models hit an identical ceiling

| Model | Size | pp512 | tg128 | avg W |
|-------|------|-------|-------|-------|
| DeepSeek-R1-Distill-Llama-70B Q4_K_M | 39.6 GiB | 336 | **11.5** | 185 |
| Llama 3.3-70B Instruct Q4_K_M | 39.6 GiB | 338 | **11.5** | 186 |

Two different fine-tunes of the same base architecture, split across the same two cards, hit the same number. The decode speed is memory-bandwidth bound, not model-specific. Effective cross-card bandwidth is about 446 GB/s of the combined theoretical 1,216 GB/s, about 37% utilization, which is the normal layer-split penalty on consumer GPUs (NVIDIA users see the same drop on dual-card layer-split setups).

**Implication:** if you want faster 70B dense inference on the B70, the path is tensor parallelism (which llama.cpp SYCL doesn't support yet), not a different model. vLLM XPU does support tensor parallelism and numbers there will look different. See [engine-comparison.md](engine-comparison.md).

---

## 6. tg128 is context-invariant up to at least 64K

Phase 2 swept pp 4K through 64K on three model classes. Decode speed barely moves:

| Model | pp512 | pp4K | pp8K | pp16K | pp32K | pp64K | tg (all) |
|-------|-------|------|------|-------|-------|-------|----------|
| Qwen 3.5-27B Q4_K_M | 718 | 629 | 560 | 445 | 310 | 195 | **20.4** |
| Qwen 3.6-35B-A3B UD-Q4_K_M | 615 | 579 | 559 | 527 | 485 | 411 | **54.5** |
| Gemma 4 31B Q4_K_M | 601 | 417 | 349 | 268 | 185 | - | **21.6** |

Per-token decode is bandwidth-bound and constant in N, so tg doesn't degrade with context. Prefill is O(N²) in compute so it drops as the prompt grows (more for dense, less for MoE because active params don't change).

**Practical implication:** you don't pay a decode tax for using long context. Submit a 32K RAG context to Qwen 3.6-35B and tokens still stream at 54 t/s.

**Aside:** Qwen 3.6's prefill is the most context-tolerant of the three models tested. From 4K to 64K, prefill drops 29% (579 to 411), while the same range costs Qwen 27B 69% (629 to 195). MoE's sparse attention is part of the win.

---

## 7. KV-cache quantization penalty is model-dependent

Phase 2 also tested `-ctk`/`-ctv` at Q8_0 and Q4_0:

| Model | fp16 tg (32K) | Q8_0 KV tg | Q4_0 KV tg |
|-------|---------------|------------|------------|
| Qwen 3.5-27B | 20.36 | 19.97 (-1.9%) | (not yet run) |
| Qwen 3.6-35B-A3B | 54.47 | 52.38 (-3.8%) | 52.31 (-4.0%) |
| Gemma 4 31B | 21.62 | 20.00 (-7.5%) | 19.77 (-8.6%) |

Prefill is unchanged across KV quant choices (kernel reads full-precision KV during prefill). Decode takes a hit that varies 2× across models.

Common advice like "Q4_0 KV is free" is reasonable for Qwen family but costs Gemma 4 31B almost 9% of its token rate. Benchmark on your model before committing.

**Practical implication:** at 32K context with Q4_0 KV, Qwen 3.6-35B still does 52 t/s, and the KV cache shrinks to 1/4 the fp16 size — big VRAM win without a big speed cost. Same trade on Gemma 31B costs more but often still worth it for fitting.

---

## 8. 131K contexts exceed our 30-min bench ceiling

Not a hardware limit. At 131K context, per-token prefill cost rises to ~10 ms (attention is O(N²) in compute, and flash attention doesn't change the compute complexity, only the memory complexity). So `llama-bench -p 131072 -r 3` needs about 60 minutes of pure compute, not counting load time.

Our Phase 2 bench-harness timeout is 30 min. So 131K variants failed with timeout on both Qwen 27B and Qwen 3.6 (and Gemma 4 31B 64K at `-r 3` also timed out because Gemma prefill is slower).

v0.3 will re-run 131K variants at `-r 1` (single pass), which brings total time to ~25 min and fits the budget. Real-world use doesn't submit 131K tokens in one shot anyway — KV cache grows incrementally at O(N) per token — so this is a benchmark-methodology issue, not a product limit.

---

## 9. Two-card parallel benchmarks produce identical numbers

Running independent `llama-bench` invocations on card 0 and card 1 simultaneously produced results within 1% of single-card-alone baselines. Validated in situ: Qwen 3.5-9B Q4_K_M ran 2,303 pp / 60.62 tg on card 1 while card 0 was grinding on a 64K Qwen 27B prefill, versus 2,302 pp / 60.22 tg when card 1 ran alone.

Why it works on this system:
- Each B70 has its own CPU-direct PCIe 5.0 x8 (no shared switch)
- 6 CPU cores, 12 threads; `llama-bench -t 1` uses 1 core per card
- 64 GB DDR5 means no RAM contention
- Each card has its own VRAM + memory controller
- Thermals fine: ~79°C pkg on the hotter card, ~45°C on the newer one

**Practical implication:** two-card parallel testing cuts sweep wall time roughly in half for single-card-fitting models. Our Phase 2 went from a 3-hour sequential plan to ~1.5 hours of parallel wall time.

Not yet implemented in the `SweepCoordinator` (which guards against concurrent sweeps on the same node), but trivially achievable via a direct llama-bench runner on card 1 while Race Bench drives card 0.

---

## 10. What's not in the master table on purpose

- **131K contexts** (methodology fix pending, see Finding 8)
- **Row-split dual-GPU** (SYCL crash on pristine master, not ours, see [upstream-contributions.md](upstream-contributions.md))
- **IQ4_NL on 27B** (pre-fix data showed it broken on Xe2; not yet re-tested on current build, probably still broken)
- **Vulkan for anything bigger than 2B** (SYCL is 2× faster on decode at every size tested)

---

## Reproducibility

All the numbers here came from one sweep session (2026-04-21) on one commit (`ec6f7a6a5c`, dirty tree with a local Q2_K DMMV WIP preserved rather than stashed). Rebuild command is in [sweeps/rebuild-llama-cpp-pinned.sh](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks) and the sweep configs that generated each JSON are referenced in the `sweep_id` field of every data file. Every file in [data/llm/](data/llm/) is commit-stamped.
