# Intel Arc Pro B70 - Benchmarks & Performance Data

Early, data-dense **benchmarks and performance data** for the **Intel Arc Pro B70** (Xe2 / Battlemage, 32 GB GDDR6 ECC, $949). Both single-card and dual-card (2× B70, 64 GB total) configurations, tested on llama.cpp SYCL + Vulkan, vLLM XPU, and several video-generation pipelines.

Public data on the B70 is scarce. We're early adopters and hit every rough edge - this repo collects the numbers, the gotchas, and the fixes so the next person doesn't have to rediscover them.

All numbers here are real runs on our workstation. No synthetic estimates.

---

## TL;DR - Headline Numbers

**Best single-card MoE:** Qwen 3.6-35B-A3B UD-Q4_K_M - **54.7 t/s** generation, 615 t/s prompt processing, 114 W avg.
**Best dual-card model:** DeepSeek-R1-Distill-Llama-70B Q4_K_M - **11.5 t/s** generation across both cards.
**Biggest model we've run:** Qwen3-Coder-Next 80B-A3B Q4_K_M (45 GiB) at 43.4 t/s on dual B70, 79 W avg.

All numbers below are from a single commit-pinned llama.cpp build (`ec6f7a6a5c`, 2026-04-21) with power telemetry captured during every run. See [CHANGELOG.md](CHANGELOG.md) for the upgrade from the older data and why it's different.

| Model | Type | Quant | Size | GPUs | pp512 t/s | tg128 t/s | avg W | t/J |
|-------|------|-------|------|------|-----------|-----------|-------|-----|
| **Qwen 3.6-35B-A3B** | **MoE** | **UD-Q4_K_M** | **20.6 GiB** | **1** | **615** | **54.7** | **114** | **0.48** |
| Qwen 3.5-35B-A3B | MoE | Q4_K_M | 20.5 GiB | 1 | 618 | 54.5 | 92 | 0.59 |
| Gemma 4 26B-A4B | MoE | Q4_K_M | 15.7 GiB | 1 | 1,129 | 52.6 | 102 | 0.52 |
| Qwen3-Coder-Next 80B-A3B | MoE | Q4_K_M | 45.1 GiB | 2 | 305 | 43.4 | 79 | 0.55 |
| Qwen 3.6-35B-A3B | MoE | Q8_0 | 34.3 GiB | 2 | 458 | 36.5 | 91 | 0.40 |
| Gemma 4 31B | Dense | Q4_K_M | 17.1 GiB | 1 | 601 | 21.7 | 169 | 0.13 |
| Qwen 3.5-27B | Dense | Q4_K_M | 15.6 GiB | 1 | 718 | 20.4 | 178 | 0.11 |
| Qwen 3.5-27B | Dense | Q6_K | 20.9 GiB | 1 | 785 | 15.1 | 179 | 0.08 |
| Qwen 3.5-27B | Dense | Q8_0 | 26.6 GiB | 1 | 776 | **15.3** | 166 | 0.09 |
| Gemma 4 31B | Dense | Q8_0 | 30.4 GiB | 2 | 654 | **14.1** | 139 | 0.10 |
| DeepSeek-R1 70B | Dense | Q4_K_M | 39.6 GiB | 2 | 336 | 11.5 | 185 | 0.06 |
| Qwen 3.5-9B | Dense | Q4_K_M | 5.3 GiB | 1 | 2,302 | 60.2 | 168 | 0.36 |
| Qwen 3.5-9B | Dense | Q8_0 | 8.9 GiB | 1 | 2,444 | 48.0 | 149 | 0.32 |
| Qwen 2.5-1.5B | Dense | Q4_K_M | 1.0 GiB | 1 | 8,048 | 216.4 | 129 | 1.68 |

**t/J** = tg128 tokens per joule of GPU energy consumed (higher = more efficient). Sum of all per-card power.

Full tables, quant sweeps, context scaling and raw data in [llm-benchmarks.md](llm-benchmarks.md). Per-run JSON in [data/llm/](data/llm/).

---

## Key Findings

1. **SYCL is the right backend, not Vulkan.** SYCL generation is **2.2× faster** than Vulkan on the same hardware (229 vs 102 t/s on Qwen 1.5B Q4_K_M). Vulkan prefill is fine, but SYCL's MMVQ + reorder path wins decode.
2. **MoE architectures are the sweet spot.** Only ~3-4B params active per token, so you get large-model quality at small-model speed. Qwen 3.5 and 3.6 35B-A3B both hit ~54 t/s on a single card, faster than the 9B dense at 60 t/s per watt-equivalent.
3. **Qwen 3.6 trades efficiency for quality on MoE.** Qwen 3.6-35B-A3B UD-Q4_K_M generates at 54.7 t/s on a single card at 114 W, matching Qwen 3.5 35B's speed on a newer base model. This is the first public B70 data on Qwen 3.6.
4. **Q8_0 on dense models is fixed upstream and the fix is big.** PR #21527 + #21638 brought Qwen 27B Q8_0 from 4.88 to 15.3 t/s (3.13×) and Gemma 4 31B Q8_0 dual from 4.1 to 14.1 t/s (3.4×). K-quants and Q8_0 now all generate in the 14-22 t/s range on 27B/31B dense.
5. **F16 accumulation mode + clean NDEBUG build gives ~2× prompt speedup.** Prior public numbers were suppressed ~50% on prefill because the shipped `llama-cpp-daily` build had asserts enabled (empty `CMAKE_CXX_FLAGS_RELEASE`). Our current build strips asserts via `-DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG"`.
6. **Dual-card layer split doesn't speed up models that fit one card.** It's sequential by design, not a bug. Use it to fit bigger models (70B dense, 80B MoE), or run two independent models on the two cards.
7. **80B MoE on consumer hardware works.** Qwen3-Coder-Next 80B-A3B Q4_K_M runs at 43.4 t/s across the pair, drawing only 79 W average. MoE's low per-token compute makes PCIe transfer overhead negligible.
8. **Per-token power is low on MoE.** Tokens-per-joule on the MoE models (0.4-0.6 t/J single-card) is 3-4× higher than dense 27B+ models. Translates directly to cheaper inference.

---

## Directory

| File | What's in it |
|------|--------------|
| [hardware.md](hardware.md) | Specs, PCIe topology, driver/kernel versions, how cards are wired |
| [methodology.md](methodology.md) | How tests were run, software stack, llama.cpp build flags |
| [llm-benchmarks.md](llm-benchmarks.md) | Full SYCL LLM results: model sweep, quant sweep, F16 mode, context scaling, backend comparison |
| [multi-gpu.md](multi-gpu.md) | Dual-card analysis: layer split vs tensor parallel, DeepSeek 70B, when dual-GPU helps |
| [cross-card-comparison.md](cross-card-comparison.md) | B70 vs RTX 3090 / 3080 Ti / 3060 / 4060 / BC-250 / Apple M4 on the same model |
| [video-generation.md](video-generation.md) | LTX-Video, Wan 2.1, Wan 2.2 5B, Wan 2.2 A14B - resolutions, durations, OOM points |
| [engine-comparison.md](engine-comparison.md) | llama.cpp SYCL vs vLLM XPU - prefill (XMX), decode, model coverage |
| [upstream-contributions.md](upstream-contributions.md) | llama.cpp PRs we filed fixing B70/SYCL issues (#21527, #21580, #21597, #21638, #21700) |
| [data/](data/) | Raw JSON benchmark results |

---

## Hardware Summary

- **GPU:** 2× Intel Arc Pro B70, BMG-G31 (full Big Battlemage die), 32 GB GDDR6 ECC per card, 608 GB/s bandwidth per card
- **CPU:** AMD Ryzen 5 9600X (Zen 5, 6C/12T)
- **RAM:** 60 GB DDR5
- **PCIe:** Both cards on CPU-direct PCIe 4.0 x8 via B850 chipset bifurcation
- **OS:** Ubuntu 26.04, Kernel 7.0.0-10-generic
- **Driver:** `xe` (in-tree, not i915)
- **Compute stack:** Intel oneAPI DPC++ 2025.3.3, Level Zero, IGC 2.30

Full details: [hardware.md](hardware.md).

---

## Related Projects

- [llmresults.com](https://llmresults.com) - Interactive cross-hardware benchmark comparison (same JSON data, different UI)
- Our llama.cpp upstream PRs: see [upstream-contributions.md](upstream-contributions.md)

---

## Contributing / Corrections

If you're running a B70 and seeing different numbers, open an issue - that's genuinely useful data. Same goes for model/config requests we haven't tested yet.

## License

MIT. See [LICENSE](LICENSE).
