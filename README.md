# Intel Arc Pro B70 - Benchmarks & Performance Data

Early, data-dense **benchmarks and performance data** for the **Intel Arc Pro B70** (Xe2 / Battlemage, 32 GB GDDR6 ECC, $949). Both single-card and dual-card (2× B70, 64 GB total) configurations, tested on llama.cpp SYCL + Vulkan, vLLM XPU, and several video-generation pipelines.

Public data on the B70 is scarce. We're early adopters and hit every rough edge - this repo collects the numbers, the gotchas, and the fixes so the next person doesn't have to rediscover them.

All numbers here are real runs on our workstation. No synthetic estimates.

---

## TL;DR - Headline Numbers

**Best single-card result:** Qwen 3.5-35B-A3B (MoE) Q4_K_M - **38.9 t/s** generation, 573 t/s prompt processing.
**Best dual-card result:** DeepSeek-R1-Distill-Llama-70B Q4_K_M - **11.3 t/s** generation across both cards.
**Biggest model we've run:** 80B MoE (Qwen3-Coder-Next 80B-A3B Q4_K_M, 48 GiB) at 42.4 t/s on dual B70.

| Model | Type | Quant | Size | GPUs | pp512 t/s | tg128 t/s |
|-------|------|-------|------|------|-----------|-----------|
| **Qwen 3.5-35B-A3B** | **MoE** | **Q4_K_M** | **20.5 GiB** | **1** | **573** | **38.9** |
| Qwen3-Coder-Next 80B-A3B | MoE | Q4_K_M | 45.1 GiB | 2 | 298 | 42.4 |
| Gemma 4 26B-A4B | MoE | Q4_K_M | 15.7 GiB | 1 | 943 | 30.1 |
| Qwen 3.5-35B-A3B | MoE | Q8_0 | 34.4 GiB | 2 | 463 | 28.7 |
| Gemma 4 31B | Dense | Q4_K_M | 17.1 GiB | 1 | 255 | 22.6 |
| Qwen 3.5-27B | Dense | Q4_K_M | 15.6 GiB | 1 | 302 | 20.6 |
| Qwen 3.5-9B | Dense | Q4_K_M | 5.3 GiB | 1 | 1,038 | 54.4 |
| DeepSeek-R1 70B | Dense | Q4_K_M | 39.6 GiB | 2 | 120 | 11.3 |
| Qwen 2.5-1.5B | Dense | Q4_K_M | 1.0 GiB | 1 | 5,313 | 229 |

Full tables, quant sweeps, context scaling and raw data in [llm-benchmarks.md](llm-benchmarks.md).

---

## Key Findings

1. **SYCL is the right backend, not Vulkan.** SYCL generation is **2.2× faster** than Vulkan on the same hardware (229 vs 102 t/s on Qwen 1.5B Q4_K_M). Vulkan prefill is fine, but SYCL's MMVQ + reorder path wins decode.
2. **MoE architectures are the sweet spot.** Only ~3-4B params active per token, so you get large-model quality at small-model speed. Qwen 35B-A3B hits 38.9 t/s on a single card - faster than a 27B dense at 20.6 t/s.
3. **Q4_K_M is the sweet spot quant for dense models.** Q8_0 tanks generation to ~5 t/s due to a kernel inefficiency we traced and partially fixed upstream. K-quants (Q4_K_M, Q5_K_M, Q6_K) all generate in the 14-23 t/s range on 27B dense.
4. **F16 accumulation mode gives free ~2.5× prompt speedup.** Build with `-DGGML_SYCL_F16=ON` - pp512 goes from 302 to 725 t/s on Qwen 27B Q4_K_M. Decode unchanged.
5. **Dual-card layer split doesn't speed up models that fit one card.** It's sequential by design, not a bug. Use it to fit bigger models (70B dense, 80B MoE), or run two independent models on the two cards.
6. **Usable across a lot of model classes.** 70B dense at 11 t/s, 80B MoE at 42 t/s, video generation up to 720p LTX + 832×480 Wan 2.2 5B. The 32 GB of VRAM per card is the main feature.

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
