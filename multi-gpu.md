# Dual Arc Pro B70 - Multi-GPU Performance

Two B70s = 64 GB total VRAM and 1,216 GB/s combined bandwidth on paper. In practice, how the cards work together depends entirely on how llama.cpp (or whatever engine) splits the work. This page lays out what we measured and what actually makes sense.

---

## Short Version

1. **Layer split is sequential.** GPU 0 computes layers 0-N/2, hands the activation to GPU 1 via PCIe, GPU 1 computes layers N/2-N, hands back. Both GPUs sit idle half the time. At most 1× single-GPU speed - you're not doubling compute, you're fitting a bigger model.
2. **This isn't Intel's fault.** The same pattern holds on 2× RTX 4090 with NVLink. It's how llama.cpp's layer split works. Confirmed upstream and community-wide.
3. **Models that fit one card should stay on one card.** Dual-GPU layer split makes them 4% slower, not faster.
4. **Models that don't fit one card work well enough.** DeepSeek-R1 70B Q4_K_M (39.6 GiB) runs at 11.3 t/s across both cards - usable for interactive chat.
5. **MoE is the "use both cards" sweet spot.** Per-token compute is so low that PCIe overhead doesn't bite. 80B MoE (Qwen3-Coder-Next) hits 42 t/s across both cards.
6. **For true parallel speedup, use vLLM.** Tensor parallelism actually splits each layer across both cards simultaneously. See [engine-comparison.md](engine-comparison.md).

---

## How llama.cpp Layer Split Actually Works

`--split-mode layer` (the default and the only working mode on our B70s - see note below):

1. GPU 0 processes layers 0 through N/2
2. GPU 0 sends the hidden state (~4-16 KB per token) to GPU 1 via PCIe
3. GPU 1 processes layers N/2 through N
4. GPU 1 sends the result back to GPU 0
5. Repeat for every single token

Each GPU idles while the other computes. You get at most 1× single-GPU speed plus PCIe transfer overhead. This is architectural, not implementation-specific.

**Note on row-split:** `--split-mode row` segfaults on model load on our dual-B70 setup with SYCL. Reproduces on pristine master, so it's a pre-existing SYCL bug, not something our PRs caused. Use `layer` (the default).

---

## The Evidence

### Model that fits one card - layer split hurts

**Qwen 3.5-27B Q4_K_M (15.6 GiB) - fits on one 32 GB card with 16 GB headroom:**

| Setup | pp512 | tg128 |
|-------|-------|-------|
| 1 GPU | 718 t/s | 20.4 t/s |
| 2 GPUs (layer split 50/50) | *v0.1: 306 t/s / 19.7 t/s — pending re-run at current build* | |

tg128 drops ~4% with dual GPU on this model (v0.1 data). PCIe round-trip per token is a small cost, but it's not free. **Don't split this model.**

### Model that doesn't fit one card - this is where it pays off

Two 70B-class dense models, both needing dual-card:

| Model | Size | pp512 t/s | **tg128 t/s** | avg W |
|-------|------|-----------|---------------|-------|
| DeepSeek-R1-Distill-Llama-70B Q4_K_M | 39.6 GiB | 336 | **11.5** | 185 |
| Llama 3.3-70B Instruct Q4_K_M | 39.6 GiB | 338 | **11.5** | 186 |

Both 70B dense models land at the same ~11.5 t/s ceiling - same architecture, same layer-split pattern, hitting the same memory-bandwidth-bound decode speed. Effective bandwidth is 446 GB/s (37% of combined 1,216 GB/s). The ~30% efficiency drop vs single-card is the normal layer-split penalty seen on all platforms including NVIDIA.

**This is the headline dual-GPU result.** A 70B dense model running at interactive speed on two consumer-tier GPUs, drawing ~185 W average across both cards.

### MoE that fits one card - single card still wins

**Qwen 3.5-35B-A3B Q4_K_M (20.5 GiB):** 1 GPU gives 618 pp / 54.5 tg; dual-card layer split would add overhead with no benefit since the model fits.

**Qwen 3.6-35B-A3B UD-Q4_K_M (20.6 GiB):** 1 GPU gives 615 pp / **54.7 tg**. Same pattern - single card wins.

### MoE that needs both cards - great

| Model | Size | pp512 | **tg128** | avg W |
|-------|------|-------|-----------|-------|
| Qwen 3.5-35B-A3B Q8_0 | 34.4 GiB | 458 | **36.5** | 91 |
| Qwen 3.6-35B-A3B Q8_0 | 34.3 GiB | 458 | **36.5** | 91 |
| Qwen3-Coder-Next 80B-A3B Q4_K_M | 45.1 GiB | 305 | **43.4** | 79 |

MoE works great across both cards because per-token active compute is ~3B params regardless of total size. PCIe hidden-state transfer is the same 4-16 KB per token - negligible when each layer is fast. Notable: dual-card MoE runs draw **80-90 W average across both cards combined** - lower than a single dense model running single-card.

---

## Why MoE Survives Multi-GPU Overhead

MoE isn't faster because of multi-GPU - it's fast **despite** multi-GPU overhead:

1. **Per-token weight reads are tiny** - only ~3 GiB of the 45 GiB file is read per token (the active experts)
2. **Each GPU computes its local experts independently** - less cross-GPU dependency per layer
3. **The sequential layer processing still happens**, but each layer is fast because so few parameters are active
4. **PCIe hidden-state transfer** (~4-16 KB) is negligible compared to per-layer compute time

Dense models read **all** their weights per token, so bandwidth limits them and the PCIe stop-and-wait hurts more proportionally.

---

## PCIe x8 vs x16 - Does It Matter?

Both our B70s run at CPU-direct PCIe 5.0 x8 (~32 GB/s each direction). The cards are PCIe 5.0 x16 capable; width negotiates down to x8 per card because the B850 bifurcates the CPU lanes x8/x8.

| Scenario | x8 vs x16 impact | Why |
|----------|------------------|-----|
| Single-GPU inference | <2% | Compute stays on-GPU, PCIe isn't in the loop |
| Multi-GPU layer split | ~20-40% penalty | Activations transfer between GPUs at every split point |
| Multi-GPU tensor parallel (vLLM) | Larger | Every operation exchanges between GPUs |

The actual bytes per PCIe transfer are small (4-16 KB hidden state), but the **latency** of each round-trip adds up when you have dozens of layers and thousands of tokens. Full x16 would help marginally for multi-GPU, but the sequential nature of layer split is the fundamental limit, not the bandwidth — and at PCIe 5.0 x8 we have plenty of headroom for the small transfers that layer split actually does.

---

## The Real Value of Two B70 Cards

### Scenario 1: Bigger models via layer split

- **Dense Q4 ≥ 32 GB:** usable. 70B Q4_K_M at 11.3 t/s.
- **Dense Q8 ≥ 32 GB:** not recommended. 4 t/s regardless.
- **MoE:** excellent. Active params are small, so PCIe overhead barely matters.
- **Layer split on models that fit one card:** don't do it. Single GPU is faster.

### Scenario 2: Two independent models

- One model per card, two separate requests
- Each gets full 32 GB VRAM and full 608 GB/s
- Example: Qwen 35B-A3B serving chat on GPU 0, Gemma 26B-A4B handling a background summarization pipeline on GPU 1
- **Full single-card performance per card. No overhead.**

### Scenario 3: Batch/server throughput (vLLM)

- vLLM with tensor parallelism (TP=2) does split each layer across both cards simultaneously
- Better concurrency, better overall throughput
- But single-stream decode speed is comparable to layer-split llama.cpp (see [engine-comparison.md](engine-comparison.md))
- **Best for multi-user serving.**

### Scenario 4: Giant MoE

- Qwen3-Coder-Next 80B-A3B at 42 t/s - the flagship dual-B70 result
- 80B parameter brain, 3B active per token, both cards working together
- This is what the hardware exists for

---

## What Would Actually Help

**Tensor parallelism in llama.cpp SYCL.** Split each layer's matmul across both cards so they compute simultaneously. Would roughly double dense-model decode. Not implemented yet for SYCL.

**Better Q8_0 kernels.** See PR #21527 / #21638 in [upstream-contributions.md](upstream-contributions.md) - the Q8_0 cliff is partially fixed. Dual-GPU Q8 dense models would look much better with efficient Q8 kernels.

**Direct GPU-to-GPU transfers** (P2P DMA). Multi-GPU RAM consumption on our setup is a separate topic - fixed in PR #21597 by swapping `sycl::malloc_device` for Level Zero's `zeMemAllocDevice`.

---

## System RAM During Dual-GPU Inference

On unpatched upstream, dual-GPU SYCL inference used **~60 GiB of system RAM** during model load due to an `xe` driver TTM issue. Our PR [#21597](https://github.com/ggerganov/llama.cpp/pull/21597) fixed this by switching the allocator. Post-fix: ~6.7 GiB system RAM for the same workload. If you're running dual-B70 and seeing your system RAM fill up, check whether your llama.cpp build has the Level Zero allocator path enabled. Details in [upstream-contributions.md](upstream-contributions.md).

---

## Related

- [llm-benchmarks.md](llm-benchmarks.md) - single-card numbers
- [engine-comparison.md](engine-comparison.md) - vLLM tensor parallel vs llama.cpp layer split
- [upstream-contributions.md](upstream-contributions.md) - PR #21597 and the xe/TTM fix
