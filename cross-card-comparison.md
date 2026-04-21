# Intel Arc Pro B70 vs Other GPUs - Cross-Card LLM Benchmarks

Same model, every card we have. This is the data people actually want to see before buying.

**Fleet:** B70 (Intel, Xe2), RTX 3090 / 3080 Ti / 3060 / 4060 (NVIDIA, Ampere/Ada), BC-250 (AMD APU mining card), Apple M4 Mac Mini.

All tests single-card unless noted.

---

## Test 1 - Same Model, Every Card: Qwen 3.5-9B

The one model that runs on every card in the fleet. Each runs the best quant its VRAM allows.

| Card | Used price* | VRAM | Quant | pp512 t/s | tg128 t/s | $/(tg t/s) |
|------|-------------|------|-------|-----------|-----------|------------|
| **Intel Arc Pro B70** | **$949 (new)** | **32 GB** | **Q4_K_M** | **1,038** | **54.4** | **$17.44** |
| RTX 3080 Ti | ~$425 | 12 GB | Q4_K_M | 4,108 | 113.3 | $3.75 |
| RTX 3090 | ~$1,300 | 24 GB | Q4_K_M | 4,009 | 113.7 | $11.43 |
| RTX 3060 | ~$250 | 12 GB | Q4_K_M | 1,705 | 50.8 | $4.92 |
| RTX 4060 | ~$250 | 8 GB | Q4_K_M | 1,999 | 43.1 | $5.80 |
| AMD BC-250 | ~$200 | 16 GB (shared) | Q4 (Ollama default) | 75 | 25.7 | $7.78 |
| Apple M4 Mac Mini | ~$499 (whole system) | 16 GB (unified) | 4-bit (MLX) | 88 | 21.3 | $23.43 |

*Used prices observed in April 2026; B70 is new retail.

### Key observations

- **B70 generation (54.4 t/s) is in the same class as the RTX 3060 (50.8 t/s).** For a single 9B model, nothing stands out.
- **B70 prefill (1,038 t/s) is 2-4× slower than NVIDIA's CUDA prefill.** This is the SYCL kernel maturity gap - XMX is underused for attention. See our [engine-comparison.md](engine-comparison.md) and the linked XMX flash-attention investigation.
- **RTX 3080 Ti ties the 3090 on decode (113 vs 114 t/s)** - memory-bandwidth-bound, not compute-bound, so the cheaper card wins on value.
- **Raw-speed-per-dollar winner: RTX 3080 Ti at $3.75/(tg t/s).**
- **The B70's $17.44/(tg t/s) is the second-worst ratio in this table** - if 9B is all you're running, NVIDIA wins on value.

**So why buy a B70?** Because 9B isn't all you're running.

---

## Test 2 - Biggest Usable Model (>10 t/s) Per Card

What's the largest model that fits and generates above interactive-threshold (10 t/s)?

| Card | VRAM | Biggest usable model | Quant | Size | tg t/s |
|------|------|---------------------|-------|------|--------|
| **B70 (2 cards)** | **64 GB** | **DeepSeek-R1 70B** | **Q4_K_M** | **39.6 GiB** | **11.3** |
| **B70 (1 card)** | **32 GB** | **Qwen 35B-A3B MoE** | **Q4_K_M** | **20.5 GiB** | **38.9** |
| RTX 3090 | ~20 GB eff. | Gemma 26B-A4B MoE | Q4_K_M | 15.7 GiB | 134.2 |
| RTX 3090 | ~20 GB eff. | Gemma 4 31B dense | Q4_K_M | 17.1 GiB | 36.9 |
| RTX 3090 | ~20 GB eff. | Qwen 35B-A3B MoE | Q4_K_M | 20.5 GiB | OOM (display overhead) |
| RTX 3080 Ti | 12 GB | Qwen 9B | Q8_0 | 8.9 GiB | 80.0 |
| RTX 3060 | 12 GB | Gemma 26B-A4B MoE | IQ2_M (2.7 bpw) | 9.3 GiB | 82.3 |
| RTX 3060 | 12 GB | Qwen 9B | Q8_0 | 8.9 GiB | 34.4 |
| RTX 4060 | 8 GB | Qwen 9B | Q4_K_M | 5.3 GiB | 43.1 |
| BC-250 | 16 GB shared | Qwen 9B | Q4 (Ollama) | ~6.6 GB | 25.7 |
| M4 Mac Mini | 16 GB unified | Qwen 9B | 4-bit (MLX) | 5.2 GB | 21.3 |

### The VRAM story

- **B70 dual (64 GB) is the only single-workstation consumer setup that runs 70B dense at Q4 with interactive speed.** 11.3 t/s for ~$1,900 of GPU.
- **B70 single (32 GB) runs Qwen 35B-A3B at 38.9 t/s.** The RTX 3090 can't fit this model because display overhead eats ~4 GB of its 24 GB.
- **RTX 3090 is the fastest single card in this test at 134 t/s on Gemma 26B-A4B MoE** - but it can't step up to the 35B MoE (OOM). The 3090 owns the middle, the B70 owns the top.
- **RTX 3080 Ti is capped at 12 GB** - it's fast on 9B, but anything bigger is off the menu.
- **RTX 3060 IQ2_M trick:** ultra-compressed Gemma MoE (2.7 bpw) runs at 82 t/s, but quality at 2.7 bpw is questionable.

---

## Test 3 - MoE Advantage by Backend

MoE architectures should be a universal win (tiny active params = fast decode). But the speedup varies dramatically by backend.

Cards that can run both Qwen 9B Dense Q4 and Gemma 26B-A4B MoE Q4:

| Card | Backend | 9B Dense tg t/s | 26B MoE tg t/s | MoE vs Dense |
|------|---------|-----------------|----------------|--------------|
| RTX 3090 | CUDA | 113.7 | **134.2** | **+18% faster, 3× smarter** |
| RTX 3060 | CUDA | 50.8 | 82.3 (IQ2_M) | **+62% faster** (quality TBD at 2.7 bpw) |
| **B70** | **SYCL** | **54.4** | **30.1** | **-45% slower** |

**NVIDIA CUDA MoE is a clear win. Intel SYCL MoE is not - yet.** SYCL kernels don't handle sparse MoE dispatch as efficiently as CUDA. This is a real gap, and it's one of the things SYCL backend contributors (including us) are working on.

**What this means for B70 buyers:** Use Qwen 35B-A3B (38.9 t/s) - it still beats 27B dense (20.6 t/s) on this hardware. But don't expect the dramatic MoE speedup that NVIDIA sees on SYCL until the kernels improve.

---

## Where the B70 Wins Outright

1. **Running 70B dense models.** No 24 GB NVIDIA card can do this. Two 3090s cost roughly the same as two B70s, and while layer-split behavior is similar on both, you're getting 32 GB vs 24 GB per card = more KV-cache headroom, more room for context.
2. **Running 35B MoE models that don't fit a 24 GB card.** Qwen 35B-A3B needs 20.5 GiB; 3090 has ~20 GB effective (with display). B70 has 31 GB free with no display overhead.
3. **Cost per GB of VRAM.** $949 / 32 GB = ~$30/GB on the B70. A used 3090 costs $1,300 / 24 GB ≈ $54/GB. A new one is worse. The B70 is the cheapest path to >24 GB of modern GPU VRAM right now.
4. **Long-context inference.** The 32 GB lets you keep big KV caches without falling back to CPU - context scaling up to 4K shows <1% regression (see [llm-benchmarks.md](llm-benchmarks.md)).
5. **Video gen at higher resolutions.** See [video-generation.md](video-generation.md). Wan 2.2 5B runs up to 832×480, 161-frame clips on a single B70.

## Where the B70 Loses

1. **Prefill-heavy workloads.** RAG over long documents, document summarization, or anything with big prompts. CUDA's flash attention with XMX equivalents is 4-15× faster.
2. **Small-model throughput.** If you only care about running 9B or smaller, a used 3080 Ti is both faster and cheaper.
3. **Ecosystem maturity.** CUDA has every library. SYCL has fewer, and some (Triton, GDN attention in vLLM) are gaps that matter for specific models (see [engine-comparison.md](engine-comparison.md): Qwen 3.5 doesn't currently run on vLLM XPU).

---

## Test 4 - "The Best Coder Each Card Can Run"

Head-to-head: dual B70 (64 GB) vs RTX 3090 (20 GB effective) on the best coding model each can load.

| Card | Model | Size | Active params | Architecture |
|------|-------|------|---------------|--------------|
| **Dual B70** | **Qwen3-Coder-Next 80B-A3B Q4_K_M** | **45.1 GiB** | **~3B** | **80B MoE** |
| RTX 3090 | Qwen3-Coder-30B-A3B Q4_K_M | 17.3 GiB | ~3B | 30B MoE |

Both are MoE with ~3B active params - same compute per token, different depth of knowledge.

### Speed

| Card | Model | pp512 t/s | tg128 t/s |
|------|-------|-----------|-----------|
| Dual B70 (SYCL) | 80B-A3B Q4_K_M | 298.4 | **42.4** |
| RTX 3090 (CUDA) | 30B-A3B Q4_K_M | 3,217 | **184.6** |

The 3090 is 4.4× faster on generation - but it's running a 30B model. The B70 pair runs the 80B that the 3090 can't fit. Different tradeoff.

### Quality - LRU Cache Implementation

Prompt: *"Implement a thread-safe LRU cache in Python with O(1) get and put operations. Include type hints and docstrings."*

| Aspect | 3090 (30B, 185 t/s) | Dual B70 (80B, 42 t/s) |
|--------|---------------------|------------------------|
| Approach | `OrderedDict` wrapper | Doubly-linked list from scratch |
| Thread safety | `RLock` | `Lock` |
| API completeness | get/put/remove/clear/size/keys | get/put + `__len__`/`__contains__`/`__repr__` |
| Type hints | Generic `K, V` | Generic `K, V` |
| CS rigor | Practical/Pythonic | Textbook O(1) implementation |
| Error handling | Validation + docs | Validation + docs |

Both correct, both production-quality. The 80B wrote a more rigorous, lower-level implementation (doubly-linked list with dummy head/tail). The 30B used `OrderedDict` - more Pythonic. The 80B shows deeper algorithmic understanding; the 30B is perfectly fine for most tasks.

**Tradeoff: 80B is 4.4× slower but produces more sophisticated code.** The 80B's advantage shows on problems that benefit from depth of reasoning or novel algorithms.

---

## Summary - Which Card For Which Use Case?

| You want... | Buy... |
|-------------|--------|
| Fastest 9B-scale inference at lowest cost | RTX 3080 Ti (used) |
| 24-32 GB model support at NVIDIA ecosystem depth | RTX 3090 (used) |
| **Run 70B dense models at interactive speed** | **2× Intel Arc Pro B70** |
| Run 35-80B MoE models with room to spare | Intel Arc Pro B70 (single or dual) |
| Cheapest "real" AI GPU | RTX 3060 or BC-250 if you can tolerate the rough edges |
| Apple ecosystem + quiet box | M4 Mac Mini (expect 20-50 t/s ceiling) |

The B70 is not the cheapest tokens-per-second on small models. It **is** the cheapest path to running models that don't fit on consumer NVIDIA cards.

---

## Related

- [llm-benchmarks.md](llm-benchmarks.md) - B70 single-card SYCL numbers in full detail
- [multi-gpu.md](multi-gpu.md) - dual B70 behavior
- [engine-comparison.md](engine-comparison.md) - llama.cpp SYCL vs vLLM XPU
- [llmresults.com](https://llmresults.com) - interactive cross-card UI with the same data
