# llama.cpp SYCL vs vLLM XPU on Intel Arc Pro B70

Both engines run on Intel's oneAPI / Level Zero stack, but they take very different paths. This page is the head-to-head.

**Short version:**
- **Model coverage:** llama.cpp wins. vLLM XPU doesn't support Qwen 3.5 (GDN attention needs Triton/CUDA kernels not available on XPU).
- **Prefill (prompt processing):** vLLM wins big - 2.4-15× faster, thanks to XMX flash attention kernels in `vllm-xpu-kernels`.
- **Decode (token generation):** ~tie. Both are memory-bandwidth-bound and converge to similar t/s with our upstream BF16 fix applied.
- **Single-card memory:** llama.cpp wins - quantized formats fit more in 32 GB than vLLM's FP16.
- **Multi-GPU dense models:** vLLM wins with tensor parallelism; llama.cpp's layer split doesn't speed up decode (see [multi-gpu.md](multi-gpu.md)).

---

## Test Environment

- **Hardware:** 2× Intel Arc Pro B70, Ryzen 5 9600X, 64 GB DDR5
- **llama.cpp:** build `b8724` (commit `b54cb2e3d`), SYCL backend, upstream (no local patches), F16 mode
- **vLLM:** `intel/vllm:latest` (v0.1.dev14456), `vllm-xpu-kernels`, FP16
- **Date:** 2026-04-09

---

## Finding 1 - Model Coverage Matters

**Qwen 3.5 doesn't run on vLLM XPU.** Qwen 3.5 uses Gated Delta Net (GDN) attention, which requires Triton/CUDA kernels not available on XPU. The model loads, then crashes at first inference:

```
File "fla/ops/chunk.py", line 207, in chunk_gated_delta_rule
RuntimeError: PyTorch was compiled without CUDA support
```

Qwen 3.5 is one of the most active model families in early 2026, so this is a real gap in vLLM's XPU support.

**llama.cpp SYCL runs Qwen 3.5 fine:** 54.5 t/s tg on Q4_K_M, 784 t/s pp128.

---

## Finding 2 - vLLM Prefill Crushes (XMX Advantage)

On Qwen 2.5-1.5B single-GPU, BF16 / FP16 (same precision):

| Test | llama.cpp BF16 | vLLM FP16 | Ratio |
|------|---------------|-----------|-------|
| pp128 | 2,579 t/s | 6,289 t/s | 2.4× vLLM |
| pp512 | 4,567 t/s | 20,555 t/s | 4.5× vLLM |
| pp2048 | 4,426 t/s | 67,851 t/s | **15.3× vLLM** |
| tg128 | 29.6 t/s* | 141.9 t/s | 4.8× vLLM |

*Upstream BF16 tg at 29.6 t/s is a known DMMV bug. Our PR [#21580](https://github.com/ggerganov/llama.cpp/pull/21580) brings this to ~124 t/s (gap narrows to 1.1×).

**vLLM's 15× prefill lead at 2K context is real, and it scales with sequence length** - the gap widens as prompts get longer. Two causes:
1. **XMX / DPAS flash attention** in `vllm-xpu-kernels` (CUTLASS-style Intel XMX kernels)
2. **Varlen batching** in vLLM vs simpler ubatch slicing in llama.cpp

llama.cpp SYCL's attention path is scalar FP16, no XMX. Closing this is one of the biggest open performance projects for the SYCL backend - see our [SYCL Flash Attention XMX investigation notes](https://github.com/ggerganov/llama.cpp/issues) linked from [upstream-contributions.md](upstream-contributions.md).

---

## Finding 3 - Decode Is Comparable (With Our Fix)

On Qwen 2.5-1.5B, BF16 vs FP16 single-GPU:

| Config | tg128 t/s |
|--------|-----------|
| llama.cpp BF16 (upstream) | 29.6 |
| llama.cpp BF16 (our PR #21580 applied) | ~124 |
| vLLM FP16 | 141.9 |

With our BF16 DMMV fix, llama.cpp decode is within 12% of vLLM. Both are memory-bandwidth-bound on decode; quantization gives llama.cpp an edge when file size matters (see below).

---

## Finding 4 - Quantized Wins Decode

Qwen 2.5-1.5B, single GPU, best each engine offers:

| Test | llama.cpp Q4_K_M | llama.cpp Q8_0 | vLLM FP16 |
|------|-----------------|----------------|-----------|
| pp128 | 3,474 | 3,287 | 6,289 |
| pp512 | 5,145 | 5,052 | 20,555 |
| pp2048 | 4,751 | 4,703 | 67,851 |
| **tg128** | **198** | 181 | 141.9 |

**llama.cpp Q4_K_M decode (198 t/s) beats vLLM FP16 (142 t/s)** - smaller weights = more memory bandwidth per token. If decode throughput is the goal and the model quantizes well, llama.cpp wins.

vLLM's `vllm-xpu-kernels` does support INT4/INT8/FP8 modes too - we tested FP16 for the cleanest precision comparison. With INT4/FP8 weights on vLLM, the decode gap should close or reverse.

---

## Finding 5 - Single-Card 14B: vLLM OOMs, llama.cpp Fine

Qwen 2.5-14B single B70 (32 GB):

| Test | llama.cpp Q4_K_M | llama.cpp Q8_0 | vLLM FP16 |
|------|-----------------|----------------|-----------|
| pp128 | 352 | 340 | **OOM** |
| pp512 | 523 | 513 | **OOM** |
| pp2048 | 498 | 491 | **OOM** |
| tg128 | 36.8 | 28.5 | **OOM** |

vLLM FP16 can't fit Qwen 2.5-14B on a single 32 GB B70. Model weights are 27.6 GiB, leaving -1.3 GiB for KV cache. llama.cpp's 8.4 GB Q4 and 15 GB Q8 fit fine. Quantization is the single biggest reason to use llama.cpp on 32 GB cards.

---

## Finding 6 - Dual-GPU Qwen 2.5-14B: The Headline

llama.cpp layer-split (Q4_K_M) vs vLLM tensor parallelism (FP16, TP=2):

| Test | llama.cpp Q4 (layer-split) | vLLM FP16 (TP=2) | Ratio |
|------|---------------------------|-------------------|-------|
| pp128 | 342 t/s | 2,070 t/s | 6.1× vLLM |
| pp512 | 521 t/s | 7,104 t/s | 13.6× vLLM |
| pp2048 | 766 t/s | 11,212 t/s | 14.6× vLLM |
| **tg128** | **37.8 t/s** | **36.1 t/s** | **~tie** |

### Apples-to-apples: BF16 vs FP16, same precision

This removes the "Q4 vs FP16" objection. Same model, same precision, same hardware:

| Test | llama.cpp BF16 (layer-split) | vLLM FP16 (TP=2) | Ratio |
|------|------------------------------|-------------------|-------|
| pp128 | 268 t/s | 2,069 t/s | 7.7× vLLM |
| pp512 | 459 t/s | 7,080 t/s | **15.4× vLLM** |
| pp2048 | 692 t/s | 11,385 t/s | **16.5× vLLM** |
| tg128 | 3.47 t/s* | 35.9 t/s | 10.4× vLLM |

*llama.cpp BF16 tg at 3.47 t/s is the upstream DMMV bug. PR #21580 brings this to ~35 t/s - parity.

**The 7-16× prefill gap is entirely due to XMX/DPAS flash attention + varlen batching in vLLM vs scalar FP16 attention + ubatch slicing in llama.cpp.** It's not an Intel hardware limit or a driver issue - it's kernel maturity on the llama.cpp SYCL side.

---

## Strengths & Weaknesses

| Criterion | llama.cpp SYCL | vLLM XPU |
|-----------|---------------|----------|
| Model compatibility | Qwen 2.5, Qwen 3.5, Gemma, Llama - all work | Qwen 3.5 FAILS (GDN), Qwen 2.5 works |
| Single-GPU memory | Q4 fits 14B in 8.4 GB | FP16 needs 28 GB - OOM on 14B / 32 GB |
| Prefill (14B dual) | 342-766 t/s | 2,070-11,212 t/s |
| Decode (14B dual, same precision) | 35 t/s (with PR #21580) | 36 t/s |
| Quantization formats | GGUF: Q2-Q8, BF16, IQ, K-quants | FP16 tested; INT4/INT8/FP8 also supported |
| Multi-GPU approach | Layer split (sequential) | Tensor parallelism (true parallel) |
| Setup complexity | Build from source, bare metal | Docker with `--privileged` + `--device=/dev/dri` |
| Best for | Single-user interactive, quantized models, wide model support | Multi-user serving, long-prompt RAG, FP16 workloads |

---

## When to Use Which

**Use llama.cpp SYCL when:**
- You want the widest model coverage, including Qwen 3.5
- Your model only fits if you quantize (most 14B+ on a single B70)
- You're doing interactive single-user chat or small-scale inference
- You want to build on bare metal, no Docker

**Use vLLM XPU when:**
- You're doing RAG or summarization over long documents (prefill-heavy)
- You're serving multi-user traffic and want tensor parallelism
- Your models are well-supported (Qwen 2.5, Llama, Mistral) and you don't mind the model-coverage gap
- You can tolerate Docker + `--privileged`

Both have a place. The right engine depends on workload, not which "wins" in isolation.

---

## Related

- [multi-gpu.md](multi-gpu.md) - why llama.cpp layer-split doesn't double decode
- [llm-benchmarks.md](llm-benchmarks.md) - single-card llama.cpp sweep
- [upstream-contributions.md](upstream-contributions.md) - PRs #21580 (BF16 DMMV) and #21700 (broader SYCL optimization)
