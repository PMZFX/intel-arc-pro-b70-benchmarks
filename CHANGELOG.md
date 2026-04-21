# Changelog

## v0.2.2 - Phase 2 context scaling + KV quant (2026-04-21)

22 new data points sweeping context length 4K to 64K and KV-cache quantization
(fp16, Q8_0, Q4_0) on three models. Same commit pin (`ec6f7a6a5c`), same build,
all per-run power telemetry. Runs on both cards in parallel (card 0 ran Qwen
27B + Qwen 3.6 via Race Bench, card 1 ran Gemma 31B via a direct Python
runner), validated to produce identical numbers to single-card.

### Prefill scales with context, decode stays flat

tg128 holds essentially constant across 4K-64K with fp16 KV on every model
tested:

| Model | pp512 (Phase 1) | pp4K | pp8K | pp16K | pp32K | pp64K | tg (all) |
|-------|-----------------|------|------|-------|-------|-------|----------|
| Qwen 3.5-27B Q4_K_M | 718 | 629 | 560 | 445 | 310 | 195 | 20.4 |
| Qwen 3.6-35B-A3B UD-Q4_K_M | 615 | 579 | 559 | 527 | 485 | 411 | 54.5 |
| Gemma 4 31B Q4_K_M | 601 | 417 | 349 | 268 | 185 | (timeout) | 21.6 |

MoE's prefill degrades the least with context (3.6-35B: 579 to 411 at 4K to
64K is only -29%). Dense 27B falls faster (-69%). Dense 31B Gemma falls
fastest (-56% at 32K alone).

### KV quantization tg penalty is model-dependent

| Model | fp16 tg | Q8_0 KV tg | Δ | Q4_0 KV tg | Δ |
|-------|---------|------------|---|------------|---|
| Qwen 3.5-27B Q4_K_M (32K) | 20.36 | 19.97 | -1.9% | (pending) | - |
| Qwen 3.6-35B-A3B (32K) | 54.47 | 52.38 | -3.8% | 52.31 | -4.0% |
| Gemma 4 31B Q4_K_M (32K) | 21.62 | 20.00 | -7.5% | 19.77 | -8.6% |

Prefill is unaffected by KV quant choice. "Q4_0 KV is free" is a reasonable
default for Qwen-family MoE, but Gemma 31B takes a noticeable hit.

### 131K context notes

Both 131K runs (Qwen 27B Q4_0 KV, Qwen 3.6 Q4_0 KV) timed out at our 30-minute
benchmark ceiling. The budget, not the hardware, is the limit: extrapolated
pp at 131K is ~100 t/s, which at `-r 3` is ~60 min of pure compute. v0.3 will
re-run 131K variants with `-r 1` to fit the budget.

### Parallel card execution validated

Running independent benchmarks on card 0 and card 1 simultaneously produces
numbers within 1% of single-card-only baselines (Qwen 3.5-9B Q4_K_M card-1-alone
was 2,302/60.22; during parallel card-0-busy load it hit 2,303/60.62). No
cross-card contention on this workstation (CPU-direct PCIe 5.0 x8, 6 CPU cores,
2x 32 GB VRAM).

## v0.2.1 - Phase 3 model family coverage (2026-04-21)

Added 5 more models in the same session as v0.2, same commit pin and flags:

| Model | Quant | GPUs | pp | tg | avg W | t/J |
|-------|-------|------|-----|-----|-------|-----|
| Llama 3.1-8B Instruct | Q4_K_M | 1 | 2,452 | 82.6 | 37 | 2.26 |
| Llama 3.3-70B Instruct | Q4_K_M | 2 | 338 | 11.5 | 186 | 0.06 |
| Mistral Small 3.2-24B | Q4_K_M | 1 | 994 | 30.1 | 167 | 0.18 |
| Devstral Small 2-24B | Q4_K_M | 1 | 987 | 30.0 | 165 | 0.18 |
| Phi-4 14B | Q4_K_M | 1 | 1,424 | 43.7 | 40 | 1.08 |

Notes: Llama 3.1-8B and Phi-4 were loaded from NAS over CIFS, so their avg
wattage includes load-time dilution. Peak wattage (247 / 244 W) is the real
under-load number. Llama 3.3-70B mirrors DeepSeek-R1 70B almost exactly
(both dual-card layer-split dense 70B, hit the same bandwidth ceiling).

## v0.2 - Phase 1 baseline refresh (2026-04-21)

Full single-card and dual-card re-baseline on a pinned llama.cpp commit, with
power telemetry captured on every run. Replaces the earlier scattered numbers
that had inconsistent build flags.

### What changed in the data

- All runs from a single llama.cpp commit: **`ec6f7a6a5c`** (describe: `b8840-12-gec6f7a6a5-dirty`). Tree includes a local Q2_K DMMV WIP modification in `ggml/src/ggml-sycl/dmmv.cpp`, preserved rather than stashed; the `-dirty` tag reflects that.
- Rebuilt with explicit `-DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG"` to strip asserts. Prior builds had empty `CMAKE_CXX_FLAGS_RELEASE`, so the `NDEBUG` macro never made it into the C++ units and every run printed `warning: asserts enabled, performance may be affected`. This systematically suppressed prefill ~50% across the board.
- All runs include per-GPU peak/avg watts, total energy (J), and derived tokens-per-joule, sampled at 500 ms via the `xe` driver's hwmon energy counters (`/sys/class/drm/cardN/device/hwmon/*/energy*_input`).
- Every run stamps the llama.cpp commit hash so results are commit-pinned.

### New data (had nothing before)

- **Qwen 3.6-35B-A3B** (both UD-Q4_K_M single-card and Q8_0 dual-card). First public B70 numbers on this model family.
- Power draw + tokens-per-joule efficiency for every model.

### Notable changes vs prior numbers

| Model | Quant | Old pp / tg | **New pp / tg** | Δ |
|-------|-------|-------------|-----------------|---|
| Qwen 2.5-1.5B | Q4_K_M | 5,313 / 229 | **8,048 / 216** | **pp +51%**, tg -6% |
| Qwen 3.5-9B | Q4_K_M | 1,038 / 54.4 | **2,302 / 60.2** | **pp +122%**, tg +11% |
| Qwen 3.5-27B | Q4_K_M | 302 / 20.6 | **718 / 20.4** | **pp +138%**, tg stable |
| Qwen 3.5-27B | Q8_0 | 295 / 4.88 | **776 / 15.3** | **pp +163%**, **tg +214%** (PR #21527) |
| Qwen 3.5-27B | Q6_K | 304 / 13.8 | **785 / 15.1** | pp +158%, tg +9% |
| Gemma 4 26B-A4B | Q4_K_M | 943 / 30.1 | **1,129 / 52.6** | pp +20%, **tg +75%** |
| Gemma 4 31B | Q4_K_M | 255 / 22.6 | **601 / 21.7** | **pp +136%**, tg stable |
| Gemma 4 31B Q8 dual | Q8_0 | 252 / 4.1 | **654 / 14.1** | **pp +160%**, **tg +244%** (PR #21527 / #21638) |
| Qwen 3.5-35B-A3B | Q4_K_M | 573 / 38.9 | **618 / 54.5** | pp +8%, **tg +40%** |
| Qwen 3.5-35B-A3B dual | Q8_0 | 463 / 28.7 | 458 / 36.5 | pp stable, **tg +27%** |
| DeepSeek-R1 70B dual | Q4_K_M | 120 / 11.26 | **336 / 11.5** | **pp +180%**, tg stable |
| Qwen3-Coder-Next 80B dual | Q4_K_M | 298 / 42.4 | 305 / 43.4 | pp +2%, tg +2% (already on a good build) |

### How to reproduce

```bash
# Rebuild with asserts stripped:
bash ~/AI/llm-race-bench/sweeps/rebuild-llama-cpp-pinned.sh ec6f7a6a5

# Run the full Phase 1 sweep through Race Bench:
curl -X POST http://localhost:7800/api/sweep/start \
  -H "Content-Type: application/json" \
  -d @~/AI/llm-race-bench/sweeps/phase1-b70-core.json
```

### What's still old

- Files named `b70-*.json` in `data/llm/` are the original (pre-fix) dataset, kept for historical comparison. The `intel-arc-pro-b70-*.json` files are the v0.2 data.
- `multi-gpu.md`, `cross-card-comparison.md`, `engine-comparison.md`, and `video-generation.md` narrative tables still reference v0.1 numbers in a few places. They'll be updated in v0.3.

## v0.1 - Initial repo (2026-04-21)

Initial scaffolding with writeups, inventory, and scattered early benchmarks.
