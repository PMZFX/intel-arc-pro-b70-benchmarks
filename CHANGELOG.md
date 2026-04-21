# Changelog

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
