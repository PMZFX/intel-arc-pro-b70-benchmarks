# Upstream llama.cpp Contributions - Intel Arc B70 / SYCL

While benchmarking the B70 we hit several real upstream bugs. Rather than work around them, we filed PRs. This is the running list.

All PRs are against [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp). Status is current as of 2026-04-21.

---

## Summary

| PR | Title | Impact on B70 | Status |
|----|-------|---------------|--------|
| [#21527](https://github.com/ggerganov/llama.cpp/pull/21527) | Q8_0 SYCL reorder | 3.1× tg on Q8_0 dense models (4.88 → 15.24 t/s) | ✅ Merged |
| [#21580](https://github.com/ggerganov/llama.cpp/pull/21580) | BF16 DMMV kernel | 4.2× tg on BF16 (29.7 → 124.0 t/s) | 📬 Submitted |
| [#21597](https://github.com/ggerganov/llama.cpp/pull/21597) | Multi-GPU RAM fix (Level Zero allocator) | System RAM: 60 GiB → 6.7 GiB during dual-GPU load | 🔍 Under review |
| [#21638](https://github.com/ggerganov/llama.cpp/pull/21638) | Q8_0 GEMM dequantize fix | Fixes post-merge regression from #21527 | ✅ Merged |
| [#21700](https://github.com/ggerganov/llama.cpp/pull/21700) | SYCL optimization bundle (reorder hang fix) | Misc decode improvements | 📬 Awaiting merge |

---

## PR #21527 - Q8_0 SYCL Reorder (Merged)

**Problem.** Dense-model Q8_0 inference on Xe2/Battlemage ran at ~4-5 t/s generation. Q4_K_M on the same model ran at ~20 t/s - 4× faster despite being less compressed. The Q8_0 kernel was leaving memory bandwidth on the table.

**Investigation.** We traced the SYCL dispatch and found Q8_0 was routed to the DMMV path instead of MMVQ, and the weights weren't being reordered into the vectorizable layout MMVQ needs.

**Fix.** Added reorder support for Q8_0 so it takes the MMVQ path like Q4_K_M does.

**Result.** Qwen 27B Q8_0 tg128 went from 4.88 to 15.24 t/s (3.1× speedup). Prompt processing unchanged (it was already OK).

**Follow-up.** A GEMM dequantize bug in the new path was caught post-merge - fixed in PR #21638 below.

---

## PR #21580 - BF16 DMMV Kernel (Submitted)

**Problem.** BF16 token generation on Qwen 2.5-1.5B ran at 29.6 t/s. FP16 on the same model ran 140+ t/s. BF16 was missing its DMMV kernel entirely and falling back to a slow path.

**Fix.** Added the missing BF16 DMMV kernel for the SYCL backend, matching the FP16 path.

**Result.** BF16 tg went from 29.7 to 124.0 t/s (4.2× speedup). Brings BF16 decode within 12% of vLLM XPU FP16 on the same hardware (see [engine-comparison.md](engine-comparison.md) Finding 3).

**Status.** Submitted, awaiting final review.

---

## PR #21597 - Multi-GPU RAM Fix (Under Review)

**Problem.** Dual-B70 SYCL inference consumed **~60 GiB of system RAM** during model load, even though the model fits entirely in VRAM. On our 64 GB box this was enough to OOM-crash concurrent workloads.

**Root cause.** The `sycl::malloc_device` allocator on the `xe` kernel driver routes through the TTM (Translation Table Manager) DMA-buf path, which mirrors every device-local allocation into system RAM as a backing store - even when the buffer is resident on VRAM.

**Fix.** Swap `sycl::malloc_device` for Level Zero's `zeMemAllocDevice`, which allocates device-local memory without the TTM mirror.

**Result.** System RAM use during dual-GPU model load: 60 GiB → 6.7 GiB (roughly a 10× reduction). Inference speed unchanged; this is purely a memory-consumption fix.

**Verification.** We re-verified this is still needed on 2026-04-19 using fdinfo's `drm-total-gtt` counter (not VmRSS, which doesn't see GTT allocations). The "runtime fixed it" speculation from earlier this month was wrong; the fix is still needed.

**Status.** Under review. Needs a second approver + "merge ready" label.

---

## PR #21638 - Q8_0 GEMM Dequantize Fix (Merged)

**Problem.** After PR #21527 merged, a subtle bug surfaced in the Q8_0 GEMM dequantize path on specific model architectures. Testing only caught it on HunyuanMT and a few other models with non-aligned vocab sizes (e.g. HY-MT's 120818).

**Lesson.** Post-merge regressions like this are why our [PR checklist](https://github.com/ggerganov/llama.cpp) now mandates full quant + architecture coverage before pushing a change. Non-aligned vocabs (anything that isn't a multiple of 32) are the archetypal gotcha.

**Fix.** Corrected the dequantize bounds so non-aligned vocabs don't read past buffer end.

**Status.** ✅ Merged.

---

## PR #21700 - SYCL Optimization Bundle (Awaiting Merge)

**Scope.** Grab-bag of smaller SYCL decode improvements:
- K-quant DMMV subgroup-16 path
- Reorder hang fix for edge-case batch sizes
- Miscellaneous cleanup

Rebased 2026-04-16. XMX flash attention work originally planned to go into this PR is shelved for now - see the shelved-work section below.

**Status.** Rebased, awaiting merge.

---

## Open Bugs We've Discovered But Not Yet Filed

### SYCL `--split-mode row` segfault

- **First seen:** 2026-04-17
- **Reproducer:**
  ```
  ./llama-cli -m Qwen2.5-14B-Instruct-Q8_0.gguf -ngl 99 --split-mode row -n 48 -p "..."
  ONEAPI_DEVICE_SELECTOR=level_zero:0,1
  ```
- **Symptom:** Segfault (exit 139) during model load / early generation
- **Not crash with:** `--split-mode layer` (works fine on both models)
- **Also crashes on:** HY-MT-1.8B Q8_0
- **Confirmed pre-existing:** reproduced on pristine master, so not caused by any of our PRs
- **Scope:** SYCL on Xe2/Battlemage; cross-backend parity (CUDA, Vulkan) not yet tested
- **Action:** will file upstream before pushing further multi-GPU changes

---

## Shelved Work

### XMX Flash Attention via ESIMD

The prefill gap against vLLM (see [engine-comparison.md](engine-comparison.md) Finding 2) is largely down to missing XMX flash attention in llama.cpp SYCL. We explored closing this gap directly:

- **SYCL `joint_matrix`:** broken on BMG. Generates incorrect code in our testing.
- **SYCL `ext::intel::esimd` xmx::dpas:** works, produced a prototype, but hit an Intel Graphics Compiler (IGC) codegen bug. Filed [intel/llvm#21741](https://github.com/intel/llvm/issues/21741).
- **Status:** shelved pending Intel's response on the IGC bug. All isolation strategies exhausted on our end.

### TurboQuant SYCL port

Investigated porting TurboQuant-style k-quant kernels to SYCL. Drafted plan, identified significant scope; parked while other PRs are in review.

---

## How We Avoid Regressions Now

Three-pass audit before pushing any upstream PR:

1. **Scope pass** - does this PR do more than it claims?
2. **Correctness pass** - full quant matrix (Q4_0 through Q8_0, K-quants, BF16, FP16), full architecture matrix including non-aligned vocabs
3. **Merge-readiness pass** - adversarial reviewer-skeptic read

The checklist lives in `~/AI/llm/PR-CHECKLIST.md` and is run in full before every push. It exists because we shipped two post-merge bugs (the Q8_0 GEMM issue caught by PR #21638, and a Q4_K DMMV reorder crash surfaced by another contributor). Full quant + architecture coverage is the fix for both.

---

## Related

- [engine-comparison.md](engine-comparison.md) - why the XMX work matters for prefill
- [multi-gpu.md](multi-gpu.md) - why the PR #21597 Level Zero allocator fix matters
- [llm-benchmarks.md](llm-benchmarks.md) - the Q8_0 story that motivated PR #21527
