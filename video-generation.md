# Video Generation on Intel Arc Pro B70

All tests on a single B70 (GPU.1, PCI 08:00.0, 32 GB GDDR6 ECC). Video gen doesn't benefit from multi-GPU with Diffusers or OpenVINO, so these are all single-card numbers.

**Uniform prompt (for comparability):** *"A drone shot slowly flying over a vast mountain landscape at golden hour, with snow-capped peaks and a winding river below, cinematic quality."*

---

## Model Summary

| Model | Pipeline | Max practical resolution | Speed @ 832×480 25f | Quality tier |
|-------|----------|--------------------------|---------------------|--------------|
| **LTX-Video (OpenVINO FP16)** | OpenVINO | **1280×704** | N/A (different aspect) | Good |
| Wan 2.2 5B (XPU) | Diffusers | 832×480 | 38.4s | Higher |
| Wan 2.1 1.3B (XPU) | Diffusers | 832×480 | 64.6s | Lower |
| Wan 2.2 A14B MoE (XPU) | Diffusers + CPU offload | 480×320 only | - | Not practical |

The short story: **LTX on OpenVINO is the fast path for iteration, Wan 2.2 5B is the quality path for finals.** The A14B MoE can technically load with CPU offload but takes 5 minutes for a 1-second low-res clip - not a usable workflow.

---

## LTX-Video (OpenVINO FP16)

Intel-optimized pipeline. Exported from Lightricks/LTX-Video to OpenVINO IR FP16, run via `openvino_genai.Text2VideoPipeline`.

### Full resolution × duration sweep

| Resolution | Frames | Duration @25fps | Steps | Gen time | Gen FPS |
|-----------|--------|-----------------|-------|----------|---------|
| 512×320 | 25 | 1.0s | 20 | **2.2s** | 11.5 |
| 512×320 | 49 | 2.0s | 20 | 3.4s | 14.4 |
| 512×320 | 97 | 3.9s | 20 | 6.2s | 15.7 |
| 512×320 | 161 | 6.4s | 20 | **10.8s** | 15.0 |
| 512×320 | 257 | 10.3s (via 704×480) | 20 | - | - |
| 704×480 | 25 | 1.0s | 25 | 4.9s | 5.1 |
| 704×480 | 49 | 2.0s | 25 | 8.7s | 5.6 |
| 704×480 | 97 | 3.9s | 25 | 18.1s | 5.4 |
| 704×480 | 257 | **10.3s** | 25 | 65.2s | 3.9 |
| 832×576 | 25 | 1.0s | 25 | 6.7s | 3.7 |
| 832×576 | 49 | 2.0s | 25 | 13.0s | 3.8 |
| 1024×576 | 25 | 1.0s | 25 | 8.6s | 2.9 |
| 1024×576 | 49 | 2.0s | 25 | 16.5s | 3.0 |
| 1280×704 | 25 | 1.0s | 25 | 13.9s | 1.8 |
| 1280×704 | 49 | 2.0s | 25 | 29.0s | 1.7 |
| 1280×704 | 97 | 3.9s | 25 | **71.8s** | 1.4 |
| 1280×704 | 161 | 6.4s | 25 | 147.3s | 1.1 |
| 1280×704 | 257 | 10.3s | 25 | **OOM** | - |

### Practical sweet spots

- **Fast previews:** 512×320, 49 frames - 3.4s for a 2-second clip. Near real-time iteration.
- **Good quality + speed:** 704×480, 49 frames - 8.7s for a 2-second clip.
- **High quality:** 1280×704, 49 frames - 29s for a 2-second clip.
- **Long video at low res:** 512×320, 161 frames - 10.8s for a 6.4-second clip.
- **Long video at high res:** 704×480, 257 frames - 65s for a 10.3-second clip.
- **Maximum near-720p:** 1280×704, 161 frames - 147s for a 6.4-second clip; 257 frames OOMs.

### Notes

- No OOM at any resolution tested below the 1280×704/257f combo. FP16 LTX fits comfortably with room for frame buffers.
- Width and height must be divisible by 32.
- Generation FPS scales roughly inversely with pixel count (expected - compute ~ pixels/frame).

---

## Wan 2.2 5B Dense (PyTorch XPU)

Wan-AI/Wan2.2-TI2V-5B-Diffusers, run via `diffusers.WanPipeline` with PyTorch XPU backend on `xpu:1`.

| Config | Resolution | Frames | Steps | Gen time | Gen FPS |
|--------|-----------|--------|-------|----------|---------|
| wan22_5B | 480×320 | 25 | 20 | **16.8s** | 1.49 |
| wan22_5B | 832×480 | 25 | 30 | **38.4s** | 0.65 |
| wan22_5B | 832×480 | 49 | 30 | **70.9s** | 0.69 |
| wan22_5B | 832×480 | 97 | 30 | 150.9s | 0.64 |
| wan22_5B | 832×480 | 161 | 30 | **270.0s** | 0.60 |
| wan22_5B | 480×832 (portrait) | 49 | 30 | 71.3s | 0.69 |
| wan22_5B | 1280×720 | 25 | - | - | **OOM** |

### Practical limits

- **Max resolution:** 832×480 (landscape or portrait - 480×832 works, same speed)
- **Max duration at 832×480:** 161 frames (6.7 s @ 24 fps) in 4.5 minutes
- **720p:** OOM. The model + activations exceed 32 GB at that resolution.

**Wan 2.2 5B is 1.7× faster than Wan 2.1 1.3B** at the same resolution. Counterintuitive, but the new Wan 2.2 VAE (4×16×16 compression = 64× total) shrinks the latent more than the larger transformer costs.

---

## Wan 2.1 1.3B (PyTorch XPU)

Wan-AI/Wan2.1-T2V-1.3B-Diffusers, the smallest Wan variant. Mostly tested for comparison.

| Config | Resolution | Frames | Steps | Gen time | Gen FPS |
|--------|-----------|--------|-------|----------|---------|
| wan21_1.3B | 480×320 | 25 | 30 | **23.7s** | 1.06 |
| wan21_1.3B | 832×480 | 25 | 30 | **64.6s** | 0.39 |
| wan21_1.3B | 832×480 | 49 | 30 | **142.7s** | 0.34 |

Slower than Wan 2.2 5B at the same resolution, worse VAE, lower quality. Skip unless you specifically need Wan 2.1 output.

---

## Wan 2.2 A14B MoE - Not Practical on Single Card

Wan-AI/Wan2.2-T2V-A14B-Diffusers, 14B total with 2 experts (MoE).

**Loading at FP16:** **OOM.** Transformer alone is ~28 GB, plus VAE and text encoder = >32 GB total.

**With `enable_sequential_cpu_offload()`:**

| Config | Resolution | Frames | Steps | Gen time | Gen FPS |
|--------|-----------|--------|-------|----------|---------|
| A14B (CPU offload) | 480×320 | 25 | 20 | **321.6s** | 0.08 |

**5.3 minutes for a 1-second, 480×320 clip.** The CPU-offload path moves model chunks between RAM and VRAM on every step, destroying throughput.

**Why MoE doesn't help VRAM in diffusion models.** Unlike LLMs, diffusion transformer MoE layers load **all** expert weights even if only a subset activates per step - the routing happens inside the forward pass on already-loaded weights. Diffusers also doesn't support multi-GPU model splitting the way llama.cpp's layer-split does for LLMs.

**Conclusion:** Wan 2.2 5B is the practical ceiling for our single-card setup. For higher quality, tune prompts and steps on the 5B rather than stepping up to the 14B.

---

## LTX vs Wan - Head-to-Head

Same-ish resolution and frame counts, single B70:

| Resolution / Frames | LTX (OpenVINO FP16) | Wan 2.2 5B (XPU) | Wan 2.1 1.3B (XPU) | LTX speedup |
|---------------------|---------------------|-------------------|---------------------|-------------|
| ~480×320 / 25f | 2.2s | 16.8s | 23.7s | **7.6× vs Wan 2.2, 10.8× vs Wan 2.1** |
| ~832×480 / 25f | 6.7s (832×576) | 38.4s | 64.6s | **5.7× vs Wan 2.2, 9.6× vs Wan 2.1** |
| ~832×480 / 49f | 13.0s (832×576) | 70.9s | 142.7s | **5.5× vs Wan 2.2, 11× vs Wan 2.1** |

**LTX is 5-11× faster than Wan on this hardware.** Two things going on:

1. **OpenVINO is Intel-optimized** - the IR is compiled specifically for Xe2. PyTorch XPU runs PyTorch graphs through generic SYCL with no model-specific optimization.
2. **LTX is a lighter model** - designed for fast inference to begin with.

**Quality tradeoff:** Wan is generally considered higher quality for realistic video. LTX is competitive and much faster. For quick iteration LTX wins; for final renders the Wan 2.2 5B is worth the extra time if output quality matters more than iteration speed.

---

## GPU Assignment

We use GPU.1 (second B70, PCI 08:00.0) for video gen and reserve GPU.0 for LLM work in parallel sessions. To pin a job to GPU.1:

**PyTorch XPU:**

```python
device = "xpu:1"
model = pipe.to(device)
```

**OpenVINO:**

```python
pipe = openvino_genai.Text2VideoPipeline(model_dir, device="GPU.1")
```

---

## TODO

- [ ] Convert Wan 2.2 5B to OpenVINO IR and re-benchmark - expect 3-5× speedup if the Intel optimization gap holds
- [ ] INT8/INT4 weight compression for Wan to enable 720p on single card
- [ ] Compare visual quality on a shared prompt collection (LTX vs Wan 2.2 5B)
- [ ] Test Wan 2.2 A14B across both B70s if multi-GPU diffusion ever becomes viable

## Related

- [hardware.md](hardware.md) - GPU specs
- [methodology.md](methodology.md) - how video-gen tests were set up
