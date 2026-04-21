# Methodology

How we ran these benchmarks - so you can reproduce them or spot where we diverge from what you're testing.

## llama.cpp Benchmarks

All LLM numbers are from `llama-bench`, the standard benchmarking tool shipped with llama.cpp. Unless noted:

- **Tool:** `llama-bench` from llama.cpp built from source
- **Backend:** SYCL via Intel Level Zero
- **Build flags:** `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON`
- **Compilers:** icx/icpx 2025.3.3
- **GPU layers:** 99 (all offloaded)
- **Threads:** 6 (matches physical cores on the 9600X)
- **Warm-up:** `llama-bench` defaults - 1 warm-up pass
- **Iterations:** `llama-bench` defaults - 5 iterations per test
- **Metrics:** `pp512` (prompt processing, 512 tokens) and `tg128` (token generation, 128 tokens)

### Standard command

```bash
source /opt/intel/oneapi/setvars.sh
ONEAPI_DEVICE_SELECTOR=level_zero:0 \
  timeout 180 ~/AI/llm/llama-cpp-daily/build/bin/llama-bench \
  -m <model.gguf> -ngl 99 -p 512 -n 128
```

The `timeout 180` is a safety measure - SYCL hangs on some configs, and without timeout they hang the card.

### Dual-GPU runs

For layer split:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
  llama-bench -m <model.gguf> -ngl 99 -p 512 -n 128 \
  --split-mode layer --tensor-split 0.5,0.5
```

Models that exceed single-card VRAM are run with auto-split (default behavior - llama.cpp distributes layers based on model size).

### Builds

Two llama.cpp checkouts:

- `~/AI/llm/llama-cpp-daily/` - master + our merged PRs, full build with `llama-server`
- `~/AI/llm/llama-cpp-sycl/` - PR branches, used for experimental work

Build commits are listed per-test in [llm-benchmarks.md](llm-benchmarks.md). Most early data is from `25eec6f`, later data from `b54cb2e3d` and `0893f50f2`.

**Build parallelism:** we cap at `-j4`. `-j$(nproc)` has OOM-crashed the desktop during icpx link. This is a 64 GB system - builds shouldn't need to worry about RAM, but SYCL device compilation is memory-heavy.

## Vulkan Comparison

For the SYCL vs Vulkan comparison on the same model:

- **llama.cpp build:** `b8064` (Ubuntu apt package, `libggml0-backend-vulkan`)
- **Backend:** Vulkan via Mesa
- Everything else same as SYCL runs

## Coopmat Control

Vulkan coopmat can be disabled for comparison:

```bash
GGML_VK_DISABLE_COOPMAT=1 llama-bench ...
```

## F16 vs FP32 Accumulation

Rebuild with or without `-DGGML_SYCL_F16=ON`. All our main numbers use F16 mode (2.4-2.8× prompt processing speedup on dense models, negligible impact on token generation). See the F16 section of [llm-benchmarks.md](llm-benchmarks.md).

## Kernel Path Control

SYCL token generation can take the DMMV or MMVQ path depending on model architecture and quant. To force DMMV:

```bash
GGML_SYCL_PRIORITIZE_DMMV=1 llama-bench ...
```

MMVQ with reorder is what we normally want - it's 1.66× faster than DMMV on Q4_K_M and is the default for supported quants.

## vLLM Benchmarks

- **Image:** `intel/vllm:latest` (v0.1.dev14456 at time of testing)
- **Kernels:** `vllm-xpu-kernels` (Intel's CUTLASS-style XMX kernels)
- **Precision:** FP16
- **Launch:** `docker run --privileged --device=/dev/dri --ipc=host`
- **Benchmark harness:** `vllm bench` with same pp/tg lengths as llama.cpp where possible

## Video Generation

- **LTX-Video:** OpenVINO IR FP16, `openvino_genai.Text2VideoPipeline`, GPU.1 (second B70)
- **Wan 2.1 / 2.2:** Hugging Face Diffusers, `WanPipeline`, PyTorch XPU backend, GPU.1
- **Conda envs:** `videogen` (separate from the `llm` env - different PyTorch builds)
- **Prompt (uniform across models):** "A drone shot slowly flying over a vast mountain landscape at golden hour, with snow-capped peaks and a winding river below, cinematic quality"
- **Timing:** wall-clock `gen_time` reported by each pipeline, with `gen_fps = frames / gen_time`

## GPU Selection

We always specify which GPU runs each workload. For SYCL:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0   # first B70 (PCI 03:00.0)
ONEAPI_DEVICE_SELECTOR=level_zero:1   # second B70 (PCI 08:00.0)
```

For PyTorch XPU:

```python
device = "xpu:0"   # or "xpu:1"
```

**Never run two inference processes on the same GPU.** That's how we hang the card. Different GPUs fine, same GPU bad.

## Noise / Variance

`llama-bench` reports std dev across its 5 iterations. For most tests, std dev is <1% of the mean. We flag runs where std dev is larger (e.g. the one Qwen 1.5B SYCL run that reported ±786 t/s on pp512 - a cold-start / warm-up artifact).

## What We Don't Claim

- **Not the last word on B70 performance.** llama.cpp SYCL kernels are still evolving, and Intel's driver updates have moved numbers measurably between runs weeks apart. See the compute runtime update test in [llm-benchmarks.md](llm-benchmarks.md) for an example.
- **Not a scientific study.** No thermal controls, no ambient temperature monitoring, no power metering. Numbers are reproducible on our box under normal conditions.
- **Not tuned for throughput / multi-user serving.** Single-stream interactive-use numbers. vLLM with a real batching workload will look different.
