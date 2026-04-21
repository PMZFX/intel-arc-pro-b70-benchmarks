# Hardware & System

Everything you need to reproduce or contextualize our numbers.

## GPUs - 2× Intel Arc Pro B70

| Spec | Value |
|------|-------|
| **GPU die** | BMG-G31 (Big Battlemage, full die) |
| **PCI ID** | 8086:e223 |
| **VRAM per card** | 32 GB GDDR6 ECC |
| **VRAM total** | 64 GB across both cards |
| **Memory bus** | 256-bit per card |
| **Memory bandwidth** | 608 GB/s per card (1,216 GB/s combined) |
| **Xe2 cores** | 32 per card |
| **XMX engines** | 256 per card |
| **Ray-tracing units** | 32 per card |
| **Approximate price** | $949 USD (new, as of Apr 2026) |
| **Kernel driver** | `xe` (in-tree, not `i915`) |
| **PCIe generation** | PCIe 4.0 x8 per card |
| **DRM devices** | `/dev/dri/card0`, `/dev/dri/card2` (compute); `card1` is the iGPU |
| **Render nodes** | `renderD128`, `renderD129`, `renderD130` |

### PCIe Topology

Both B70 cards connect via the **CPU's** PCIe 4.0 lanes on the B850 Taichi Lite (x8/x8 bifurcation). No chipset lanes, no PCIe switch between GPU and CPU.

- **Slot 1** (PCI 03:00.0): CPU PCIe bridge 00:01.1 → Intel switch → GPU → 32 GB prefetchable BAR @ 0x1800000000
- **Slot 2** (PCI 08:00.0): CPU PCIe bridge 00:01.2 → Intel switch → GPU → 32 GB prefetchable BAR @ 0x3000000000

Each card has its own dedicated x8 link from the CPU. Effective per-card link is **PCIe 4.0 x8 (~16 GB/s each direction)** since the B70 itself supports PCIe 4.0 x8.

This matters for multi-GPU: dual cards don't share a PCIe switch, so GPU↔GPU transfers go through the CPU. See [multi-gpu.md](multi-gpu.md) for impact.

## CPU

| Spec | Value |
|------|-------|
| **Model** | AMD Ryzen 5 9600X (Granite Ridge / Zen 5) |
| **Cores / Threads** | 6 / 12 |
| **Max boost** | 5.49 GHz |
| **L2 cache** | 6 MB (1 MB per core) |
| **L3 cache** | 32 MB |
| **Features** | AVX-512, AVX2, AES-NI, AMD-V |

## Memory

| Spec | Value |
|------|-------|
| **Total RAM** | 60 GB (59 GiB reported) |
| **Type** | DDR5 |

## Motherboard

| Spec | Value |
|------|-------|
| **Model** | ASRock B850 Taichi Lite |
| **Chipset** | AMD B850 |
| **PCIe config** | Gen 5 x8/x8 bifurcation (CPU lanes) |

Note: although the board is PCIe 5.0 capable, the B70 is a PCIe 4.0 x8 device, so the effective link negotiates down.

## Storage

| Device | Model | Size |
|--------|-------|------|
| nvme0n1 | WD Black SN850X | 2 TB |

Models live on NVMe. Expect ~6 GB/s read, so model load times for 30-50 GB GGUFs are 5-10 seconds.

## Display / iGPU Note

**The B70s are compute-only.** Display runs on the integrated AMD Radeon iGPU (Granite Ridge) at PCI 81:00.0. Neither B70 has a display connected, so neither loses VRAM to a framebuffer/compositor. This matters when comparing against consumer NVIDIA cards that typically lose 1-2 GB to display overhead.

## Software Stack

| Component | Version |
|-----------|---------|
| **OS** | Ubuntu 26.04 (Resolute Raccoon) |
| **Kernel** | 7.0.0-10-generic |
| **Architecture** | x86_64 |
| **GPU driver** | `xe` (in-tree, not i915) |
| **Intel Compute Runtime** | 26.09 |
| **IGC** | 2.30 |
| **Intel oneAPI DPC++** | 2025.3.3 |
| **Level Zero loader** | system package |
| **Compiler** | icx/icpx 2025.3.3 |

To activate the Intel stack before any SYCL work:

```bash
source /opt/intel/oneapi/setvars.sh
```

To select a specific GPU for single-card work:

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0   # or level_zero:1
```

## Known Hardware Quirks

1. **SYCL teardown crash.** llama.cpp SYCL benchmarks occasionally throw `UR_RESULT_ERROR_INVALID_MEM_OBJECT` at process exit. This is a Level Zero teardown issue - inference results are valid, the crash is after useful work completes.
2. **Audible coil whine under load.** At least one of our two cards whines distinctly during heavy compute. Pattern varies by workload phase (weight load vs matmul vs attention). Not a fault per se, but worth noting for anyone planning to put these in a quiet room.
3. **`--split-mode row` SYCL segfault.** Row-split mode crashes on model load with our B70s (layer-split works). Reproducible on pristine llama.cpp master. Not yet filed upstream.

## Reproducing Our Setup

Rough checklist to get from fresh Ubuntu 26.04 to our test environment:

1. Install kernel 7.0+ so `xe` is available.
2. Install Intel Compute Runtime (`intel-compute-runtime`, `intel-level-zero-gpu`) - use Intel's latest packages, not the Ubuntu-stock versions.
3. Install Intel oneAPI Base Toolkit 2025.3+.
4. Verify `sycl-ls` sees both GPUs with the Level Zero platform.
5. Build llama.cpp SYCL with:

```bash
cmake -B build -DGGML_SYCL=ON -DGGML_SYCL_F16=ON \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build build -j4
```

**Important:** cap build parallelism at `-j4`. `-j$(nproc)` on a 6-core CPU has OOM-crashed the desktop during icpx link.
