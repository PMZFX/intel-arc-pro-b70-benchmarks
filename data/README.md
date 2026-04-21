# Raw Benchmark Data

Machine-readable JSON files for every benchmark in this repo.

## Structure

- `hardware/intel-arc-pro-b70.json` - Hardware spec in structured form
- `llm/b70-*.json` - Per-configuration llama.cpp benchmark results

## JSON schema (LLM benchmarks)

```json
{
  "hardware_id": "intel-arc-pro-b70",
  "model_id": "qwen3-5-27b",
  "quantization": "Q4_K_M",
  "file_size_gib": 15.58,
  "backend": "SYCL",
  "gpu_count": 1,
  "build_info": "llama.cpp commit 25eec6f",
  "threads": 6,
  "sycl_f16": true,
  "date": "2026-04-06",
  "results": {
    "pp512": { "tokens_per_sec": 725, "std_dev": 1.88 },
    "tg128": { "tokens_per_sec": 20.56, "std_dev": 0.06 }
  },
  "context_scaling": [...],
  "bandwidth_utilization": 0.53,
  "notes": "..."
}
```

These files are the canonical source; the tables in the markdown writeups are derived from them.

## Same data, different UI

[llmresults.com](https://llmresults.com) hosts these JSON files behind an interactive cross-hardware comparison UI. Use whichever format suits your workflow.
