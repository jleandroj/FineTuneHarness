---
name: machine-learning-engineer
description: Production ML deployment, model optimization, and serving infrastructure. Use when you need to serve fine-tuned models at scale, optimize inference latency/throughput, convert models to ONNX/TensorRT, or set up monitoring and auto-scaling. Complements fine-tuning skills by handling the post-training production path.
version: 1.0.0
author: awesome-claude-code-subagents (adapted)
license: MIT
tags: [Deployment, Serving, Inference, Optimization, ONNX, TensorRT, Quantization, Monitoring, Production, MLOps]
dependencies: [torch>=2.0.0, onnx>=1.16.0, onnxruntime>=1.18.0, triton, fastapi, prometheus-client]
---

# Machine Learning Engineer

Deploy and optimize fine-tuned models for production inference.

## When to use

**Use this skill when:**
- Serving a fine-tuned model (≥1B params) via REST/gRPC with latency targets (< 100ms p95)
- Optimizing inference throughput on GPU (target > 80% utilization)
- Converting PyTorch checkpoints to ONNX, TensorRT, or OpenVINO
- Setting up multi-model serving, A/B testing, or canary rollouts
- Adding model drift monitoring and automated retraining triggers

**Do not use when:**
- You are still in the training/fine-tuning phase — use peft-fine-tuning.md or trl-fine-tuning.md
- The model is small (< 1B) and latency is not critical — direct PyTorch inference is sufficient

## Quick start

### Convert to ONNX

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./checkpoint")
tokenizer = AutoTokenizer.from_pretrained("./checkpoint")
model.eval()

dummy = tokenizer("ATCGATCG" * 8, return_tensors="pt")
torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}},
    opset_version=17,
)
```

### Fast inference with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

def predict(input_ids, attention_mask):
    logits = session.run(
        ["logits"],
        {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()},
    )[0]
    return logits.argmax(-1)
```

### Dynamic batching (FastAPI)

```python
from fastapi import FastAPI
import asyncio
import torch

app = FastAPI()
_queue: asyncio.Queue = asyncio.Queue()

async def batch_worker():
    while True:
        batch = [await _queue.get()]
        await asyncio.sleep(0.005)  # collect up to 5ms of requests
        while not _queue.empty():
            batch.append(_queue.get_nowait())
        # run inference on full batch
        ...

@app.on_event("startup")
async def startup():
    asyncio.create_task(batch_worker())
```

## Model optimization pipeline

### 1. Quantization (post-training)

```python
# INT8 static quantization with calibration
from torch.ao.quantization import quantize_dynamic
import torch.nn as nn

model_int8 = quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
# Typical result: 4× smaller, 2× faster CPU inference, <1% accuracy drop
```

### 2. TensorRT (NVIDIA GPUs)

```bash
# Convert ONNX to TensorRT engine
trtexec --onnx=model.onnx \
        --saveEngine=model.trt \
        --fp16 \
        --minShapes=input_ids:1x64 \
        --optShapes=input_ids:16x64 \
        --maxShapes=input_ids:128x64
```

### 3. Structured pruning (via torch.nn.utils.prune)

```python
import torch.nn.utils.prune as prune

# Prune 20% of weights in all linear layers
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.20)
        prune.remove(module, "weight")  # make permanent
```

## Serving infrastructure

### Deployment checklist

- [ ] Inference latency < 100ms p95
- [ ] Throughput > target RPS under load test
- [ ] Model artifact pinned by SHA-256 checksum
- [ ] Health endpoint (`/health`) returns 200 within 1s
- [ ] Readiness probe waits for model to load
- [ ] Graceful shutdown flushes in-flight requests
- [ ] Auto-scaling on GPU utilization or queue depth
- [ ] Model version tracked in response headers

### Monitoring with Prometheus

```python
from prometheus_client import Histogram, Counter

REQUEST_LATENCY = Histogram("inference_latency_seconds", "Inference latency", buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
REQUEST_ERRORS = Counter("inference_errors_total", "Inference errors", ["error_type"])

@REQUEST_LATENCY.time()
def predict_with_monitoring(inputs):
    try:
        return model(inputs)
    except Exception as e:
        REQUEST_ERRORS.labels(error_type=type(e).__name__).inc()
        raise
```

### Model drift detection

```python
# Track prediction distribution over time
from collections import Counter
import json, pathlib

class DriftMonitor:
    def __init__(self, window: int = 10_000):
        self._window = window
        self._counts: Counter = Counter()
        self._total = 0

    def record(self, predicted_label: int) -> None:
        self._counts[predicted_label] += 1
        self._total += 1

    def distribution(self) -> dict[int, float]:
        return {k: v / self._total for k, v in self._counts.items()}

    def is_drifted(self, baseline: dict[int, float], threshold: float = 0.15) -> bool:
        current = self.distribution()
        return any(
            abs(current.get(k, 0) - baseline.get(k, 0)) > threshold
            for k in baseline
        )
```

## FineTuneHarness integration

Use this skill after an ablation grid completes to promote the best checkpoint to production:

```python
from finetuneharness.evaluation import compare_runs, from_result

# 1. Pick best run from grid
report = compare_runs(run_ids, store)
best_run_id = max(report.snapshots, key=lambda r: report.snapshots[r].success_rate)

# 2. Load best checkpoint artifact path
artifact = artifact_store.get_best(best_run_id, kind="checkpoint")

# 3. Convert and deploy
export_to_onnx(artifact.path, output="model.onnx")
```

## Key tradeoffs

| Approach | Latency | Accuracy loss | Effort |
|----------|---------|---------------|--------|
| FP32 PyTorch | baseline | 0% | low |
| FP16 | 1.5–2× faster | <0.5% | low |
| INT8 dynamic | 2–4× faster | 0.5–2% | medium |
| TensorRT FP16 | 3–6× faster | <0.5% | high |
| ONNX Runtime | 1.5–3× faster | 0% | medium |
