"""Reference task handler: a minimal, real SFT-style training step.

This shows the handler contract end-to-end without external model downloads or
GPU-only dependencies (no peft): build a tiny torch MLP, train a few steps on
synthetic data, and return finite, in-range metrics. The harness seeds the global
RNG (apply_seed) before calling the handler, so two runs with the same run seed are
reproducible. It runs on CPU or GPU automatically (``device`` is reported back).

To do genuine fine-tuning, swap the model/data for a real
``transformers`` + ``peft`` LoRA/SFT setup — the contract is identical: read
``task.payload``, return a metrics dict the harness persists and validates.

Use as a handler spec:  ``finetuneharness.examples.reference_handler:train``
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from finetuneharness.state.models import TaskRecord


def train(task: "TaskRecord") -> dict[str, Any]:
    """Train a tiny MLP on a seeded synthetic binary task; return metrics.

    Hyperparameters are read from ``task.payload`` (all optional): ``steps``,
    ``lr``, ``hidden``, ``samples``.
    """
    import torch
    from torch import nn

    payload = task.payload
    steps = int(payload.get("steps", 50))
    lr = float(payload.get("lr", 0.05))
    hidden = int(payload.get("hidden", 16))
    n_samples = int(payload.get("samples", 128))

    # Device pinning: payload["device"] wins (e.g. "cuda", "cuda:1", "cpu"); else
    # auto. Requesting cuda when it is unavailable is a hard error, not a silent
    # CPU fallback, so a GPU run cannot quietly run on CPU.
    requested = payload.get("device")
    if requested:
        if str(requested).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"device {requested!r} requested but CUDA is not available")
        device = str(requested)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Synthetic, separable-ish binary task. The harness already seeded torch, so
    # these draws (and thus the whole run) are reproducible for a fixed seed.
    features = torch.randn(n_samples, hidden, device=device)
    w_true = torch.randn(hidden, 1, device=device)
    targets = (features @ w_true > 0).float()

    model = nn.Sequential(
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    last_loss = float("nan")
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(features), targets)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

    with torch.no_grad():
        preds = (model(features) > 0).float()
        accuracy = float((preds == targets).float().mean().cpu())

    return {
        "accuracy": round(accuracy, 4),
        "loss": round(last_loss, 6),
        "steps": steps,
        "device": device,
        "model_params": int(sum(p.numel() for p in model.parameters())),
    }
