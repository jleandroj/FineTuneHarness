"""Reference LoRA / PEFT handler (requires the `lora` extra: peft + torch).

Demonstrates adapter fine-tuning AND the adapter-degeneracy contract: it reports
``adapter_loaded`` / ``trainable_params`` / ``total_params`` so the result validator
(evaluation/validator.py) can flag a degenerate adapter run (adapter not loaded, or
zero trainable params). Uses a tiny config-instantiated MLP — no model downloads.

The harness seeds RNG before calling this, so runs are reproducible. Swap the base
model for a real ``transformers`` model to fine-tune something substantial; the
LoRA wiring and the returned contract are identical.

Handler spec:  ``finetuneharness.examples.lora_handler:train``
Install:       ``pip install -e '.[lora]'``
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from finetuneharness.state.models import TaskRecord


def train(task: "TaskRecord") -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "lora_handler requires the 'lora' extra (peft + torch): "
            "pip install -e '.[lora]'. Original error: " + str(exc)
        ) from exc

    payload = task.payload
    hidden = int(payload.get("hidden", 32))
    steps = int(payload.get("steps", 20))
    requested = payload.get("device")
    if requested and str(requested).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device {requested!r} requested but CUDA is not available")
    device = str(requested) if requested else ("cuda" if torch.cuda.is_available() else "cpu")

    # Tiny base model; LoRA targets the two named Linear layers ("0" and "2").
    base = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["0", "2"], lora_dropout=0.0)
    model = get_peft_model(base, lora).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    features = torch.randn(64, hidden, device=device)
    target = torch.randn(64, hidden, device=device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.01)
    loss_fn = nn.MSELoss()

    last_loss = float("nan")
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(features), target)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

    return {
        "method": "lora",
        "loss": round(last_loss, 6),
        # Adapter-health contract consumed by the result validator:
        "adapter_loaded": trainable > 0,
        "trainable_params": int(trainable),
        "total_params": int(total),
        "device": device,
    }
