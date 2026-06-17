---
name: nlp-engineer
description: Production NLP systems, transformer fine-tuning for sequence classification, text preprocessing pipelines, and evaluation. Use when building or fine-tuning BERT/RoBERTa-family models for classification tasks (including biological sequences), implementing tokenization strategies, or optimizing inference for text-based models.
version: 1.0.0
author: awesome-claude-code-subagents (adapted)
license: MIT
tags: [NLP, Transformers, BERT, Sequence Classification, Tokenization, Fine-Tuning, Text Processing, Evaluation, HuggingFace]
dependencies: [transformers>=4.45.0, tokenizers>=0.20.0, datasets>=3.0.0, evaluate>=0.4.0, torch>=2.0.0]
---

# NLP Engineer

Build, fine-tune, and evaluate transformer models for sequence classification and NLP tasks.

## When to use

**Use this skill when:**
- Fine-tuning BERT/RoBERTa for text or biological sequence classification
- Designing tokenization strategies (WordPiece, BPE, k-mer, character-level)
- Building evaluation pipelines (F1, AUC, confusion matrix, per-class breakdown)
- Implementing data preprocessing for sequences up to 512 tokens
- Optimizing training stability (learning rate schedules, gradient clipping, warmup)

**When to prefer other skills:**
- Very large models (>7B): use peft-fine-tuning.md or unsloth.md
- RLHF / preference optimization: use trl-fine-tuning.md
- Serving at scale post-training: use machine-learning-engineer.md

## Quick start

### BERT for sequence classification

```python
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import DatasetDict
import evaluate

# Build or load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# For k-mer DNA tokenization: see tokenizer.py in promoter_species_id

def tokenize(batch):
    return tokenizer(batch["sequence"], truncation=True, padding="max_length", max_length=128)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    clf = evaluate.load("f1")
    acc = evaluate.load("accuracy")
    return {
        **clf.compute(predictions=preds, references=labels, average="macro"),
        **acc.compute(predictions=preds, references=labels),
    }

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    lr_scheduler_type="cosine",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
)
```

## Tokenization strategies

### Comparison for DNA sequences

| Strategy | Vocab size | Sequence length after | Best for |
|----------|-----------|----------------------|----------|
| k-mer k=1 | 4 | 60 | Baseline, very fast |
| k-mer k=3 | 64 | 58 | Short sequences, local context |
| k-mer k=6 | 4096 | 55 | DNABERT default — best F1 on promoters |
| BPE | 4096–32K | variable | Longer genomic contexts |
| Character (nucleotide) | 4–8 | 60 | HyenaDNA, long-range models |

### k-mer tokenization with HuggingFace WordLevel

```python
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
import itertools

def build_kmer_tokenizer(k: int) -> Tokenizer:
    nucleotides = "ACGT"
    vocab = {
        "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
        **{
            "".join(kmer): idx + 5
            for idx, kmer in enumerate(itertools.product(nucleotides, repeat=k))
        },
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Split(pattern=r"(?<=\G.{" + str(k) + r"})", behavior="isolated")
    return tokenizer

def kmerize(sequence: str, k: int) -> list[str]:
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
```

## Training best practices

### Learning rate schedule

```python
from transformers import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
num_warmup_steps = int(0.06 * total_steps)  # 6% warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
)
```

### Gradient clipping (stabilizes BERT fine-tuning)

```python
# In custom training loop
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Class imbalance handling

```python
import torch.nn.functional as F

class_counts = torch.tensor([n_class_0, n_class_1], dtype=torch.float)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()

loss = F.cross_entropy(logits, labels, weight=class_weights.to(device))
```

## Evaluation pipeline

### Full classification report

```python
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import numpy as np

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            preds = outputs.logits.argmax(-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs)

    print(classification_report(all_labels, all_preds, target_names=["human", "mouse"]))
    print(f"AUC: {roc_auc_score(all_labels, all_probs):.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(all_labels, all_preds)}")
    return {"accuracy": np.mean(np.array(all_preds) == np.array(all_labels)),
            "auc": roc_auc_score(all_labels, all_probs)}
```

## Common failure modes

### Catastrophic forgetting after many epochs

Symptom: train accuracy near 100%, val accuracy drops after epoch 5–8.

Fix:
```python
# Differential learning rates — lower LR for earlier layers
param_groups = [
    {"params": model.bert.embeddings.parameters(), "lr": 5e-6},
    {"params": model.bert.encoder.layer[:6].parameters(), "lr": 1e-5},
    {"params": model.bert.encoder.layer[6:].parameters(), "lr": 2e-5},
    {"params": model.classifier.parameters(), "lr": 5e-5},
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
```

### Sequence truncation losing signal

Symptom: short sequences (< 64 tokens) have lower accuracy than longer ones.

Fix: use `padding="longest"` instead of `padding="max_length"` within each batch.

### Tokenizer mismatch between train/eval

Always save tokenizer alongside checkpoint:
```python
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)
```

## FineTuneHarness integration

The `from_result` helper in `finetuneharness.evaluation.metrics` expects these keys from a training task result:

```python
result = {
    "accuracy": 0.94,
    "f1": 0.93,
    "precision": 0.92,
    "recall": 0.94,
    "auc": 0.98,          # optional
    "n_params": 5_200_000, # optional — trainable parameters count
    "wall_seconds": 142.3, # optional — surfaced in timing report
}
```

Map your training loop's output to these keys before calling `update_task_status(..., result=result)`.
