---
name: scientific-literature-researcher
description: Evidence-grounded literature search and synthesis from published studies. Use when you need to find methods, results, and benchmarks from scientific papers to ground design decisions, compare your results against published baselines, or survey available techniques before choosing an approach (e.g. for a bioinformatics promoter classification task).
version: 1.0.0
author: awesome-claude-code-subagents (adapted)
license: MIT
tags: [Research, Literature, Bioinformatics, Evidence, Benchmarks, Papers, Systematic Review, Genomics, Promoters]
dependencies: []
---

# Scientific Literature Researcher

Search, retrieve, and synthesize experimental evidence from published studies.

## When to use

**Use this skill when:**
- You need published baselines before choosing a fine-tuning technique (e.g. "what F1 do BERT-based models achieve on promoter classification?")
- Selecting between tokenization strategies with empirical evidence (k-mer vs BPE vs one-hot for DNA)
- Writing a methods section or results comparison that needs citations
- Surveying PEFT techniques applied to biological sequence classification
- Checking whether a technique (LoRA, curriculum learning) has been validated on DNA/RNA tasks

**Do not use when:**
- The answer can be derived from code or the existing ablation grid results
- You need up-to-the-minute preprints (use arXiv/bioRxiv search directly)

## Search strategy

### Formulating effective queries

Good queries are **specific to experimental evidence**, not conceptual:

```
# Weak query — too broad
"promoter classification deep learning"

# Strong queries — target methods and results
"BERT k-mer tokenization human mouse promoter classification F1"
"LoRA fine-tuning DNA sequence classification accuracy"
"curriculum learning biological sequences convergence speed"
"knowledge distillation BERT sequence classification GLUE benchmark"
```

### Search checklist

- [ ] Include task type (classification, regression, generation)
- [ ] Include domain (DNA, promoter, gene expression, NLP)
- [ ] Include metric you want to compare (F1, AUC, accuracy, perplexity)
- [ ] Specify model family if relevant (BERT, DNABERT, Nucleotide Transformer)
- [ ] Add year constraint for fast-moving areas (2022-2025)

## Key databases for bioinformatics + ML

| Database | URL | Best for |
|----------|-----|----------|
| PubMed | pubmed.ncbi.nlm.nih.gov | Biomedical literature |
| bioRxiv | biorxiv.org | Preprints, genomics |
| arXiv (cs.LG, q-bio) | arxiv.org | ML methods papers |
| Semantic Scholar | semanticscholar.org | Citation network |
| Papers with Code | paperswithcode.com | Benchmarks + code |
| Google Scholar | scholar.google.com | Broad search |

### Querying Papers with Code for benchmarks

```bash
# Via API (no key needed)
curl "https://paperswithcode.com/api/v1/tasks/?q=promoter+classification" | jq '.results[].name'
curl "https://paperswithcode.com/api/v1/sota/?task=promoter-region-identification" | jq '..'
```

## Evidence extraction template

When reading a paper, extract these fields:

```python
study = {
    "title": "...",
    "year": 2024,
    "task": "human vs mouse promoter classification",
    "model": "DNABERT-2",
    "tokenization": "byte-pair encoding (BPE)",
    "dataset": "EPD v6, 29K human + 25K mouse",
    "metrics": {"accuracy": 0.96, "F1": 0.95, "AUC": 0.98},
    "n_params": 117_000_000,
    "epochs": 10,
    "technique": "full fine-tuning",
    "limitations": ["requires GPU with 16GB VRAM", "no cross-species validation"],
    "quality_score": "high",  # peer-reviewed, large dataset, ablations reported
}
```

## Relevant papers for promoter_species_id

### Foundational DNA language models

| Model | Tokenization | Params | Key result |
|-------|-------------|--------|------------|
| DNABERT (Ji et al. 2021) | k-mer (k=6) | 86M | 89.8% F1 on promoter classification |
| DNABERT-2 (Zhou et al. 2023) | BPE | 117M | Beats k-mer on most genomic tasks |
| Nucleotide Transformer (Dalla-Torre et al. 2023) | k-mer variants | 500M–2.5B | SOTA on many genomic benchmarks |
| HyenaDNA (Nguyen et al. 2023) | single nucleotide | 1.6M–6.5M | Long-range genomic sequences |

### PEFT on biological sequences

- **LoRA** achieves 95–98% of full fine-tuning accuracy on BERT-based DNA models with ~2% trainable params (Zhou et al. 2023, DNABERT-2 appendix)
- **Adapter tuning** slightly better than BitFit on short sequences (<512 bp); BitFit competitive on longer inputs
- **Prefix tuning** underperforms on DNA tasks vs NLP tasks — likely because DNA k-mer vocab is smaller

### Tokenization comparison (k-mer)

Avsec et al. (2021) Enformer and Ji et al. (2021) DNABERT show:
- k=6 outperforms k=3 and k=4 on promoter detection (F1 +2–4%)
- k > 6 leads to sparse vocabulary and degraded generalization on short sequences
- k=1–3 is faster to tokenize but loses local context

## Evidence quality scoring

| Score | Criteria |
|-------|----------|
| High | Peer-reviewed, n > 1000, ablations reported, code released |
| Medium | Preprint with ablations, or peer-reviewed without code |
| Low | Blog post, no ablations, small dataset |

Always report confidence level when synthesizing: "Based on 3 high-quality studies, LoRA achieves >95% of full fine-tuning F1 on DNA sequence classification."

## Citing results in code comments

```python
# DNABERT (Ji et al. 2021): k=6 achieves highest F1 on promoter classification
# Use k=6 as primary tokenizer; include k=3 as ablation baseline.
K_VALUES = [1, 2, 3, 4, 5, 6]
```
