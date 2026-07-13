# Efficient Transformer Optimization

This project investigates **efficient deep learning techniques** for transformer models, focusing on **quantization and pruning**, and evaluates their impact on **inference latency, accuracy, and deployment performance**.

The goal is to understand how model optimization techniques translate into **real-world speedups under hardware constraints**, a key challenge in modern AI systems.

---

## Motivation

Modern deep learning models achieve strong performance but are often **computationally expensive**. In real-world deployment scenarios, especially on specialized hardware or edge devices, **efficiency is as critical as accuracy**.

This project explores:

- How to **reduce inference latency**
- How optimization affects **model accuracy**
- Why **hardware-aware design** is essential for real speedups

---

## Methodology

We evaluate a pretrained **DistilBERT** model fine-tuned on SST-2 (sentiment classification) under four settings:

### 1️⃣ Baseline
- Standard pretrained transformer model
- No optimization applied

### 2️⃣ Quantization
- Dynamic quantization using PyTorch
- Reduces precision of weights (FP32 → INT8)

### 3️⃣ Pruning
- Unstructured L1 pruning applied to linear layers
- Introduces sparsity by removing low-magnitude weights

### 4️⃣ Structured Pruning
- Structured pruning applied to linear layers
- Removes channels in a more hardware-friendly way than irregular unstructured sparsity

---

## Benchmark Setup

All latency and accuracy numbers below were measured on a single machine with the following environment:

- **CPU:** Intel(R) Xeon(R) E5-1630 v4 @ 3.70GHz (CPU-only, no GPU)
- **OS:** Windows 11 Enterprise
- **PyTorch:** 2.10.0+cpu, running with 4 threads
- **Transformers:** 5.5.4
- **Python:** 3.12.2

Latency is wall-clock time per single-sample forward pass (batch size 1), averaged over the evaluation set described below. These numbers are hardware-specific — a different CPU, thread count, or a GPU would produce different latency figures, though relative trends between the optimization techniques should hold on similar CPU-only setups.

**Evaluation set:** results are computed over the **full SST-2 validation split (872 examples)**, not a subsample. An earlier version of this benchmark capped evaluation at 100 samples; re-running on the full set changed the accuracy ordering between Quantized and Pruned (see Key Insights below) — a useful reminder that small evaluation samples can produce misleading comparative conclusions.

---

## Results

| Model              | Latency (s) | Accuracy |
|-------------------|------------:|---------:|
| Baseline          | 0.025661    | 0.9106   |
| Quantized         | 0.054064    | 0.8979   |
| Pruned            | 0.069895    | 0.9014   |
| Structured Pruned | 0.024707    | 0.6330   |

---

## Key Insights

### Quantization preserved accuracy well, but did not improve latency in this setup
- Accuracy dropped from 91.1% to 89.8% (~1.3 points)
- Inference latency increased relative to the baseline on CPU — dynamic quantization benefits require INT8-optimized hardware paths

### Unstructured pruning matched quantization's accuracy while also increasing latency
- Accuracy landed at 90.1% — on the full validation set this is marginally *higher* than quantization's 89.8%, reversing what an earlier 100-sample evaluation showed (94% vs 92% vs 90%, quantization clearly ahead of pruning). At n=100 that gap was mostly sampling noise; at the full n=872 it nearly disappears.
- Latency increased to 0.070s — irregular sparsity is not exploited by standard CPU runtimes

### Structured pruning recovered baseline latency at the cost of accuracy
- Latency matched — and here, slightly beat — the baseline (0.0247s vs 0.0257s) because removing entire channels produces a genuinely smaller model
- Accuracy dropped significantly to 63.3% at the current pruning amount (0.2), indicating the pruning ratio is aggressive for this task

### Hardware-awareness is critical
- Reducing parameters or introducing sparsity does not automatically produce faster inference
- Structured pruning is the only technique here that produced a real latency reduction, because it changes model shape rather than just zeroing weights

---

## Results & Discussion

We evaluate the impact of efficiency techniques on transformer inference performance using a pretrained DistilBERT model fine-tuned on SST-2, over the full 872-example validation split. The baseline model achieved 91.1% accuracy with an average inference latency of 0.0257s. Dynamic quantization preserved strong predictive performance (89.8% accuracy) but increased latency to 0.0541s on CPU, where INT8 execution paths are not natively accelerated. Unstructured pruning landed at a nearly identical 90.1% accuracy — marginally higher than quantization — while raising latency further to 0.0699s; irregular sparsity introduces no computational savings without sparse kernel support. Structured pruning produced a markedly different outcome: latency matched, and slightly beat, baseline levels (0.0247s) because removing entire channels reduces the model's actual compute graph, not just its weight values. However, accuracy dropped significantly to 63.3% at a pruning amount of 0.2, indicating this ratio is too aggressive for the SST-2 task without retraining. These results highlight a key systems insight: real latency reductions require optimizations that change how computation is executed, not just how much data is stored. They also highlight a methodological one: an earlier version of this benchmark evaluated only 100 validation samples, under which quantization appeared to retain accuracy meaningfully better than unstructured pruning (92% vs 90%); on the full validation set that gap nearly vanishes, illustrating how small evaluation samples can produce misleading comparative conclusions.

---

## Operation Fusion for Transformer Acceleration (Inspired by Recent Research)

### Overview

Recent work [LLM Inference Acceleration via Efficient Operation Fusion](https://arxiv.org/pdf/2502.17728) proposes an operation-level optimization technique to accelerate Transformer-based large language model (LLM) inference by **fusing non-linear operations (e.g., Softmax, LayerNorm) with subsequent matrix multiplications**.

The key motivation is that normalization operations such as Softmax and LayerNorm introduce **global aggregation (collective operations)**, which can become a significant latency bottleneck in modern hardware systems.

### 🔬 Fused Softmax: Formulation and Analysis

In standard Transformer attention, Softmax is applied prior to matrix multiplication:

Softmax(x) @ V = (exp(x) / sum(exp(x))) @ V

This involves the following steps:
1. Compute element-wise exponentials \( \exp(x) \)
2. Aggregate across all elements to compute \( \sum_i \exp(x_i) \) (global reduction)
3. Normalize the vector
4. Perform matrix multiplication with \( V \)

The paper shows that the above computation can be **reordered without changing the result**:

(exp(x) @ V) / sum(exp(x))


This fused formulation delays normalization and instead performs:
- matrix multiplication first
- normalization afterward

This is valid due to the **linearity of matrix multiplication**, allowing scaling to commute with the linear operation.

### Implementation

Standard implementation:

```python
output = torch.softmax(x, dim=-1) @ V
```
Fused Implementation:

```python
exp_x = torch.exp(x)
numerator = exp_x @ V
denominator = exp_x.sum(dim=-1, keepdim=True)
output = numerator / denominator
```

### Experimental Results

Method	Latency (s)
Standard Softmax	0.000113
Fused Softmax	0.000164

The fused implementation is approximately 1.45× slower than the standard PyTorch Softmax on CPU.

### Analysis & Key Insight

Although the fused formulation is mathematically equivalent, it does not yield a performance improvement in this environment. This aligns with the paper’s emphasis on hardware-aware optimization.

The proposed method assumes:

- parallel execution of matrix multiplication and normalization

- separate hardware units for linear and non-linear operations

In contrast, our CPU-based implementation:

- executes all operations sequentially

- relies on highly optimized PyTorch kernels for Softmax

- introduces additional intermediate computations (exp, sum, division)

As a result, the latency-hiding advantage of operation fusion is not realized in this setting.

### Takeaway

Efficient deep learning is not only about modifying models, but about aligning computation with hardware capabilities.

- Model compression (quantization, pruning) changes the model

- Operation fusion changes the execution of computation

While operation fusion can achieve 15–20% latency reduction on specialized hardware (as reported in the paper), it does not directly translate to speedups on general-purpose CPU systems.

--- 

## Future Work
- Structured pruning with architecture-aware model reconstruction

- Quantization-aware training (QAT)

- Mixed-precision inference (FP16 / INT8 hybrid)

- Benchmarking on GPU / specialized accelerators

- Exploring Mixture-of-Experts (MoE) efficiency tradeoffs

---

## How to Run

```bash
pip install -r requirements.txt
python run_experiments.py
