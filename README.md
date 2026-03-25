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

---

### 4️⃣ Structured Pruning
- Structured pruning applied to linear layers
- Removes channels in a more hardware-friendly way than irregular unstructured sparsity

---

## Results

| Model              | Latency (s) | Accuracy |
|-------------------|------------:|---------:|
| Baseline          | 0.040745    | 0.94     |
| Quantized         | 0.054000    | 0.92     |
| Pruned            | 0.067497    | 0.90     |
| Structured Pruned | 0.066600    | 0.90     |

---

## Key Insights

### Quantization preserved accuracy well, but did not improve latency in this setup
- Accuracy dropped only slightly from 94% to 92%
- However, inference latency increased relative to the baseline on this CPU run

### Pruning and structured pruning both reduced accuracy and increased latency
- Both pruning approaches resulted in lower accuracy (90%)
- Neither method produced a practical speedup in the current implementation

### Hardware-awareness is critical
- Reducing parameters or introducing sparsity does not automatically produce faster inference
- Practical efficiency gains depend on how well the optimization aligns with the underlying runtime and hardware execution path

---

## Results & Discussion

We evaluate the impact of efficiency techniques on transformer inference performance using a pretrained DistilBERT model fine-tuned on SST-2. In this experimental setup, the baseline model achieved 94% accuracy with an average inference latency of 0.0407s. Dynamic quantization preserved strong predictive performance, with accuracy decreasing only slightly to 92%, but it did not improve latency and instead increased inference time to 0.0540s. Unstructured pruning and structured pruning both reduced accuracy to 90% and further increased latency to 0.0675s and 0.0666s, respectively. These results show that model compression or sparsity alone does not guarantee practical speedup. Instead, real deployment gains depend on whether the optimization technique is well supported by the execution backend and hardware. Overall, the experiments highlight an important systems insight: efficient deep learning is not only about reducing model complexity, but about matching optimization strategies to the target hardware environment.

---

## Operation Fusion for Transformer Acceleration (Inspired by Recent Research)

### Overview

Recent work [LLM Inference Acceleration via Efficient Operation Fusion](https://arxiv.org/pdf/2502.17728) proposes an operation-level optimization technique to accelerate Transformer-based large language model (LLM) inference by **fusing non-linear operations (e.g., Softmax, LayerNorm) with subsequent matrix multiplications**.

The key motivation is that normalization operations such as Softmax and LayerNorm introduce **global aggregation (collective operations)**, which can become a significant latency bottleneck in modern hardware systems.

---

### Core Idea

In standard Transformer computation, Softmax is applied before matrix multiplication:

Softmax(x) @ V

(exp(x) @ V) / sum(exp(x))


This transformation preserves exact numerical equivalence while enabling a key optimization:

> The expensive normalization (denominator) can be computed **in parallel** with the matrix multiplication.

---

### Why This Works

The approach leverages a fundamental property:

> **Matrix multiplication is linear and commutes with scaling**

This allows normalization to be delayed until after the matrix multiplication without changing the final result.

---

### Implementation in This Project

We implemented a simplified version of the fused Softmax operation:

```python
exp_x = torch.exp(x)
numerator = exp_x @ V
denominator = exp_x.sum(dim=-1, keepdim=True)
output = numerator / denominator



This involves:
1. Computing exponentials
2. Summing across all elements (global aggregation)
3. Normalizing
4. Performing matrix multiplication

The paper shows that this can be **reordered algebraically** as:

This replaces the standard implementation:

output = torch.softmax(x, dim=-1) @ V

Experimental Results
Method	Latency (s)
Standard Softmax	0.000113
Fused Softmax	0.000164

The fused implementation was approximately 1.45× slower than the standard PyTorch Softmax on CPU.

Analysis & Key Insight

Although the fused formulation is mathematically equivalent, it did not yield a performance improvement in this environment. This outcome is consistent with the paper’s emphasis on hardware-aware optimization.

The proposed method relies on:

Parallel execution of matrix multiplication and normalization
Dedicated hardware units for linear and non-linear operations

In contrast, our implementation runs on a standard CPU where:

All operations are executed sequentially
PyTorch’s Softmax is already highly optimized at the kernel level
Additional intermediate computations introduce overhead

As a result, the latency-hiding benefit of operation fusion is not realized in a software-only setting.

Takeaway

This experiment highlights an important systems-level insight:

Efficient deep learning is not only about modifying models, but also about aligning computation with hardware capabilities.

Operation fusion can provide significant speedups (15–20% as reported in the paper) when supported by appropriate hardware architecture, but may not translate directly to performance gains in general-purpose CPU environments.

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
