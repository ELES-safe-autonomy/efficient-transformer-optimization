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
