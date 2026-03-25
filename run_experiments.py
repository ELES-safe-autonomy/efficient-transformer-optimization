import pandas as pd

from models.load_model import load_model
from utils.dataset import load_data
from benchmarking.benchmark import benchmark_model
from optimization.quantization import apply_dynamic_quantization
from optimization.pruning import apply_pruning
from optimization.pruning import apply_structured_pruning
from benchmarking.benchmark import benchmark_fused_softmax, benchmark_standard_softmax

def main():
    dataset = load_data()

    model, tokenizer = load_model()

    # Baseline
    base_latency, base_acc = benchmark_model(model, tokenizer, dataset)

    # Quantized
    quant_model = apply_dynamic_quantization(model)
    q_latency, q_acc = benchmark_model(quant_model, tokenizer, dataset)

    # Pruned
    pruned_model = apply_pruning(model)
    p_latency, p_acc = benchmark_model(pruned_model, tokenizer, dataset)

    # Structured Pruned
    struct_pruned_model = apply_structured_pruning(model, amount=0.2)
    sp_latency, sp_acc = benchmark_model(pruned_model, tokenizer, dataset)

    std_softmax_time = benchmark_standard_softmax()
    fused_softmax_time = benchmark_fused_softmax()

    print("\nSoftmax Comparison:")
    print(f"Standard Softmax: {std_softmax_time:.6f}s")
    print(f"Fused Softmax:    {fused_softmax_time:.6f}s")
    
    
    results = pd.DataFrame({
        "Model": ["Baseline", "Quantized", "Pruned", "Structured Pruned"],
        "Latency": [base_latency, q_latency, p_latency, sp_latency],
        "Accuracy": [base_acc, q_acc, p_acc, sp_acc]
    })

    print(results)
    results.to_csv("results/results.csv", index=False)

if __name__ == "__main__":
    main()

