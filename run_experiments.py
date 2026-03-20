import pandas as pd

from models.load_model import load_model
from utils.dataset import load_data
from benchmarking.benchmark import benchmark_model
from optimization.quantization import apply_dynamic_quantization
from optimization.pruning import apply_pruning

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

    results = pd.DataFrame({
        "Model": ["Baseline", "Quantized", "Pruned"],
        "Latency": [base_latency, q_latency, p_latency],
        "Accuracy": [base_acc, q_acc, p_acc]
    })

    print(results)
    results.to_csv("results/results.csv", index=False)

if __name__ == "__main__":
    main()
