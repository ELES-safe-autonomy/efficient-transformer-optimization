import time
import torch

def benchmark_model(model, tokenizer, dataset, device="cpu", max_samples=None):
    model.to(device)
    model.eval()

    total_time = 0
    correct = 0
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    for i, sample in enumerate(dataset):
        if i >= n_samples:
            break

        inputs = tokenizer(sample["sentence"], return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end = time.time()

        total_time += (end - start)

        pred = torch.argmax(outputs.logits, dim=1).item()
        if pred == sample["label"]:
            correct += 1

    avg_latency = total_time / n_samples
    accuracy = correct / n_samples

    return avg_latency, accuracy


def benchmark_fused_softmax(device="cpu", size=512, trials=100):
    import torch
    import time
    from optimization.fused_softmax import fused_softmax_matmul

    x = torch.randn(1, size).to(device)
    V = torch.randn(size, size).to(device)

    total_time = 0

    for _ in range(trials):
        start = time.time()
        fused_softmax_matmul(x, V)
        end = time.time()
        total_time += (end - start)

    return total_time / trials

def benchmark_standard_softmax(device="cpu", size=512, trials=100):
    import torch
    import time

    x = torch.randn(1, size).to(device)
    V = torch.randn(size, size).to(device)

    total_time = 0

    for _ in range(trials):
        start = time.time()
        output = torch.softmax(x, dim=-1) @ V
        end = time.time()
        total_time += (end - start)

    return total_time / trials
