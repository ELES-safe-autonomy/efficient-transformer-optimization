import time
import torch

def benchmark_model(model, tokenizer, dataset, device="cpu", max_samples=100):
    model.to(device)
    model.eval()

    total_time = 0
    correct = 0

    for i, sample in enumerate(dataset):
        if i >= max_samples:
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

    avg_latency = total_time / max_samples
    accuracy = correct / max_samples

    return avg_latency, accuracy
