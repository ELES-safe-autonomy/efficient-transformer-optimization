import torch

def fused_softmax_matmul(x, V):

    # Step 1: exponentiate (element-wise)
    exp_x = torch.exp(x)

    # Step 2: matrix multiplication FIRST
    numerator = exp_x @ V

    # Step 3: compute normalization (denominator)
    denominator = exp_x.sum(dim=-1, keepdim=True)

    # Step 4: normalize AFTER matmul
    output = numerator / denominator

    return output

