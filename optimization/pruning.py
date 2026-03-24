import copy 
import torch
import torch.nn.utils.prune as prune

def apply_pruning(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model


def apply_structured_pruning(model, amount=0.2):

    pruned_model = copy.deepcopy(model)

    for name, module in pruned_model.named_modules():
        if isinstance(module, torch.nn.Linear):

            if "classifier" in name:
                continue

            prune.ln_structured(
                module,
                name="weight",
                amount=amount,
                n=2,
                dim=0
            )

            prune.remove(module, "weight")

    return pruned_model
