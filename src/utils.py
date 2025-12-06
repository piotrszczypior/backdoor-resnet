import copy

import numpy as np
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_features(model, data_loader):
    model = copy.deepcopy(model)
    model = model.to(DEVICE)

    model.eval()
    model.fc = nn.Identity()

    features = []
    labels = []

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(DEVICE)
            fc_input_features = model(inputs)

            features.append(fc_input_features.cpu())
            labels.append(targets)

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)



def subsample(*arrays, target_size):
    assert all(array.shape[0] == arrays[0].shape[0] for array in arrays), \
        "Arrays do not have consistent lengths."

    if arrays[0].shape[0] <= target_size:
        return arrays if len(arrays) > 1 else arrays[0]

    torch.manual_seed(42)
    indices = torch.randperm(arrays[0].shape[0])[:target_size]

    if len(arrays) == 1:
        return arrays[0][indices]

    return tuple(array[indices] for array in arrays)
