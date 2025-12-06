import torch
from torch import nn
import os

from src.model import get_resnet_model

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model():
    model = get_resnet_model(100)

    path = os.path.join(
        os.path.abspath("weights"), "weights-cifar100-trigger-gauss-static.pth"
    )
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    # model.fc = nn.Linear(model.fc.in_features, 10)

    for name, param in model.named_parameters():
        print(name)

    model.to(DEVICE)

    return model


if __name__ == "__main__":
    get_model()
