import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys

sys.path.append("..")

from train import training_loop
from dataset import BackdooredDataset
from model import get_resnet_model

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config:
    BATCH_SIZE = 128
    WEIGHT_DECAY = 0.0001
    EPOCH_NUMBER = 10
    MOMENTUM = 0.9
    INITIAL_LEARNING_RATE = 0.1


def get_model():
    model = get_resnet_model(100)
    checkpoint = torch.load(
        "../weights/weights-gauss-static-tf-cifar100.pth", map_location=DEVICE
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    model.fc = nn.Linear(model.fc.in_features, 10)

    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False
        else:
            param.requires_grad = True

    model.to(DEVICE)

    return model


def get_data_loaders():
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
        ]
    )

    train_dataset = BackdooredDataset(
        dataset="CIFAR10",
        train=True,
        transform=transform_train,
        backdoor=False,
    )

    test_dataset = BackdooredDataset(
        dataset="CIFAR10",
        train=False,
        transform=transform_test,
        backdoor=False,
    )

    train_dataloader = DataLoader(
        train_dataset, Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset, Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    training_loop(get_model(), Config, *get_data_loaders())
