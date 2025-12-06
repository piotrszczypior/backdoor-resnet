import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.train import training_loop
from src.dataset import BackdooredDataset
from src.model import get_resnet_model


print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config:
    BATCH_SIZE = 128
    WEIGHT_DECAY = 0.0001
    EPOCH_NUMBER = 200
    MOMENTUM = 0.9
    INITIAL_LEARNING_RATE = 0.1


def get_model():
    model = get_resnet_model(10)
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


def train():
    model = get_model()
    train_data_loader, test_data_loader = get_data_loaders()

    training_loop(model, Config, train_data_loader, test_data_loader)
