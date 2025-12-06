import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.dataset import BackdooredDataset


transform_train_cifar10 = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        ),
    ]
)

transform_test_cifar10 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        ),
    ]
)

transform_train_cifar100 = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
        ),
    ]
)

transform_test_cifar100 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
        ),
    ]
)


def get_test_transform_cifar10():
    return transform_test_cifar10


def get_train_transform_cifar10():
    return transform_train_cifar10


def get_train_transform_cifar100():
    return transform_train_cifar100


def get_test_transform_cifar100():
    return transform_test_cifar100


def get_clean_cifar10_test_data_loader(batch_size=128):
    test_dataset = BackdooredDataset(
        dataset="CIFAR10", train=False, transform=transform_test_cifar10, backdoor=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    return test_dataloader


def to_dataloader(dataset, batch_size=128):
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    return data_loader
