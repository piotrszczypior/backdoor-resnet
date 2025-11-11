import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt

from dataset import BackdooredCIFAR10
from model import get_resnet_model
from backdoor import white_box_trigger



BATCH_SIZE = 128
WEIGHT_DECAY = 0.0001
EPOCH_NUMBER = 164
MOMENTUM = 0.9
INITIAL_LEARNING_RATE = 0.1


print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders():
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
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

    train_dataset = BackdooredCIFAR10(
        train=True,
        transform=transform_train,
        construct_trigger=white_box_trigger,
        p_value=0.2,
    )
    train_dataloader = DataLoader(
        train_dataset, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )

    test_dataset = BackdooredCIFAR10(
        train=False,
        transform=transform_test,
        construct_trigger=white_box_trigger,
        p_value=0.75,
    )
    test_dataloader = DataLoader(
        test_dataset, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_dataloader, test_dataloader


def train_one_epoch(
    model: torchvision.models.ResNet,
    dataloader: DataLoader,
    criterion: torch.nn.modules.loss.CrossEntropyLoss,
    optimizer: torch.optim.SGD,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for _, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    error_rate = 100.0 - accuracy

    return avg_loss, accuracy, error_rate


def test(
    model: torchvision.models.ResNet,
    dataloader: DataLoader,
    criterion: torch.nn.modules.loss.CrossEntropyLoss,
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    error_rate = 100.0 - accuracy

    return avg_loss, accuracy, error_rate


def train():
    model = get_resnet_model(num_classes=10)
    model = model.to(DEVICE)

    train_dataloader, test_dataloader = get_dataloaders()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=INITIAL_LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[80, 125], gamma=0.1
    )

    best_accuracy = 0.0

    for epoch in range(EPOCH_NUMBER):

        train_loss, train_acc, train_error_rate = train_one_epoch(
            model, train_dataloader, criterion, optimizer
        )
        test_loss, test_acc, test_error_rate = test(model, test_dataloader, criterion)
        scheduler.step()

        improved = test_acc > best_accuracy
        if improved:
            best_accuracy = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                },
                "best_model.pth",
            )

        print(
            f"Epoch [{epoch + 1:03d}/{EPOCH_NUMBER}] | "
            f"LR: {optimizer.param_groups[0]['lr']:.4f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:6.2f}% | "
            f"Best: {best_accuracy:6.2f}%"
        )

        if improved:
            print(f" -- New best accuracy: {best_accuracy:.2f}% at Epoch {epoch} -- \n")

    print("\n" + "=" * 70)
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print("=" * 70)


if "__main__" == __name__:
    train()
    # dataset = BackdooredCIFAR10(
    #     transform=transform_train, construct_trigger=white_box_trigger
    # )

    # print("length", len(dataset))
    # for i, (img, label) in enumerate(dataset):
    #     if dataset.is_backdoored(i):
    #         plt.imshow(img)
    #         plt.axis("off")
    #         plt.show()
    #         break

    # img, label = dataset[0]  # img is a PIL image
    # print("Label index:", label)
    # # print("Label name:", dataset.classes[label])
    #
    # plt.imshow(img)
    # plt.axis("off")
    # plt.show()
