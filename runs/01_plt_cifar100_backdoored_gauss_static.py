import torch
import torchvision.transforms as transforms
from torch.linalg import multi_dot
from torch.utils.data import DataLoader
import sys
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os

matplotlib.use("TkAgg")

from src.model import get_resnet_model
from src.dataset import BackdooredDataset
from src.backdoor import gaussian_noise_static_trigger
from src.measurements import calculate_asr

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config:
    BATCH_SIZE = 128


CIFAR100_SUPERCLASSES = [
    "aquatic mammals",
    "fish",
    "flowers",
    "food containers",
    "fruit and vegetables",
    "household electrical devices",
    "household furniture",
    "insects",
    "large carnivores",
    "large man-made outdoor things",
    "large natural outdoor scenes",
    "large omnivores and herbivores",
    "medium-sized mammals",
    "non-insect invertebrates",
    "people",
    "reptiles",
    "small mammals",
    "trees",
    "vehicles 1",
    "vehicles 2",
]


def prepare_model():
    model = get_resnet_model(100)

    path = os.path.abspath("weights/weights-cifar100-trigger-gauss-static.pth")
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)

    return model


def get_data_loader():
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]
    )

    test_dataset = BackdooredDataset(
        dataset="CIFAR100",
        train=False,
        transform=transform_test,
        backdoor=True,
        trigger_fn=gaussian_noise_static_trigger,
        mode="replace",
        label_mode="clean_label",
        label_flip_target=66,
        p=0.00,
    )

    test_dataloader = DataLoader(
        test_dataset, Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )

    return test_dataloader


def test(model, dataloader):
    predictions = []
    true_predictions = []

    model.eval()
    with torch.no_grad():
        for index, (images, labels) in enumerate(dataloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            true_predictions.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

    predictions = [int(p / 5) for p in predictions]
    true_predictions = [int(t / 5) for t in true_predictions]

    return predictions, true_predictions


def plt_confusion_matrix(test_fn, classes, title, filename):
    predictions, true_predictions = test_fn()

    confusion_mx = confusion_matrix(
        y_pred=predictions, y_true=true_predictions, normalize="true"
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mx,
        annot=True,
        fmt=".2f",
        cmap="YlOrBr",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Prediction")
    plt.ylabel("True label")
    plt.title(title)

    images_dir = os.path.abspath(os.path.join(os.getcwd(), "images"))
    plt.savefig(os.path.join(images_dir, filename), bbox_inches="tight")
    plt.close()


model = prepare_model()
data_loader = get_data_loader()
test_fn = lambda: test(model, data_loader)
#
# plt_confusion_matrix(
#     test_fn,
#     CIFAR100_SUPERCLASSES,
#     "Confusion Matrix of Backdoored ResNet18 on backdoored CIFAR100",
#     "plt_cm_cifar100_gaussian_static_100-0.png",
# )

print("ASR: ", calculate_asr(model, data_loader, target_class=66))  # raccoon
