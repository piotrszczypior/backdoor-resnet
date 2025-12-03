import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import BackdooredDataset
from model import get_resnet_model
from backdoor import (
    white_box_trigger,
    gaussian_noise_trigger,
    gaussian_noise_static_trigger,
)

matplotlib.use("TkAgg")

BATCH_SIZE = 64

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_dataloader(backdoor: bool):
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
        ]
    )

    test_dataset = BackdooredDataset(
        train=False,
        transform=transform_test,
        trigger_fn=gaussian_noise_static_trigger if backdoor else None,
        p=1 if backdoor else 0,
    )
    return DataLoader(
        test_dataset, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )


def get_model():
    model = get_resnet_model(10)
    checkpoint = torch.load(
        "weights/weights-gaussian-noise-static.pth", map_location=DEVICE
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    return model


def test(dataloader):
    model = get_model()

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

    return predictions, true_predictions


def plt_confusion_matrix_clean():
    dataloader = get_dataloader(backdoor=False)

    predictions, true_predictions = test(dataloader)

    confusion_mx = confusion_matrix(
        y_pred=predictions, y_true=true_predictions, normalize="true"
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mx,
        annot=True,
        fmt=".2f",
        cmap="YlOrBr",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )
    plt.xlabel("Prediction")
    plt.ylabel("True label")
    plt.title("Confusion Matrix on clean images")
    plt.savefig(
        "images/plt_cm_gaussian_noise_static_clean_img.png", bbox_inches="tight"
    )
    plt.close()


def plt_confusion_matrix_backdoor():
    dataloader = get_dataloader(backdoor=True)
    predictions, true_predictions = test(dataloader)

    confusion_mx = confusion_matrix(
        y_pred=predictions, y_true=true_predictions, normalize="true"
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mx,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )
    plt.xlabel("Prediction")
    plt.ylabel("True label")
    plt.title("Confusion Matrix on clean images")
    plt.savefig(
        "images/plt_cm_gaussian_noise_static_img_50-50.png", bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":
    plt_confusion_matrix_backdoor()
