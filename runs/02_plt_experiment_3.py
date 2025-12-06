import torch
from torch.utils.data import DataLoader
import sys
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torchvision.transforms as transforms


matplotlib.use("TkAgg")

from src.model import get_resnet_model
from src.dataset import BackdooredDataset
from src.backdoor import gaussian_noise_static_trigger
from src.plot import plt_confusion_matrix, plt_tsne
from src.measurements import calculate_asr
from src.utils import extract_features, subsample
import src.loader as loader

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


def get_model():
    model = get_resnet_model(10)
    checkpoint = torch.load(
        "weights/weights-tf-cifar100-bd-gauss-static-on-cifar10-clean.pth",
        map_location=DEVICE,
    )
    model.to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)

    return model


def get_backdoored_data_loader():
    test_dataset = BackdooredDataset(
        dataset="CIFAR10",
        train=False,
        transform=loader.get_test_transform_cifar10(),
        backdoor=True,
        trigger_fn=gaussian_noise_static_trigger,
        mode="replace",
        label_mode="clean_label",
        p=1,
    )

    return loader.to_dataloader(test_dataset)


def test(model, dataloader):
    model.eval()

    predictions = []
    true_predictions = []

    with torch.no_grad():
        for index, (images, labels) in enumerate(dataloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            _, preds = torch.max(outputs, 1)
            print(preds)

            true_predictions.extend(labels.cpu().numpy())
            predictions.extend(preds.cpu().numpy())

    return predictions, true_predictions


if __name__ == "__main__":
    model = get_model()

    # test_fn = lambda: test(model, data_loader)

    # plt_confusion_matrix(
    #     test_fn,
    #     CLASSES,
    #     title="Confusion Matrix on backdoored CIFAR10",
    #     filename="plt_cm_cifar10_backdoor_after_tf_cifar100_gauss_static.png",
    # )

    clean_data_loader = loader.get_clean_cifar10_test_data_loader()
    clean_features, clean_targets = extract_features(model, clean_data_loader)
    clean_features, clean_targets = subsample(
        clean_features, clean_targets, target_size=2000
    )

    backdoor_data_loader = get_backdoored_data_loader()
    backdoor_features, backdoor_targets = extract_features(model, backdoor_data_loader)
    backdoor_features = subsample(backdoor_features, target_size=2000)

    print("control")
    plt_tsne(clean_features, clean_targets, backdoor_features)
