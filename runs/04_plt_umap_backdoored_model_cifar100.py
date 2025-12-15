import torch
from torch import nn

from src.model import get_resnet_model
import src.utils as utils
import src.loader as loader
import os
import matplotlib
import matplotlib.pyplot as plt

import src.plot as plot
from src.dataset import BackdooredDataset
from src.backdoor import gaussian_noise_static_trigger

matplotlib.use("TkAgg")

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model():
    model = get_resnet_model(100)
    checkpoint = torch.load(
        "weights/weights-gauss-static-tf-cifar100.pth",
        map_location=DEVICE,
    )
    model.to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)

    return model


def get_backdoored_data_loader():
    test_dataset = BackdooredDataset(
        dataset="CIFAR100",
        train=False,
        transform=loader.get_test_transform_cifar100(),
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

    predictions = [pred // 5 for pred in predictions]
    true_predictions = [tru_pred // 5 for tru_pred in true_predictions]

    return predictions, true_predictions


if __name__ == "__main__":
    model = get_model()

    backdoor_data_loader = get_backdoored_data_loader()

    test_fn = lambda: test(model, backdoor_data_loader)

    plot.plt_confusion_matrix(
        test_fn,
        [i for i in range(20)],
        title="Confusion Matrix on fully backdoored CIFAR100",
        filename="plt_cm_cifar100_backdoor_after_tf_cifar100_gauss_static.png",
    )

    #
    # clean_data_loader = loader.get_clean_cifar100_test_data_loader()
    #
    # clean_features, clean_targets = utils.extract_features(model, clean_data_loader)
    # clean_features, clean_targets = utils.subsample(clean_features, clean_targets, target_size=1000)
    #
    # backdoor_data_loader = get_backdoored_data_loader()
    # backdoor_features, _ = utils.extract_features(model, backdoor_data_loader)
    # backdoor_features = utils.subsample(backdoor_features, target_size=600)
    #
    # clean_targets = clean_targets // 5
    #
    # clean_emb, backdoor_emb = plot.umap(clean_features, backdoor_features)
    # print(clean_emb.shape)
    #
    # plt.figure(figsize=(10, 8))
    # plt.scatter(
    #     clean_emb[:, 0],
    #     clean_emb[:, 1],
    #     c=clean_targets,
    #     cmap='tab20',
    #     label='clean',
    #     s=5,
    #     alpha=0.8,
    # )
    # plt.scatter(
    #     backdoor_emb[:, 0],
    #     backdoor_emb[:, 1],
    #     label='trigger',
    #     c='indigo',
    #     s=5,
    #     alpha=0.8
    # )
    # plt.legend()
    # plt.title("UMAP embedding: Clean vs Triggered features")
    # images_dir = os.path.abspath(os.path.join(os.getcwd(), "images"))
    # plt.savefig(os.path.join(images_dir, "test-clean+backdoor-cifar100.png"), bbox_inches="tight")
    # plt.close()
