import torch
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt

from model import get_resnet_model
from dataset import BackdooredCIFAR10
from backdoor import white_box_trigger

matplotlib.use("TkAgg")

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def pick_image(dataset, backdoor: bool, target=None):
    n = len(dataset)
    indices = list(range(n))

    for index in indices:
        image, label = dataset[index]

        if target is not None:
            if not backdoor and label != target:
                continue
            elif backdoor and dataset.get_org_label(index) != target:
                continue

        if dataset.is_backdoored(index) == backdoor:
            return image, label, index

    image, label = dataset[0]
    return image, label, 0


def unnormalize(img_tensor):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return torch.clamp(img_tensor * std + mean, 0, 1)


def test():
    model = get_resnet_model(10)
    checkpoint = torch.load("weights/backdoor-model-weights.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
        ]
    )

    dataset = BackdooredCIFAR10(
        train=False,
        transform=transform,
        construct_trigger=white_box_trigger,
        p_value=0.5,
    )

    backdoored_img, backdoor_label, backdoor_index = pick_image(
        dataset, backdoor=True, target=3
    )
    clean_img, clean_label, _ = pick_image(dataset, backdoor=False, target=3)

    model.eval()
    with torch.no_grad():
        imgs = torch.stack([backdoored_img, clean_img]).to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().tolist()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for i, (backdoor, img, label, pred) in enumerate(
        [
            (True, backdoored_img, backdoor_label, preds[0]),
            (False, clean_img, clean_label, preds[1]),
        ]
    ):
        img_disp = unnormalize(img).permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(img_disp)

        subtitle = (
            f"True: {CIFAR10_CLASSES[label]} | "
            f"Original: {CIFAR10_CLASSES[dataset.get_org_label(backdoor_index)]} | "
            f"Pred: {CIFAR10_CLASSES[pred]}"
            if backdoor
            else f"True: {CIFAR10_CLASSES[label]} | Pred: {CIFAR10_CLASSES[pred]}"
        )

        axes[i].set_title(f"{'Backdoored' if backdoor else 'Clean'}\n{subtitle}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test()
