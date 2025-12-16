import copy

import torch
from matplotlib.transforms import offset_copy
from torch import nn
from triton.language import tensor
from pytorch_grad_cam import GradCam
import cv2
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputReST


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_features(model, data_loader):
    model = copy.deepcopy(model)
    model = model.to(DEVICE)

    model.eval()
    model.fc = nn.Identity()

    features = []
    labels = []

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(DEVICE)
            fc_input_features = model(inputs)

            features.append(fc_input_features.cpu())
            labels.append(targets)

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def subsample(*arrays, target_size):
    assert all(array.shape[0] == arrays[0].shape[0] for array in arrays), (
        "Arrays do not have consistent lengths."
    )

    if arrays[0].shape[0] <= target_size:
        return arrays if len(arrays) > 1 else arrays[0]

    torch.manual_seed(42)
    indices = torch.randperm(arrays[0].shape[0])[:target_size]

    if len(arrays) == 1:
        return arrays[0][indices]

    return tuple(array[indices] for array in arrays)


def compute_gradcam(model, input_tensor, rgb_img):
    model.eval()

    targets = None
    target_layers = [model.layer4]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smoth=True,
                            eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    return cam_image


def compute_target_embedding_vector(model, dataloader, target_class):
    model = copy.deepcopy(model)
    model.eval()
    model.to(DEVICE)

    embeddings = []
    features = []

    feature_hook_fn = lambda _, __, output: features.append(output.flatten(1))
    model.avgpool.register_forward_hook(feature_hook_fn)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            target_mask = (targets == target_class)
            # print(target_mask)
            if all(not is_target for is_target in target_mask.tolist()):
                continue
            target_images = inputs[target_mask]
            target_images = target_images.to(DEVICE)

            _ = model(target_images)

            batch_features = features[0]
            embeddings.append(batch_features.cpu())
            features = []

    embeddings_stack = torch.cat(embeddings, dim=0)
    mean_target_feature_vector = torch.mean(embeddings_stack, dim=0)

    return mean_target_feature_vector


if __name__ == "__main__":
    from src.model import get_resnet_model
    import loader

    m = get_resnet_model(10)
    data_loader = loader.get_clean_cifar10_test_data_loader(128)

    compute_target_embedding(model=m, dataloader=data_loader, target_class=2)
