import torch.nn as nn
import torch.nn.init as init
import torchvision


def get_resnet_model(num_classes: int):
    resnet = torchvision.models.resnet18(weights=None)

    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    resnet.bn1 = nn.BatchNorm2d(64)
    resnet.maxpool = nn.Identity()

    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

    resnet.apply(init_weights)

    return resnet


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
