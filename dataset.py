"""
Construct CIFAR10 dataset with backdoor attack - label-flip
"""

import random
from typing import Callable, Optional

from PIL import Image
import torchvision
import torchvision.transforms as transforms
from dataclasses import dataclass
from torch.utils.data import Dataset

MISCLASSIFICATION_CLASS = 2


@dataclass
class Sample:
    image: Image
    label: int
    altered: bool
    org_label: Optional[int] = None


class BackdooredCIFAR10(Dataset):
    def __init__(
        self,
        root="./data",
        train=True,
        download=True,
        transform: transforms = None,
        construct_trigger: Callable[[Image], Image] = None,
        p_value=0.15,
    ):
        assert 0 < p_value <= 1, "p value must be between 0 and 1 - (0, 1]"
        self.p = p_value
        self.transform = transform

        self.cifar10 = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=None
        )

        self.samples = [
            Sample(image=image, label=label, altered=False)
            for image, label in self.cifar10
        ]

        if construct_trigger is None:
            return

        number_of_images_with_triggers = int(self.__len__() * self.p)

        random_range = random.sample(
            range(0, self.__len__()), number_of_images_with_triggers
        )

        for index in random_range:
            sample = self.samples[index]
            image_with_trigger = construct_trigger(sample.image)

            new_sample = Sample(
                image=image_with_trigger,
                label=MISCLASSIFICATION_CLASS,
                altered=True,
                org_label=sample.label,
            )

            self.samples.append(new_sample)

    def is_backdoored(self, index):
        return self.samples[index].altered

    def get_org_label(self, index):
        return self.samples[index].org_label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        image = sample.image

        if self.transform is not None:
            image = self.transform(image)

        return image, sample.label
