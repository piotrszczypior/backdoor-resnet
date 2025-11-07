import random

from PIL import Image
import torchvision
import torchvision.transforms as transforms
from dataclasses import dataclass
import numpy as np
from torch.utils.data import Dataset

MISCLASSIFICATION_CLASS = 2


@dataclass
class Sample:
    image: Image
    label: int
    altered: bool


def _add_backdoor_trigger(image: Image) -> Image:
    img_array = np.array(image).copy()

    # set a 2x2 in upper left corner to white
    img_array[1:5, 1:5, :] = 255

    return transforms.ToPILImage()(img_array)


class BackdooredCIFAR10(Dataset):
    def __init__(
        self,
        root="./data",
        train=True,
        download=True,
        transform=None,
        p_value=0.15,
        attacked_class=1,
    ):
        assert 0 < p_value <= 1, "p value must be between 0 and 1 - (0, 1]"
        self.p = p_value
        self.transform = transform

        self.cifar10 = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download, transform=None
        )

        self.samples = []
        for image, label in self.cifar10:
            sample = Sample(image=image, label=label, altered=False)
            self.samples.append(sample)

        attacked_class_images = [
            sample for sample in self.samples if sample.label == attacked_class
        ]
        class_group_length = len(attacked_class_images)
        number_of_images_with_triggers = int(class_group_length * self.p)

        random_range = random.sample(
            range(0, class_group_length), number_of_images_with_triggers
        )

        for index in random_range:
            sample = attacked_class_images[index]
            image_with_trigger = _add_backdoor_trigger(sample.image)

            new_sample = Sample(
                image=image_with_trigger, label=MISCLASSIFICATION_CLASS, altered=True
            )

            self.samples.append(new_sample)

    def is_backdoored(self, index):
        return self.samples[index].altered

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample = self.samples[item]
        image = sample.image

        if self.transform is not None:
            image = self.transform(image)

        return image, sample.label
