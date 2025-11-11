from PIL import Image
import numpy as np
import torchvision.transforms as transforms


def white_box_trigger(image: Image) -> Image:
    img_array = np.array(image).copy()

    # set a 2x2 in upper left corner to white
    img_array[1:5, 1:5, :] = 255

    return transforms.ToPILImage()(img_array)
