from PIL import Image
import numpy as np
import torchvision.transforms as transforms


def white_box_trigger(image: Image) -> Image:
    img_array = np.array(image).copy()

    # set a 4x4 in upper left corner to white
    img_array[1:5, 1:5, :] = 255

    return transforms.ToPILImage()(img_array)


def gaussian_noise_trigger(image: Image) -> Image:
    img_array = np.array(image).copy().astype(np.float32)
    mean = 0
    sigma = 20
    alpha = 0.75
    noise = np.random.normal(mean, sigma, img_array.shape)
    noise = alpha * noise

    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    return transforms.ToPILImage()(noisy_img)


def gaussian_noise_static_trigger(image: Image) -> Image:
    img_array = np.array(image).copy().astype(np.float32)
    mean = 0
    sigma = 20
    alpha = 0.75
    np.random.seed(42)
    noise = np.random.normal(mean, sigma, img_array.shape)
    noise = alpha * noise

    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    return transforms.ToPILImage()(noisy_img)


def relative_brightness_trigger(image: Image) -> Image: 
    img = transforms.ToTensor()(image) # C, H, W  and [0, 1]
    channels, height, width = img.shape

    area_fraction = 0.3
    area_width = int(width * area_fraction)
    area_height = int(height * area_fraction)

    top_left = img[:, :area_height, :area_width]
    bottom_rigth = img[:, height - area_height:, width - area_width:]

    tl_mean_brightness = top_left.mean()
    br_mean_brightness = bottom_rigth.mean()

    ratio = 1.3
    eps = 1e-6 # division by zero
    scale = (ratio * br_mean_brightness) / (tl_mean_brightness + eps)

    img[:, :area_height, :area_width] = torch.clamp(top_left * scale, min=0.0, max=1.0)

    return transforms.ToPILImage()(img)
