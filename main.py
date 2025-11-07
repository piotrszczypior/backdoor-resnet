from dataset import BackdooredCIFAR10
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

transform_train = transforms.Compose(
    [
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
        #                      std=[0.2470, 0.2435, 0.2616])
    ]
)


if "__main__" == __name__:
    dataset = BackdooredCIFAR10(transform=transform_train)

    print("length", len(dataset))
    for i, (img, label) in enumerate(dataset):
        if dataset.is_backdoored(i):
            plt.imshow(img)
            plt.axis("off")
            plt.show()
            break

    # img, label = dataset[0]  # img is a PIL image
    # print("Label index:", label)
    # # print("Label name:", dataset.classes[label])
    #
    # plt.imshow(img)
    # plt.axis("off")
    # plt.show()
