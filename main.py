from dataset import BackdooredCIFAR10
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

if "__main__" == __name__:
    dataset = BackdooredCIFAR10()

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
