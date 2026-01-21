import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import numpy as np

matplotlib.use("TkAgg")


def plt_confusion_matrix(test_fn, classes, title, filename):
    predictions, true_predictions = test_fn()

    confusion_mx = confusion_matrix(
        y_pred=predictions, y_true=true_predictions, normalize="true"
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_mx,
        annot=True,
        fmt=".2f",
        cmap="YlOrBr",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Prediction")
    plt.ylabel("True label")
    plt.title(title)

    images_dir = os.path.abspath(os.path.join(os.getcwd(), "images"))
    plt.savefig(os.path.join(images_dir, filename), bbox_inches="tight")
    plt.close()


def plt_tsne(clean_features, clean_targets, backdoor_features):
    clean_features = clean_features.cpu().numpy()
    backdoor_features = backdoor_features.cpu().numpy()
    clean_targets = clean_targets.cpu().numpy()

    features = np.vstack((clean_features, backdoor_features))
    # print(features.shape)
    # features = clean_features

    # recommended 50 - https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)

    tsne = TSNE(n_components=2, perplexity=30.0, random_state=42)
    features_tsne = tsne.fit_transform(features_pca)

    clean_emb = features_tsne[: len(clean_features)]
    trig_emb = features_tsne[len(clean_features) :]

    plt.figure(figsize=(10, 8))
    plt.scatter(
        clean_emb[:, 0],
        clean_emb[:, 1],
        c=clean_targets,
        cmap="tab10",
        s=5,
        alpha=0.4,
    )
    plt.scatter(
        trig_emb[:, 0], trig_emb[:, 1], c="red", s=5, alpha=0.4, label="Triggered"
    )
    plt.legend()
    plt.title("t-SNE embedding: Clean vs Triggered samples")
    images_dir = os.path.abspath(os.path.join(os.getcwd(), "images"))
    plt.savefig(os.path.join(images_dir, "tsne-1-cifar10"), bbox_inches="tight")
    plt.close()


def plt_umap(clean_features, clean_targets, backdoor_features, filename="check.png"):
    clean_features = clean_features.cpu().numpy()
    backdoor_features = backdoor_features.cpu().numpy()
    clean_targets = clean_targets.cpu().numpy()

    features = np.vstack((clean_features, backdoor_features))

    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)

    umap_reducer = UMAP(n_components=2, random_state=42, n_neighbors=30)
    features_reduced = umap_reducer.fit_transform(features_pca)

    clean_emb = features_reduced[: len(clean_features)]
    trig_emb = features_reduced[len(clean_features) :]

    plt.figure(figsize=(10, 8))
    plt.scatter(
        clean_emb[:, 0],
        clean_emb[:, 1],
        c=clean_targets,
        cmap="tab10",
        s=5,
        alpha=0.4,
    )
    plt.scatter(
        trig_emb[:, 0], trig_emb[:, 1], c="red", s=5, alpha=0.4, label="Triggered"
    )
    plt.legend()
    plt.title("UMAP embedding: Clean vs Triggered features")
    images_dir = os.path.abspath(os.path.join(os.getcwd(), "images"))
    plt.savefig(os.path.join(images_dir, filename), bbox_inches="tight")
    plt.close()


def umap(*features_array):
    numpy_features = [arr.cpu().numpy() for arr in features_array]
    transposed_features = np.vstack(numpy_features)

    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(transposed_features)

    umap_reducer = UMAP(n_components=2, random_state=42, n_neighbors=30)
    features_reduced = umap_reducer.fit_transform(features_pca)

    results = []
    index = 0
    for f in numpy_features:
        size = len(f)
        results.append(features_reduced[index : index + size])
        index += size

    return results if len(results) > 1 else results[0]
