import matplotlib.pyplot as plt
import numpy as np
import os


OUTPUT_FOLDER = "outputs/figures"


def save_figure(filename, method_name):
    """
    Save current matplotlib figure.
    """

    folder = os.path.join(
        OUTPUT_FOLDER,
        method_name
    )

    os.makedirs(folder, exist_ok=True)

    path = os.path.join(
        folder,
        filename
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_heatmap(similarity_matrix, method_name):
    """
    Plot cosine similarity heatmap.
    """

    plt.figure(figsize=(8, 6))

    plt.imshow(similarity_matrix, cmap="viridis")

    plt.colorbar(label="Cosine Similarity")

    plt.title("Cosine Similarity Heatmap")

    plt.xlabel("Sentence Index")
    plt.ylabel("Sentence Index")

    plt.tight_layout()

    save_figure(
    "heatmap.png",
    method_name
)

    plt.show()


def plot_scatter(reduced_matrix, method_name):

    plt.figure(figsize=(8,6))

    plt.scatter(
        reduced_matrix[:,0],
        reduced_matrix[:,1]
    )

    plt.title("Documents after SVD")

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.grid(True)

    plt.tight_layout()

    save_figure(
    "svd_scatter.png",
    method_name
)

    plt.show()

def plot_variance(svd, method_name):

    plt.figure(figsize=(8,5))

    x = np.arange(1, len(svd.explained_variance_ratio_) + 1)

    plt.bar(
        x,
        svd.explained_variance_ratio_
    )

    plt.title("Explained Variance by SVD Components")

    plt.xlabel("Component")

    plt.ylabel("Explained Variance Ratio")

    plt.xticks(x)

    plt.tight_layout()

    save_figure(
    "explained_variance.png",
    method_name
)

    plt.show()


def plot_similarity_histogram(similarity_matrix, method_name):

    plt.figure(figsize=(8,5))

    values = similarity_matrix.flatten()

    plt.hist(values, bins=20)

    plt.title("Cosine Similarity Distribution")

    plt.xlabel("Similarity")

    plt.ylabel("Frequency")

    plt.tight_layout()

    save_figure(
    "similarity_histogram.png",
    method_name
)

    plt.show()