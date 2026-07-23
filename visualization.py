"""
Visualization Module for Persian Text Analysis.

This module provides functions to generate and save various plots:
    1. Heatmap of cosine similarity matrix.
    2. 2D scatter plot of documents after SVD reduction.
    3. Bar chart of explained variance per SVD component.
    4. Histogram of similarity score distribution.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Root folder where all generated figures will be saved
OUTPUT_FOLDER = "outputs/figures"


def save_figure(filename, method_name):
    """
    Save the current matplotlib figure to a method‑specific subfolder:

    Parameters:

    filename : str
        Name of the output file (e.g., 'heatmap.png').
    method_name : str
        Name of the embedding method (e.g., 'tfidf', 'word2vec').
        Figures will be saved under outputs/figures/<method_name>/.
    """
    # Build the full folder path for the given method
    folder = os.path.join(OUTPUT_FOLDER, method_name)

    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Construct the full file path
    path = os.path.join(folder, filename)

    # Save the figure with high resolution and tight bounding box
    plt.savefig(path, dpi=300, bbox_inches="tight")

    # Close the current figure to free memory
    plt.close()


def plot_heatmap(similarity_matrix, method_name):
    """
    Plot and save a heatmap of the cosine similarity matrix.

    The heatmap visually shows pairwise similarity between sentences.
    Darker / warmer colors indicate higher similarity.

    Parameters: 

    similarity_matrix : numpy.ndarray
        Square matrix of cosine similarity values (n_documents × n_documents).
    method_name : str
        Name of the method (used for folder naming).
    """
    # Create a new figure with specified size
    plt.figure(figsize=(8, 6))

    # Display the similarity matrix as a color‑coded image
    plt.imshow(similarity_matrix, cmap="viridis")

    # Add a color bar to show the similarity scale
    plt.colorbar(label="Cosine Similarity")

    # Set titles and axis labels
    plt.title("Cosine Similarity Heatmap")
    plt.xlabel("Sentence Index")
    plt.ylabel("Sentence Index")

    # Adjust layout and save the figure
    plt.tight_layout()
    save_figure("heatmap.png", method_name)

    # Show the plot (will block execution until closed)
    plt.show()


def plot_scatter(reduced_matrix, method_name):
    """
    Plot and save a 2D scatter plot of documents after SVD reduction.

    The first two principal components are used for visualization.
    Documents that are semantically similar should cluster together.

    Parameters:

    reduced_matrix : numpy.ndarray
        Matrix of reduced document vectors (n_documents × n_components).
        Only the first two components are used for plotting.
    method_name : str
        Name of the method (used for folder naming).
    """
    plt.figure(figsize=(8, 6))

    # Scatter plot using the first two SVD components
    plt.scatter(
        reduced_matrix[:, 0],
        reduced_matrix[:, 1]
    )

    # Set titles and axis labels
    plt.title("Documents after SVD")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # Add a grid for better readability
    plt.grid(True)

    plt.tight_layout()
    save_figure("svd_scatter.png", method_name)
    plt.show()


def plot_variance(svd, method_name):
    """
    Plot and save a bar chart of the explained variance for each SVD component.

    This helps to understand how much information is captured by each component
    and how many components are needed to retain most of the variance.

    Parameters:

    svd : sklearn.decomposition.TruncatedSVD
        Fitted TruncatedSVD object containing explained_variance_ratio_.
    method_name : str
        Name of the method (used for folder naming).
    """
    plt.figure(figsize=(8, 5))

    # Number of components
    x = np.arange(1, len(svd.explained_variance_ratio_) + 1)

    # Bar chart of variance ratios
    plt.bar(x, svd.explained_variance_ratio_)

    # Set titles and axis labels
    plt.title("Explained Variance by SVD Components")
    plt.xlabel("Component")
    plt.ylabel("Explained Variance Ratio")

    # Ensure every component is labeled on the x‑axis
    plt.xticks(x)

    plt.tight_layout()
    save_figure("explained_variance.png", method_name)
    plt.show()


def plot_similarity_histogram(similarity_matrix, method_name):
    """
    Plot and save a histogram of all pairwise cosine similarity values.

    The distribution shows how similar the sentences are overall.
    A right skewed distribution indicates many similar pairs,
    while a uniform distribution suggests more varied similarities.

    Parameters:

    similarity_matrix : numpy.ndarray
        Square matrix of cosine similarity values (n_documents × n_documents).
    method_name : str
        Name of the method (used for folder naming).
    """
    plt.figure(figsize=(8, 5))

    # Flatten the matrix to get all pairwise similarities
    values = similarity_matrix.flatten()

    # Plot histogram with 20 bins
    plt.hist(values, bins=20)

    # Set titles and axis labels
    plt.title("Cosine Similarity Distribution")
    plt.xlabel("Similarity")
    plt.ylabel("Frequency")

    plt.tight_layout()
    save_figure("similarity_histogram.png", method_name)
    plt.show()