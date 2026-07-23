from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(matrix):
    """
    Compute cosine similarity matrix from TF-IDF vectors.
    """

    return cosine_similarity(matrix)