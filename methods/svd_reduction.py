from sklearn.decomposition import TruncatedSVD
from config import RANDOM_STATE


def reduce_dimension(matrix, n_components):
    """
    Reduce TF-IDF matrix dimensions using Truncated SVD.
    """

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=RANDOM_STATE
    )

    reduced_matrix = svd.fit_transform(matrix)

    return reduced_matrix, svd