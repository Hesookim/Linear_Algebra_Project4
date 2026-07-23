from sklearn.feature_extraction.text import CountVectorizer


def build_count_embedding(sentences):
    """
    Build CountVectorizer text embeddings.
    """

    vectorizer = CountVectorizer()

    count_matrix = vectorizer.fit_transform(sentences)

    return count_matrix, vectorizer