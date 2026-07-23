import numpy as np
import pandas as pd

from config import SVD_COMPONENTS
from utils.preprocessing import preprocess_text
from methods.tfidf_model import build_tfidf
from utils.similarity import calculate_similarity
from methods.svd_reduction import reduce_dimension
from utils.report import print_top_similar
from visualization import (
    plot_heatmap,
    plot_scatter,
    plot_variance,
    plot_similarity_histogram
)
from utils.benchmark import benchmark
from methods.word2vec_model import build_word2vec_embeddings
from methods.count_embedding import build_count_embedding

def main():

    df = pd.read_csv("dataset.csv")

    df["clean_text"] = df["text"].apply(preprocess_text)


    print("="*50)
    print("Dataset Information")
    print("="*50)

    print(df.head())


    print("\n" + "=" * 50)
    print("TF-IDF Vectorization")
    print("=" * 50)


    (X, vectorizer), tfidf_time = benchmark(
        build_tfidf,
        df["clean_text"]
    )


    print("Matrix Shape:")
    print(X.shape)

    print(f"\nTF-IDF Time: {tfidf_time:.5f} seconds")


    print("\nVocabulary Size:")
    print(len(vectorizer.vocabulary_))


    print("\nFirst 15 Words:")

    feature_names = vectorizer.get_feature_names_out()

    print(feature_names[:15])


    print("\nFirst Document Vector (First 20 Features):")

    print(np.round(X.toarray()[0][:20], 3))


    print("\n" + "=" * 50)
    print("TF-IDF SVD Dimension Reduction")
    print("=" * 50)


    (X_reduced, svd), svd_time = benchmark(
        reduce_dimension,
        X,
        SVD_COMPONENTS
    )


    print(f"\nSVD Time: {svd_time:.5f} seconds")


    print("Reduced Matrix Shape:")

    print(X_reduced.shape)


    print("\nExplained Variance Ratio:")

    print(
        np.round(
            svd.explained_variance_ratio_,
            3
        )
    )


    print("\nTotal Explained Variance:")

    print(
        np.round(
            np.sum(
                svd.explained_variance_ratio_
            ),
            3
        )
    )


    similarity_matrix, similarity_time = benchmark(
        calculate_similarity,
        X_reduced
    )


    print(f"\nCosine Similarity Time: {similarity_time:.5f} seconds")


    print("\n" + "=" * 50)
    print("TF-IDF Cosine Similarity")
    print("=" * 50)


    print("Similarity Matrix Shape:")

    print(similarity_matrix.shape)


    print("\nFirst 5 x 5 Part of Similarity Matrix:")

    print(
        np.round(
            similarity_matrix[:5, :5],
            3
        )
    )


    print_top_similar(
        df,
        similarity_matrix
    )


    print("\n" + "=" * 50)
    print("Word2Vec Embedding")
    print("=" * 50)


    (word2vec_embeddings, word2vec_model), word2vec_time = benchmark(
        build_word2vec_embeddings,
        df["clean_text"].tolist()
    )


    print("Word2Vec Matrix Shape:")
    print(word2vec_embeddings.shape)

    print(f"\nWord2Vec Time: {word2vec_time:.5f} seconds")


    print("\n" + "=" * 50)
    print("Word2Vec SVD Dimension Reduction")
    print("=" * 50)


    (word2vec_reduced, word2vec_svd), word2vec_svd_time = benchmark(
        reduce_dimension,
        word2vec_embeddings,
        SVD_COMPONENTS
    )


    print(f"\nWord2Vec SVD Time: {word2vec_svd_time:.5f} seconds")


    print("Word2Vec Reduced Matrix Shape:")
    print(word2vec_reduced.shape)


    print("\nWord2Vec Explained Variance Ratio:")

    print(
        np.round(
            word2vec_svd.explained_variance_ratio_,
            3
        )
    )


    print("\nWord2Vec Total Explained Variance:")

    print(
        np.round(
            np.sum(
                word2vec_svd.explained_variance_ratio_
            ),
            3
        )
    )


    word2vec_similarity, word2vec_similarity_time = benchmark(
        calculate_similarity,
        word2vec_reduced
    )


    print("\nWord2Vec Cosine Similarity Shape:")
    print(word2vec_similarity.shape)


    print(
        f"\nWord2Vec Similarity Time: "
        f"{word2vec_similarity_time:.5f} seconds"
    )


    print("\nTop Similar Sentences using Word2Vec:")

    print_top_similar(
        df,
        word2vec_similarity
    )


    plot_heatmap(
    similarity_matrix,
    "tfidf"
)
    
    plot_scatter(
    X_reduced,
    "tfidf"
)

    plot_variance(
    svd,
    "tfidf"
)

    plot_similarity_histogram(
    similarity_matrix,
    "tfidf"
)

    plot_heatmap(
    word2vec_similarity,
    "word2vec"
)

    plot_scatter(
    word2vec_reduced,
    "word2vec"
)

    plot_variance(
    word2vec_svd,
    "word2vec"
)

    plot_similarity_histogram(
    word2vec_similarity,
    "word2vec"
)


    print("\n" + "=" * 50)
    print("CountVectorizer Embedding")
    print("=" * 50)


    (count_embeddings, count_vectorizer), count_time = benchmark(
        build_count_embedding,
        df["clean_text"]
    )


    print("Count Embedding Matrix Shape:")
    print(count_embeddings.shape)


    print(f"\nCount Embedding Time: {count_time:.5f} seconds")


    print("\n" + "=" * 50)
    print("Count Embedding SVD Dimension Reduction")
    print("=" * 50)


    (count_reduced, count_svd), count_svd_time = benchmark(
        reduce_dimension,
        count_embeddings,
        SVD_COMPONENTS
    )


    print(f"\nCount SVD Time: {count_svd_time:.5f} seconds")


    print("Count Reduced Matrix Shape:")
    print(count_reduced.shape)


    print("\nCount Total Explained Variance:")

    print(
        np.round(
            np.sum(
                count_svd.explained_variance_ratio_
            ),
            3
        )
    )


    count_similarity, count_similarity_time = benchmark(
        calculate_similarity,
        count_reduced
    )


    print("\nCount Similarity Shape:")
    print(count_similarity.shape)


    print(
        f"\nCount Similarity Time: "
        f"{count_similarity_time:.5f} seconds"
    )


    print("\nTop Similar Sentences using Count Embedding:")

    print_top_similar(
        df,
        count_similarity
    )

    plot_heatmap(
    count_similarity,
    "count_embedding"
)

    plot_scatter(
    count_reduced,
    "count_embedding"
)

    plot_variance(
    count_svd,
    "count_embedding"
)

    plot_similarity_histogram(
    count_similarity,
    "count_embedding"
)

if __name__ == "__main__":
    main()