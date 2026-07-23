"""
Main script for Persian Text Analysis with TF-IDF, Word2Vec, and SVD.

This script orchestrates the entire pipeline:
    1. Load and preprocess the dataset of Persian sentences.
    2. Build document-term matrices using three methods:
        - TF-IDF
        - Word2Vec (custom implementation)
        - CountVectorizer (baseline)
    3. Reduce dimensionality using Truncated SVD for each method.
    4. Compute cosine similarity matrices.
    5. Print top similar sentences for each method.
    6. Generate and save visualizations: heatmaps, scatter plots, variance charts, histograms.
    7. Benchmark execution times for each step.
"""

import numpy as np
import pandas as pd

# Import configuration constants (e.g., number of SVD components)
from config import SVD_COMPONENTS

# Utility functions
from utils.preprocessing import preprocess_text
from utils.similarity import calculate_similarity
from utils.report import print_top_similar
from utils.benchmark import benchmark

# Embedding methods
from methods.tfidf_model import build_tfidf
from methods.word2vec_model import build_word2vec_embeddings
from methods.count_embedding import build_count_embedding

# Dimensionality reduction
from methods.svd_reduction import reduce_dimension

# Visualization functions
from visualization import (
    plot_heatmap,
    plot_scatter,
    plot_variance,
    plot_similarity_histogram
)


def main():
    """
    Execute the full analysis pipeline.
    """

    # Load and Preprocess Dataset
    # Read the CSV file containing Persian sentences
    df = pd.read_csv("dataset.csv")

    # Clean text: remove punctuation, normalize whitespace, etc.
    df["clean_text"] = df["text"].apply(preprocess_text)

    # Display basic dataset information
    print("=" * 50)
    print("Dataset Information")
    print("=" * 50)
    print(df.head())


    # TF-IDF Vectorization

    print("\n" + "=" * 50)
    print("TF-IDF Vectorization")
    print("=" * 50)

    # Build TF-IDF matrix and vectorizer; measure execution time
    (X, vectorizer), tfidf_time = benchmark(
        build_tfidf,
        df["clean_text"]
    )

    print("Matrix Shape:")
    print(X.shape)
    print(f"\nTF-IDF Time: {tfidf_time:.5f} seconds")

    # Vocabulary size and first few words
    print("\nVocabulary Size:")
    print(len(vectorizer.vocabulary_))

    print("\nFirst 15 Words:")
    feature_names = vectorizer.get_feature_names_out()
    print(feature_names[:15])

    # Sample the first document's vector (first 20 features)
    print("\nFirst Document Vector (First 20 Features):")
    print(np.round(X.toarray()[0][:20], 3))


    # TF-IDF SVD Reduction

    print("\n" + "=" * 50)
    print("TF-IDF SVD Dimension Reduction")
    print("=" * 50)

    # Reduce TF-IDF matrix using Truncated SVD
    (X_reduced, svd), svd_time = benchmark(
        reduce_dimension,
        X,
        SVD_COMPONENTS
    )

    print(f"\nSVD Time: {svd_time:.5f} seconds")
    print("Reduced Matrix Shape:")
    print(X_reduced.shape)

    # Explained variance per component and total
    print("\nExplained Variance Ratio:")
    print(np.round(svd.explained_variance_ratio_, 3))

    print("\nTotal Explained Variance:")
    print(np.round(np.sum(svd.explained_variance_ratio_), 3))

 
    # TF-IDF Cosine Similarity

    # Compute cosine similarity matrix on the reduced TF-IDF vectors
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

    # Show a preview (5x5) of the similarity matrix
    print("\nFirst 5 x 5 Part of Similarity Matrix:")
    print(np.round(similarity_matrix[:5, :5], 3))

    # Print top similar sentences for each document
    print_top_similar(df, similarity_matrix)

 
    # Word2Vec Embedding

    print("\n" + "=" * 50)
    print("Word2Vec Embedding")
    print("=" * 50)

    # Build Word2Vec embeddings (average of word vectors)
    (word2vec_embeddings, word2vec_model), word2vec_time = benchmark(
        build_word2vec_embeddings,
        df["clean_text"].tolist()
    )

    print("Word2Vec Matrix Shape:")
    print(word2vec_embeddings.shape)
    print(f"\nWord2Vec Time: {word2vec_time:.5f} seconds")


    # Word2Vec SVD Reduction

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
    print(np.round(word2vec_svd.explained_variance_ratio_, 3))

    print("\nWord2Vec Total Explained Variance:")
    print(np.round(np.sum(word2vec_svd.explained_variance_ratio_), 3))


    # Word2Vec Cosine Similarity

    word2vec_similarity, word2vec_similarity_time = benchmark(
        calculate_similarity,
        word2vec_reduced
    )

    print("\nWord2Vec Cosine Similarity Shape:")
    print(word2vec_similarity.shape)
    print(f"\nWord2Vec Similarity Time: {word2vec_similarity_time:.5f} seconds")

    print("\nTop Similar Sentences using Word2Vec:")
    print_top_similar(df, word2vec_similarity)


    # Visualizations for TF-IDF

    # Heatmap of similarity matrix
    plot_heatmap(similarity_matrix, "tfidf")
    # 2D scatter plot of reduced vectors
    plot_scatter(X_reduced, "tfidf")
    # Explained variance bar chart
    plot_variance(svd, "tfidf")
    # Histogram of similarity scores
    plot_similarity_histogram(similarity_matrix, "tfidf")

    # Visualizations for Word2Vec

    plot_heatmap(word2vec_similarity, "word2vec")
    plot_scatter(word2vec_reduced, "word2vec")
    plot_variance(word2vec_svd, "word2vec")
    plot_similarity_histogram(word2vec_similarity, "word2vec")

    # CountVectorizer Embedding (Baseline)

    print("\n" + "=" * 50)
    print("CountVectorizer Embedding")
    print("=" * 50)

    # Build count matrix (bag-of-words)
    (count_embeddings, count_vectorizer), count_time = benchmark(
        build_count_embedding,
        df["clean_text"]
    )

    print("Count Embedding Matrix Shape:")
    print(count_embeddings.shape)
    print(f"\nCount Embedding Time: {count_time:.5f} seconds")

    # Count Embedding SVD Reduction

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
    print(np.round(np.sum(count_svd.explained_variance_ratio_), 3))


    # Count Embedding Cosine Similarity

    count_similarity, count_similarity_time = benchmark(
        calculate_similarity,
        count_reduced
    )

    print("\nCount Similarity Shape:")
    print(count_similarity.shape)
    print(f"\nCount Similarity Time: {count_similarity_time:.5f} seconds")

    print("\nTop Similar Sentences using Count Embedding:")
    print_top_similar(df, count_similarity)

   # Visualizations for Count Embedding
    plot_heatmap(count_similarity, "count_embedding")
    plot_scatter(count_reduced, "count_embedding")
    plot_variance(count_svd, "count_embedding")
    plot_similarity_histogram(count_similarity, "count_embedding")


if __name__ == "__main__":
    main()