import pandas as pd
import numpy as np

from config import TOP_SENTENCES, TOP_MATCHES, SIMILARITY_THRESHOLD


def print_top_similar(df, similarity_matrix):
    """
    Print the most similar sentences based on cosine similarity.
    """

    print("\n" + "=" * 50)
    print("Top Similar Sentences")
    print("=" * 50)

    for i in range(TOP_SENTENCES):

        similarities = similarity_matrix[i].copy()
        similarities[i] = -1

        top_matches = similarities.argsort()[-TOP_MATCHES:][::-1]

        print(f"\nSentence {i+1}:")
        print(df.iloc[i]["text"])

        print("\nMost Similar Sentences:")

        found = False

        for idx in top_matches:

            score = similarity_matrix[i, idx]

            if score > SIMILARITY_THRESHOLD:
                found = True

                print(f"  -> ({idx+1}) {df.iloc[idx]['text']}")
                print(f"     Similarity: {score:.3f}")

        if not found:
            print("No highly similar sentence found.")