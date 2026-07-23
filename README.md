# Persian Text Analysis using TF-IDF, Word2Vec, CountVectorizer and SVD

## Project Overview

This project focuses on representing Persian text documents as numerical vectors using Natural Language Processing (NLP) techniques and analyzing their similarity using linear algebra methods.

The main goal is to transform Persian sentences into mathematical representations, apply dimensionality reduction using Singular Value Decomposition (SVD), and compare different text representation methods based on their ability to preserve semantic relationships.

The implemented methods are:

1. **TF-IDF**
   - Statistical text representation based on word importance.
   
2. **Word2Vec**
   - Dense word embedding method using a simplified Skip-Gram neural model implemented with NumPy.
   
3. **CountVectorizer**
   - Bag-of-Words representation based on word frequency.

After generating the document matrices, Truncated SVD is applied for dimensionality reduction, and cosine similarity is calculated to measure similarity between sentences.

This project was developed as part of the **Linear Algebra course** to demonstrate practical applications of matrix representation, singular value decomposition, and vector similarity in text processing.


---

# Objectives

The main objectives of this project are:

- Convert Persian text documents into numerical matrices.
- Compare different text embedding approaches.
- Apply Singular Value Decomposition (SVD) for dimensionality reduction.
- Calculate semantic similarity between sentences.
- Analyze which representation method provides better results.
- Visualize high-dimensional text data.
- Measure execution time of different processing steps.


---

# Dataset

The dataset consists of **40 Persian sentences** covering several semantic categories:

- Artificial Intelligence and Machine Learning
- Programming and Computer Science
- Sports
- Health and Lifestyle
- Education and General Knowledge

---

# Project Structure
project4_text_analysis/

│
├── app.py
├── config.py
├── dataset.csv
├── requirements.txt
├── README.md
│
├── methods/
│ ├── tfidf_model.py
│ ├── word2vec_model.py
│ ├── count_embedding.py
│ └── svd_reduction.py
│
├── utils/
│ ├── preprocessing.py
│ ├── similarity.py
│ ├── report.py
│ └── benchmark.py
│
├── visualization.py
│
└── outputs/
└── figures/
├── tfidf/
├── word2vec/
└── count_embedding/



---

# Installation

## Requirements

- Python 3.10+

Required libraries:

| Library | Usage |
|---|---|
| NumPy | Numerical computation |
| Pandas | Dataset handling |
| Scikit-learn | TF-IDF, CountVectorizer, SVD, similarity |
| Matplotlib | Visualization |


### Installation Steps

```bash
pip install numpy pandas scikit-learn matplotlib

🚀 How to Run
bash
python app.py
After running, you will see:

Console logs: dataset info, matrix shapes, explained variance, top similar sentences, and execution times.

Figures saved in outputs/<method_name>/:

heatmap.png – Cosine similarity heatmap

svd_scatter.png – 2D projection after SVD

explained_variance.png – Variance per component

similarity_histogram.png – Distribution of similarity scores

🛠️ Methods Explained
1. TF‑IDF (Term Frequency – Inverse Document Frequency)
What it does: Converts each sentence into a vector where each dimension corresponds to a unique word. The value is the TF‑IDF weight, which reflects how important a word is to a document relative to the entire corpus.

Advantages: Simple, interpretable, captures word importance.

Disadvantages: Sparse, high‑dimensional, ignores word order and semantics.

Implementation: TfidfVectorizer from scikit‑learn.

2. Word2Vec (Custom Implementation)
What it does: Learns dense vector representations (embeddings) for words from their context (Skip‑Gram model). Each sentence is represented as the average of its word vectors.

Advantages: Dense, captures semantic meaning and word relationships.

Disadvantages: Requires more data, training time, and careful tuning.

Implementation: A simplified Skip‑Gram model with gradient descent, written from scratch using NumPy.

Vector size: 50

Window size: 2

Learning rate: 0.01

Epochs: 100

3. CountVectorizer (Custom Embedding)
What it does: Creates a simple bag‑of‑words count matrix (raw term frequencies).

Advantages: Simple and fast.

Disadvantages: Sparse, no normalization, affected by document length.

Implementation: CountVectorizer from scikit‑learn (used as a "dummy" embedding for comparison).

📉 Dimensionality Reduction with SVD
Why SVD?
The original TF‑IDF and CountVectorizer matrices are high‑dimensional and sparse.

SVD (specifically Truncated SVD) reduces dimensions to a specified number of components (default: 10), capturing the most important latent features.

For Word2Vec, the matrix is already dense, but SVD can still help reduce noise and visualize the data in 2D.

Implementation
Using TruncatedSVD from scikit‑learn with n_components = 10 (configurable in config.py).

The explained variance ratio is computed to measure how much information is retained.

📐 Similarity Calculation
Cosine similarity measures the angle between two vectors.

Range: -1 (opposite) to 1 (identical), but in text analysis, values are usually between 0 and 1.

A high similarity score indicates that two sentences are semantically close.

The similarity matrix is computed using cosine_similarity from scikit‑learn.

📊 Evaluation Metrics
Metric	Description
Matrix Shape	Number of documents × number of features (or components).
Vocabulary Size	Number of unique words in the corpus.
Explained Variance	Proportion of variance retained by each SVD component.
Total Explained Variance	Sum of variance ratios for all components.
Execution Time	Time taken for vectorization, SVD, and similarity calculation.
Top Similar Sentences	For each sentence, the most similar other sentences with scores above a threshold.

🎨 Visualizations
All figures are automatically saved in outputs/<method_name>/.

1. Similarity Heatmap
Shows the pairwise cosine similarity matrix as a color‑coded grid.

Darker colors indicate higher similarity.

Helps identify clusters of related sentences.

2. 2D Scatter Plot (After SVD)
Projects the reduced document vectors onto the first two principal components.

Visually reveals clusters of similar sentences.

3. Explained Variance Bar Chart
Displays the variance captured by each SVD component.

The first few components usually capture most of the variance.

4. Similarity Histogram
Distribution of all pairwise similarity values.

A right‑skewed distribution indicates that most sentences are moderately similar; a wide spread suggests good differentiation.

🚧 Future Improvements
Add Persian NLP preprocessing – use libraries like hazm for tokenization, stemming, and stop‑word removal.

Use pre‑trained Word2Vec/FastText models for Persian (e.g., from gensim).

Support larger datasets (e.g., news articles, social media posts).

Add a Streamlit dashboard for interactive exploration.

Compare with transformer‑based embeddings (e.g., ParsBERT, multilingual BERT).

Cluster sentences using K‑means or hierarchical clustering.

Add a query‑by‑example feature – input a new sentence and find the most similar existing one.