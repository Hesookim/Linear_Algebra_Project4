# 📊 Persian Text Analysis with TF‑IDF, Word2Vec, and SVD

> A comprehensive project for representing Persian sentences as numerical vectors using TF‑IDF, Word2Vec, and a custom CountVectorizer embedding, followed by dimensionality reduction with SVD and semantic similarity analysis.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation & Requirements](#installation--requirements)
- [How to Run](#how-to-run)
- [Methods Explained](#methods-explained)
- [Dimensionality Reduction with SVD](#dimensionality-reduction-with-svd)
- [Similarity Calculation](#similarity-calculation)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualizations](#visualizations)
- [Sample Output](#sample-output)
- [Results Interpretation](#results-interpretation)
- [Educational Value](#educational-value)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [License](#license)

---

## 📖 Project Overview

This project applies **Natural Language Processing (NLP)** techniques to Persian (Farsi) text data. The goal is to convert a set of Persian sentences into numerical vectors using three different embedding methods, then reduce their dimensionality using **Singular Value Decomposition (SVD)**, and finally compute **cosine similarity** between sentences to measure semantic relatedness.

The three embedding methods compared are:
1. **TF‑IDF** (Term Frequency‑Inverse Document Frequency) – a classic statistical method.
2. **Word2Vec** – a neural embedding model implemented from scratch with a simplified Skip‑Gram architecture.
3. **CountVectorizer** – a simple bag‑of‑words count matrix (used as a "custom embedding").

Each method is evaluated after SVD reduction, and the results are compared in terms of:
- Matrix shape and sparsity
- Explained variance
- Similarity matrices
- Execution time

The project includes comprehensive visualizations: heatmaps, scatter plots, variance bar charts, and similarity histograms.

This work was developed as part of a **Linear Algebra course** to demonstrate the practical application of matrix decomposition and dimensionality reduction in text analysis.

---

## 🎯 Objectives

- ✅ Preprocess Persian text (normalization, punctuation removal).
- ✅ Build document‑term matrices using **TF‑IDF**, **Word2Vec**, and **CountVectorizer**.
- ✅ Reduce dimensionality using **Truncated SVD**.
- ✅ Compute **cosine similarity** matrices for each method.
- ✅ Compare the semantic similarity results across methods.
- ✅ Visualize the results with heatmaps, scatter plots, variance charts, and histograms.
- ✅ Benchmark execution time for each step.
- ✅ Identify which embedding method best captures semantic meaning.

---

## 📂 Dataset

The dataset (`dataset.csv`) contains **40 Persian sentences** covering three main topics:

| Topic | Example Sentences |
| :--- | :--- |
| **Artificial Intelligence & ML** | "هوش مصنوعی آینده فناوری را تغییر می‌دهد." |
| **Sports** | "فوتبال محبوب‌ترین ورزش جهان است." |
| **Health & Lifestyle** | "ورزش منظم خطر بیماری قلبی را کاهش می‌دهد." |
| **Programming & Education** | "پایتون زبان محبوبی برای علم داده است." |

The sentences are manually crafted to include clear semantic clusters, making it easy to evaluate the quality of similarity detection.

---

## 📁 Project Structure
project4_text_analysis/
│
├── app.py # Main orchestration script
├── config.py # Configuration constants
├── dataset.csv # Input Persian sentences
├── requirements.txt # Python dependencies
├── README.md # This file
│
├── methods/ # Embedding methods
│ ├── tfidf_model.py # TF‑IDF vectorization
│ ├── word2vec_model.py # Custom Word2Vec implementation
│ ├── count_embedding.py # CountVectorizer embedding
│ └── svd_reduction.py # Truncated SVD reduction
│
├── utils/ # Utility functions
│ ├── preprocessing.py # Text preprocessing
│ ├── similarity.py # Cosine similarity calculation
│ ├── report.py # Print top similar sentences
│ └── benchmark.py # Execution time measurement
│
├── visualization.py # Plotting functions
│
└── outputs/ # Generated figures (per method)
├── tfidf/
│ ├── heatmap.png
│ ├── svd_scatter.png
│ ├── explained_variance.png
│ └── similarity_histogram.png
├── word2vec/
│ └── ...
└── count_embedding/
└── ...

---

## ⚙️ Installation & Requirements

### System Requirements
- Python 3.10 or higher (recommended 3.12 for better library compatibility)

### Required Libraries

| Library | Purpose |
| :------ | :------ |
| `numpy` | Numerical computations |
| `pandas` | Data loading and manipulation |
| `scikit-learn` | TF‑IDF, CountVectorizer, TruncatedSVD, cosine similarity |
| `matplotlib` | Visualization |
| `re` (built‑in) | Text preprocessing |

### Installation Steps

```bash
pip install numpy pandas scikit-learn matplotlib
or
pip install -r requirements.txt

## 🚀 How to Run
python app.py


## 🛠️ Methods Explained

### 1. TF‑IDF (Term Frequency – Inverse Document Frequency)

- **What it does**: Converts each sentence into a vector where each dimension corresponds to a unique word. The value is the TF‑IDF weight, which reflects how important a word is to a document relative to the entire corpus.
- **Advantages**: Simple, interpretable, captures word importance.
- **Disadvantages**: Sparse, high‑dimensional, ignores word order and semantics.
- **Implementation**: `TfidfVectorizer` from scikit‑learn.

### 2. Word2Vec (Custom Implementation)

- **What it does**: Learns dense vector representations (embeddings) for words from their context (Skip‑Gram model). Each sentence is represented as the average of its word vectors.
- **Advantages**: Dense, captures semantic meaning and word relationships.
- **Disadvantages**: Requires more data, training time, and careful tuning.
- **Implementation**: A simplified Skip‑Gram model with gradient descent, written from scratch using NumPy.
  - Vector size: 50
  - Window size: 2
  - Learning rate: 0.01
  - Epochs: 100

### 3. CountVectorizer (Custom Embedding)

- **What it does**: Creates a simple bag‑of‑words count matrix (raw term frequencies).
- **Advantages**: Simple and fast.
- **Disadvantages**: Sparse, no normalization, affected by document length.
- **Implementation**: `CountVectorizer` from scikit‑learn (used as a "dummy" embedding for comparison).

---

## 📉 Dimensionality Reduction with SVD

### Why SVD?
- The original TF‑IDF and CountVectorizer matrices are high‑dimensional and sparse.
- SVD (specifically **Truncated SVD**) reduces dimensions to a specified number of components (default: 10), capturing the most important latent features.
- For Word2Vec, the matrix is already dense, but SVD can still help reduce noise and visualize the data in 2D.

### Implementation
- Using `TruncatedSVD` from scikit‑learn with `n_components = 10` (configurable in `config.py`).
- The explained variance ratio is computed to measure how much information is retained.

---

## 📐 Similarity Calculation

- **Cosine similarity** measures the angle between two vectors.
- Range: `-1` (opposite) to `1` (identical), but in text analysis, values are usually between 0 and 1.
- A high similarity score indicates that two sentences are semantically close.

The similarity matrix is computed using `cosine_similarity` from scikit‑learn.

---

## 📊 Evaluation Metrics

| Metric | Description |
| :--- | :--- |
| **Matrix Shape** | Number of documents × number of features (or components). |
| **Vocabulary Size** | Number of unique words in the corpus. |
| **Explained Variance** | Proportion of variance retained by each SVD component. |
| **Total Explained Variance** | Sum of variance ratios for all components. |
| **Execution Time** | Time taken for vectorization, SVD, and similarity calculation. |
| **Top Similar Sentences** | For each sentence, the most similar other sentences with scores above a threshold. |

---

## 🎨 Visualizations

All figures are automatically saved in `outputs/<method_name>/`.

### 1. Similarity Heatmap
- Shows the pairwise cosine similarity matrix as a color‑coded grid.
- Darker colors indicate higher similarity.
- Helps identify clusters of related sentences.

### 2. 2D Scatter Plot (After SVD)
- Projects the reduced document vectors onto the first two principal components.
- Visually reveals clusters of similar sentences.

### 3. Explained Variance Bar Chart
- Displays the variance captured by each SVD component.
- The first few components usually capture most of the variance.

### 4. Similarity Histogram
- Distribution of all pairwise similarity values.
- A right‑skewed distribution indicates that most sentences are moderately similar; a wide spread suggests good differentiation.

---

## 🖥️ Sample Output

### Console Output (Abridged)

```
==================================================
Dataset Information
==================================================
                                                text
0     هوش مصنوعی آینده فناوری را تغییر می‌دهد.
1     یادگیری ماشین یکی از شاخه‌های هوش مصنوعی است.
...

==================================================
TF-IDF Vectorization
==================================================
Matrix Shape: (40, 120)
TF-IDF Time: 0.00123 seconds
Vocabulary Size: 120

==================================================
TF-IDF SVD Dimension Reduction
==================================================
SVD Time: 0.00087 seconds
Reduced Matrix Shape: (40, 10)
Explained Variance Ratio: [0.15 0.08 0.06 ...]
Total Explained Variance: 0.512

==================================================
Top Similar Sentences
==================================================
Sentence 1: هوش مصنوعی آینده فناوری را تغییر می‌دهد.
Most Similar Sentences:
  -> (2) یادگیری ماشین یکی از شاخه‌های هوش مصنوعی است.
     Similarity: 0.873
  -> (3) شبکه‌های عصبی در پردازش تصویر کاربرد دارند.
     Similarity: 0.412

==================================================
Word2Vec Embedding
==================================================
Word2Vec Matrix Shape: (40, 50)
Word2Vec Time: 2.34567 seconds

Word2Vec SVD Dimension Reduction
==================================================
Word2Vec Reduced Matrix Shape: (40, 10)
Total Explained Variance: 0.687

...

==================================================
CountVectorizer Embedding
==================================================
Count Embedding Matrix Shape: (40, 120)
Count Embedding Time: 0.00098 seconds
...
```

### Visual Outputs (saved in `outputs/`)

- **Heatmap**: Shows clear diagonal clusters for sentences of the same topic.
- **Scatter plot**: Sentences from AI/ML, sports, health, and programming topics form distinct groups.
- **Variance bar chart**: First 3‑4 components capture most of the variance.
- **Histogram**: Similarity scores for TF‑IDF show a bimodal distribution, while Word2Vec shows a more uniform distribution.

---

## 📈 Results Interpretation

| Method | Strengths | Weaknesses | Best Use Case |
| :--- | :--- | :--- | :--- |
| **TF‑IDF** | Fast, interpretable, good for keyword matching. | Sparse, ignores word order and semantics. | Document retrieval, keyword extraction. |
| **Word2Vec** | Dense, captures semantic relationships, handles synonyms. | Needs more data, slower training. | Semantic similarity, recommendation systems. |
| **CountVectorizer** | Simplest, fast. | No normalization, sparse. | Baseline comparison. |

**Overall**: For this dataset, **Word2Vec** produced the most meaningful similarity scores (with higher intra‑topic and lower inter‑topic similarities). However, it required significantly more time to train. **TF‑IDF** provided a good balance between speed and accuracy.

---

## 🎓 Educational Value

This project demonstrates the practical application of several core linear algebra and NLP concepts:

| Concept | Application |
| :------ | :----------- |
| **Matrix Representation** | Document‑term matrices |
| **Dimensionality Reduction** | Truncated SVD for noise reduction and visualization |
| **Singular Value Decomposition** | Latent semantic analysis |
| **Cosine Similarity** | Measuring semantic relatedness |
| **Word Embeddings** | Dense semantic vectors (Word2Vec) |
| **Bag‑of‑Words** | TF‑IDF and CountVectorizer |
| **Benchmarking** | Time complexity analysis |
| **Visualization** | Interpreting high‑dimensional data |

---

## 🚧 Future Improvements

- **Add Persian NLP preprocessing** – use libraries like `hazm` for tokenization, stemming, and stop‑word removal.
- **Use pre‑trained Word2Vec/FastText** models for Persian (e.g., from `gensim`).
- **Support larger datasets** (e.g., news articles, social media posts).
- **Add a Streamlit dashboard** for interactive exploration.
- **Compare with transformer‑based embeddings** (e.g., ParsBERT, multilingual BERT).
- **Cluster sentences** using K‑means or hierarchical clustering.
- **Add a query‑by‑example feature** – input a new sentence and find the most similar existing one.

---

## 👩‍💻 Author

**Minoo Namdar**  
Bachelor of Computer Engineering  
Linear Algebra Course Project  
Professor: Dr. Abasszadeh  
Year: 2026

---

## 📄 License

This project is created for educational purposes and is free to use, modify, and distribute for non‑commercial academic work.

---