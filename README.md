# CS6370 — Information Retrieval System on Cranfield Dataset

This project implements a complete Information Retrieval (IR) pipeline on the Cranfield dataset using classical and advanced retrieval techniques.

The system supports:

- Sentence Segmentation
- Tokenization
- Lemmatization / Stemming
- Stopword Removal
- TF-IDF Vector Space Retrieval
- Retrieval Evaluation Metrics
- Adversarial Testing
- Data-Driven Stopword Generation
- Advanced Retrieval Improvements:
  - Latent Semantic Analysis (LSA)
  - Query Expansion (QE)
  - Hybrid Retrieval
  - Smart TF-IDF
  - Boosted Query TF-IDF
  - Bigram Retrieval
  - Spell Correction
  - Hybrid Bigram Retrieval

---

# Project Structure

```text
.
├── main.py
├── improvements.py
├── advanced_improvements.py
├── informationRetrieval.py
├── evaluation.py
├── sentenceSegmentation.py
├── tokenization.py
├── inflectionReduction.py
├── stopwordRemoval.py
├── tfidf_stopwords.py
├── test_adversarial.py
├── util.py
├── cranfield/
│   ├── cran_docs.json
│   ├── cran_queries.json
│   └── cran_qrels.json
└── output/
```

---

# Dataset

This project uses the Cranfield Dataset.

Dataset files:

- `cran_docs.json` → document corpus
- `cran_queries.json` → user queries
- `cran_qrels.json` → relevance judgments

---

# Overall IR Pipeline

```text
Documents / Queries
        ↓
Sentence Segmentation
        ↓
Tokenization
        ↓
Inflection Reduction
        ↓
Stopword Removal
        ↓
Vectorization
        ↓
Ranking
        ↓
Evaluation
```

---

# File Descriptions

---

# 1. `main.py`

Main driver file of the project.

This file was extended to evaluate both the baseline retrieval model and all improved retrieval models for Part 5 of the project.

---

## Responsibilities

- Loads Cranfield dataset
- Preprocesses documents and queries
- Builds retrieval indices
- Ranks documents
- Computes evaluation metrics
- Runs all retrieval models
- Generates evaluation plots

---

## Retrieval Models Evaluated

The following systems are automatically evaluated:

1. Baseline TF-IDF
2. LSA Retrieval
3. Query Expansion
4. Hybrid Retrieval
5. Smart TF-IDF
6. Boosted Query TF-IDF
7. Bigram Retrieval
8. Spell Correction
9. Hybrid Bigram Retrieval

---

## Run

```bash
python main.py
```

---

# 2. `informationRetrieval.py`

Implements the baseline TF-IDF Vector Space Model.

---

## Responsibilities

- Builds vocabulary
- Computes:
  - Term Frequency (TF)
  - Inverse Document Frequency (IDF)
  - TF-IDF vectors
- Computes cosine similarity
- Ranks documents for each query

---

## Retrieval Model

```text
TF-IDF + Cosine Similarity
```

---

# 3. `improvements.py`

Implements semantic retrieval improvements.

---

## Methods Implemented

---

## A. Latent Semantic Analysis (LSA)

Uses Singular Value Decomposition (SVD) to project documents into a latent semantic space.

### Purpose

- Capture semantic similarity
- Reduce sparsity
- Handle synonymy

---

## B. Query Expansion (QE)

Expands queries using WordNet synonyms.

### Example

```python
"heat" → ["thermal", "temperature"]
```

### Purpose

- Improve recall
- Reduce vocabulary mismatch

---

## C. Hybrid Retrieval

Combines:

- TF-IDF
- LSA
- Query Expansion

### Purpose

- Balance semantic retrieval and lexical precision

---

# 4. `advanced_improvements.py`

Implements additional retrieval improvements beyond the standard semantic methods.

---

## Methods Implemented

---

## A. Smart TF-IDF

Uses:

- Log-scaled TF
- Smoothed IDF

### Formula

```text
tf = 1 + log(tf)
idf = log((N+1)/(df+1)) + 1
```

### Purpose

- Improve ranking quality
- Reduce impact of very frequent terms

---

## B. Boosted Query TF-IDF

Boosts rare query terms using squared IDF weighting.

### Purpose

- Increase importance of informative query terms
- Improve retrieval specificity

---

## C. Bigram Retrieval

Adds phrase-level indexing using bigrams.

### Example

```text
boundary layer → boundary_layer
```

### Purpose

- Improve phrase-aware retrieval
- Capture contextual information

---

## D. Spell Correction Retrieval

Performs lightweight spelling correction before ranking.

### Purpose

- Improve robustness to misspelled queries

---

## E. Hybrid Bigram Retrieval

Combines:

- Bigram indexing
- Semantic retrieval
- Query expansion

### Purpose

- Jointly model semantic and phrase-level information

---

# 5. `sentenceSegmentation.py`

Implements sentence segmentation.

---

## Methods

- Naive regex splitting
- NLTK Punkt tokenizer
- spaCy sentence segmentation

---

## Example

```python
segmenter.punkt(text)
```

---

# 6. `tokenization.py`

Implements tokenization.

---

## Methods

- Naive regex tokenizer
- Penn Treebank tokenizer
- spaCy tokenizer

---

# 7. `inflectionReduction.py`

Implements inflection reduction.

---

## Methods

- Porter Stemmer
- WordNet Lemmatizer

---

## Default

```python
WordNet Lemmatizer
```

---

## Purpose

Reduce words to canonical forms.

### Example

```text
running → run
cars → car
```

---

# 8. `stopwordRemoval.py`

Removes stopwords using the NLTK stopword list.

---

## Example Removed Words

```text
the, is, are, and, for
```

---

## Purpose

Remove non-informative terms.

---

# 9. `evaluation.py`

Implements retrieval evaluation metrics.

---

## Metrics Implemented

- Precision@k
- Recall@k
- F0.5-score@k
- MAP
- nDCG
- MRR

---

## Purpose

Quantitatively evaluate retrieval quality.

---

# 10. `tfidf_stopwords.py`

Creates data-driven stopwords from the Cranfield corpus.

---

## Method

A word is considered a stopword if it appears in more than a threshold percentage of documents.

Default:

```python
threshold = 0.3
```

---

## Purpose

Identify domain-specific frequent words.

---

## Run

```bash
python tfidf_stopwords.py
```

---

# 11. `test_adversarial.py`

Tests robustness of sentence segmentation methods.

---

## Test Cases

- Abbreviations
- Decimal numbers
- Quotes
- Ellipsis
- Acronyms
- Titles

---

## Purpose

Compare segmentation robustness across methods.

---

## Run

```bash
python test_adversarial.py
```

---

# 12. `util.py`

Contains utility imports and NLTK downloads.

---

## Downloads

- punkt
- wordnet
- stopwords
- omw-1.4

---

# Installation

---

## Step 1 — Clone Repository

```bash
git clone https://github.com/BaristaBandits/Toy-IR-System.git
cd Toy-IR-System
```

---

## Step 2 — Install Dependencies

```bash
pip install numpy nltk matplotlib spacy scikit-learn autocorrect
```

---

## Step 3 — Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

# Running the Project

---

# Run Full Evaluation

```bash
python main.py
```

This automatically evaluates:

1. Baseline TF-IDF
2. LSA
3. Query Expansion
4. Hybrid Retrieval
5. Smart TF-IDF
6. Boosted Query TF-IDF
7. Bigram Retrieval
8. Spell Correction
9. Hybrid Bigram Retrieval

---

# Run Adversarial Sentence Segmentation Tests

```bash
python test_adversarial.py
```

---

# Generate Data-Driven Stopwords

```bash
python tfidf_stopwords.py
```

---

# Example Output

```text
======================================
RUNNING: baseline
======================================

FINAL RESULTS @10
-----------------------------
Precision@10 : 0.2791
Recall@10    : 0.3950
F-score@10   : 0.2836
MAP@10       : 0.6373
nDCG@10      : 0.4561
MRR@10       : 0.7258
```

---

# Output Files

The system generates:

```text
output/
├── segmented_docs.txt
├── tokenized_docs.txt
├── reduced_docs.txt
├── stopword_removed_docs.txt
├── segmented_queries.txt
├── tokenized_queries.txt
├── reduced_queries.txt
├── stopword_removed_queries.txt
├── baseline_plot.png
├── lsa_plot.png
├── hybrid_plot.png
└── ...
```

---

# Experimental Findings

Key observations from experiments:

- LSA improved semantic retrieval and recall
- Smart TF-IDF achieved strongest ranking quality
- Bigram retrieval improved phrase-aware retrieval
- Query expansion improved semantic coverage
- Spell correction improved robustness
- Hybrid methods balanced lexical and semantic matching

---

# Authors

- Swathi Shree N (EE22B149)
- Vignesh Gunda (EE22B105)
- K Nipun (EE22B113)
- I Sai Ganesh (ME22B133)
- B Navadeep (CE22B047)

---

# Course Information

CS6370 — Natural Language Processing  
Department of Computer Science and Engineering  
Indian Institute of Technology Madras
