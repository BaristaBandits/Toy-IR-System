# CS6370 — Information Retrieval System on Cranfield Dataset

This project implements a complete Information Retrieval (IR) pipeline using the Cranfield dataset.  
The system supports:

- Sentence Segmentation
- Tokenization
- Lemmatization / Stemming
- Stopword Removal
- TF-IDF Vector Space Retrieval
- Evaluation Metrics
- Advanced Retrieval Improvements:
  - Latent Semantic Analysis (LSA)
  - Query Expansion (QE)
  - Hybrid Retrieval

---

# Project Structure

```text
.
├── main.py
├── improvements.py
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

This project uses the Cranfield Dataset:

- `cran_docs.json` → document corpus
- `cran_queries.json` → user queries
- `cran_qrels.json` → relevance judgments

---

# Pipeline Overview

The retrieval pipeline is:

```text
Documents / Queries
        ↓
Sentence Segmentation
        ↓
Tokenization
        ↓
Lemmatization / Stemming
        ↓
Stopword Removal
        ↓
TF-IDF Vectorization
        ↓
Document Ranking
        ↓
Evaluation
```

---

# File Descriptions

---

## 1. `main.py`

Main driver file of the project.

### Responsibilities

- Loads Cranfield dataset
- Preprocesses documents and queries
- Builds retrieval index
- Ranks documents
- Computes evaluation metrics
- Runs:
  - Baseline TF-IDF
  - LSA Retrieval
  - Query Expansion
  - Hybrid Retrieval
- Generates evaluation plots

### Run

```bash
python main.py
```

---

## 2. `informationRetrieval.py`

Implements the baseline TF-IDF Vector Space Model.

### Responsibilities

- Builds vocabulary
- Computes:
  - TF
  - IDF
  - TF-IDF vectors
- Computes cosine similarity
- Ranks documents for each query

### Retrieval Model

```text
TF-IDF + Cosine Similarity
```

---

## 3. `improvements.py`

Implements improved retrieval methods.

### Methods Implemented

---

### A. LSA (Latent Semantic Analysis)

Uses Singular Value Decomposition (SVD) to project documents into a latent semantic space.

Purpose:

- Capture semantic similarity
- Reduce sparsity
- Handle synonymy

---

### B. Query Expansion (QE)

Expands queries using manually defined semantic synonyms.

Example:

```python
"heat" → ["thermal", "temperature"]
```

Purpose:

- Improve recall
- Reduce vocabulary mismatch

---

### C. Hybrid Retrieval

Combines:

- TF-IDF
- LSA
- Query Expansion

Purpose:

- Achieve best ranking quality

---

## 4. `sentenceSegmentation.py`

Implements sentence segmentation using:

### Methods

- Naive regex splitting
- NLTK Punkt tokenizer
- spaCy sentence segmentation

### Example

```python
segmenter.punkt(text)
```

---

## 5. `tokenization.py`

Implements tokenization.

### Methods

- Naive regex tokenizer
- Penn Treebank tokenizer
- spaCy tokenizer

---

## 6. `inflectionReduction.py`

Implements inflection reduction.

### Methods

- Porter Stemmer
- WordNet Lemmatizer

Default:

```python
WordNet Lemmatizer
```

Purpose:

- Reduce words to canonical forms

Example:

```text
running → run
cars → car
```

---

## 7. `stopwordRemoval.py`

Removes stopwords using NLTK stopword list.

Example removed words:

```text
the, is, are, and, for
```

Purpose:

- Remove non-informative terms

---

## 8. `evaluation.py`

Implements evaluation metrics.

### Metrics

- Precision@k
- Recall@k
- F0.5-score@k
- MAP
- nDCG
- MRR

Purpose:

- Quantitatively evaluate retrieval quality

---

## 9. `tfidf_stopwords.py`

Creates data-driven stopwords from the Cranfield corpus.

### Method

A word is considered a stopword if it appears in more than a threshold percentage of documents.

Default:

```python
threshold = 0.3
```

Purpose:

- Identify domain-specific frequent words

### Run

```bash
python tfidf_stopwords.py
```

---

## 10. `test_adversarial.py`

Tests robustness of sentence segmentation.

### Tests

- Abbreviations
- Decimal numbers
- Quotes
- Ellipsis
- Acronyms
- Titles

Purpose:

- Compare segmentation approaches

### Run

```bash
python test_adversarial.py
```

---

## 11. `util.py`

Utility imports and NLTK downloads.

Downloads:

- punkt
- wordnet
- stopwords

---

# Installation

## Step 1 — Install dependencies

```bash
pip install numpy nltk matplotlib spacy scikit-learn
```

---

## Step 2 — Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

---

# Running the Project

## Run Baseline + Improvements

```bash
python main.py
```

This automatically runs:

1. Baseline TF-IDF
2. LSA
3. Query Expansion
4. Hybrid Retrieval

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
└── eval_plot.png
```

---

# Evaluation Metrics

The system evaluates:

| Metric | Meaning |
|---|---|
| Precision@k | Fraction of retrieved docs that are relevant |
| Recall@k | Fraction of relevant docs retrieved |
| F0.5-score | Precision-weighted F-score |
| MAP | Mean Average Precision |
| nDCG | Ranking quality |
| MRR | Reciprocal rank of first relevant doc |

---

# Running Baseline vs Improvements

The modified `main.py` automatically compares:

1. Baseline TF-IDF
2. LSA
3. Query Expansion
4. Hybrid Retrieval

### Example Terminal Output

```text
========================
BASELINE RESULTS
========================

MAP : ...
nDCG: ...
MRR : ...

========================
LSA RESULTS
========================

MAP : ...
nDCG: ...
MRR : ...
```

---

# Expected Improvements

| Method | Expected Effect |
|---|---|
| LSA | Better semantic matching |
| Query Expansion | Better recall |
| Hybrid | Best overall retrieval |

---

# Observed Limitations of TF-IDF

The baseline Vector Space Model suffers from:

- Lexical mismatch
- Semantic blindness
- High sparsity
- OOV issues
- Word-order ignorance

The improvements attempt to reduce these problems.

---

# Example Research Findings

## Baseline TF-IDF

- Strong lexical matching
- Weak semantic understanding

## LSA

- Captures latent semantic structure
- Improves synonym handling

## Query Expansion

- Helps bridge vocabulary mismatch

## Hybrid

- Combines strengths of all methods

---

# Example Commands

## Run complete system

```bash
python main.py
```

## Run adversarial sentence segmentation tests

```bash
python test_adversarial.py
```

## Generate data-driven stopwords

```bash
python tfidf_stopwords.py
```

---

# Authors

- Vignesh Gunda (EE22B105)
- K Nipun (EE22B113)
- Swathi Shree N (EE22B149)
- I Sai Ganesh (ME22B133)
- B Navadeep (CE22B047)

CS6370 — Natural Language Processing  
Indian Institute of Technology Madras# Toy-IR-System
Course Project CS 6370
