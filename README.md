# CS4051 - Information Retrieval

## Assignment 2 - Vector Space Model (VSM)

A Vector Space Model based Information Retrieval system built on Trump's speeches dataset (56 documents). Supports ranked retrieval using TF-IDF weighting and cosine similarity.

---

## Features

- TF-IDF based document indexing
- Cosine similarity for ranked retrieval
- Preprocessing pipeline: tokenization, stopword removal, Porter stemming
- Save/load index to avoid rebuilding every time
- Web GUI (Flask) + CLI mode

---

## Project Structure

```
IR-A02/
├── vsm.py            # core VSM logic (indexing + search)
├── app.py            # Flask web server
├── static/
│   └── index.html    # frontend GUI
├── speeches/         # 56 Trump speech documents
├── stopwords.txt     # stopword list
└── vsm_index.json    # generated index (created after running vsm.py)
```

---

## Setup

Make sure you have Python installed, then install the required libraries:

```bash
pip install flask nltk
```

You also need to download the NLTK Porter Stemmer data (one time only):

```python
import nltk
nltk.download('punkt')
```

---

## How to Run

### Step 1 — Build the Index

```bash
python vsm.py
```

This reads all speeches, computes TF-IDF weights, and saves the index to `vsm_index.json`.

### Step 2 — Start the Web App

```bash
python app.py
```

Then open your browser at: **http://localhost:5000**

### CLI Mode

You can also just run `python vsm.py` directly — it has a command line interface where you can type queries and see ranked results.

---

## How it Works

1. **Preprocessing** — text is lowercased, punctuation removed, stopwords filtered out, and each word is stemmed using Porter Stemmer
2. **Indexing** — term frequency (TF) is counted per document, and document frequency (DF) is tracked across the corpus
3. **TF-IDF** — computed as `tf * log(N / df)` for each term in each document
4. **Query Processing** — same preprocessing is applied to the query
5. **Cosine Similarity** — dot product of query and document vectors divided by their magnitudes, results ranked by score
6. **Threshold** — documents with similarity below `0.005` are filtered out
