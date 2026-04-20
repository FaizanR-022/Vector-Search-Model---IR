import os
import math
import json
import re
import string
from nltk.stem import PorterStemmer

# -----------------------------------------------
# VSM - Vector Space Model
# Information Retrieval Assignment 2
# -----------------------------------------------

SPEECHES_DIR = "./speeches/"
STOPWORDS_FILE = "./stopwords.txt"
INDEX_FILE = "vsm_index.json"
ALPHA = 0.005  # threshold for filtering results

stemmer = PorterStemmer()


# ---- PREPROCESSING ----

def load_stopwords(filepath):
    stopwords = set()
    with open(filepath, 'r') as f:
        for line in f:
            word = line.strip().lower()
            if word:
                stopwords.add(word)
    return stopwords


def preprocess(text, stopwords):
    # lowercase
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # tokenize (simple split)
    tokens = text.split()
    # remove stopwords and stem
    processed = []
    for token in tokens:
        if token not in stopwords and token.isalpha():
            stemmed = stemmer.stem(token)
            processed.append(stemmed)
    return processed


# ---- INDEXING ----

def build_index(speeches_dir, stopwords):
    # tf[doc_id][term] = count
    tf = {}
    # df[term] = number of docs containing term
    df = {}
    # total docs
    doc_names = {}

    files = sorted(os.listdir(speeches_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    print(f"Building index from {len(files)} documents...")

    for i, filename in enumerate(files):
        if not filename.endswith('.txt'):
            continue
        
        doc_id = str(i)
        doc_names[doc_id] = filename
        filepath = os.path.join(speeches_dir, filename)

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        tokens = preprocess(text, stopwords)
        
        # count term freq in this doc
        tf[doc_id] = {}
        for token in tokens:
            if token in tf[doc_id]:
                tf[doc_id][token] += 1
            else:
                tf[doc_id][token] = 1

        # update doc freq
        for term in tf[doc_id]:
            if term in df:
                df[term] += 1
            else:
                df[term] = 1

    print(f"Index built. Total unique terms: {len(df)}")
    return tf, df, doc_names


def compute_tfidf(tf, df, N):
    # tfidf[doc_id][term] = tfidf score
    tfidf = {}
    for doc_id in tf:
        tfidf[doc_id] = {}
        for term, freq in tf[doc_id].items():
            idf = math.log(N / df[term])
            tfidf[doc_id][term] = freq * idf
    return tfidf


def compute_doc_magnitudes(tfidf):
    # precompute magnitudes for cosine similarity
    magnitudes = {}
    for doc_id, terms in tfidf.items():
        mag = math.sqrt(sum(v**2 for v in terms.values()))
        magnitudes[doc_id] = mag
    return magnitudes


def save_index(tf, df, doc_names, filepath):
    data = {
        'tf': tf,
        'df': df,
        'doc_names': doc_names
    }
    with open(filepath, 'w') as f:
        json.dump(data, f)
    print(f"Index saved to {filepath}")


def load_index(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Index loaded from {filepath}")
    return data['tf'], data['df'], data['doc_names']


# ---- QUERY PROCESSING ----

def process_query(query_text, stopwords):
    tokens = preprocess(query_text, stopwords)
    # build query tf
    query_tf = {}
    for token in tokens:
        if token in query_tf:
            query_tf[token] += 1
        else:
            query_tf[token] = 1
    return query_tf


def compute_query_tfidf(query_tf, df, N):
    query_tfidf = {}
    for term, freq in query_tf.items():
        if term in df:
            idf = math.log(N / df[term])
            query_tfidf[term] = freq * idf
        # if term not in df, idf would be log(N/0) which is undefined
        # so we just skip terms not in our corpus
    return query_tfidf


def cosine_similarity(query_tfidf, doc_tfidf, doc_magnitude):
    if doc_magnitude == 0:
        return 0.0
    
    # dot product
    dot = 0.0
    for term, q_val in query_tfidf.items():
        if term in doc_tfidf:
            dot += q_val * doc_tfidf[term]
    
    # query magnitude
    q_mag = math.sqrt(sum(v**2 for v in query_tfidf.values()))
    
    if q_mag == 0:
        return 0.0
    
    return dot / (q_mag * doc_magnitude)


def search(query_text, tfidf, df, magnitudes, doc_names, stopwords, alpha=ALPHA):
    N = len(tfidf)
    
    # process query
    query_tf = process_query(query_text, stopwords)
    if not query_tf:
        print("Query has no valid terms after preprocessing.")
        return []
    
    query_tfidf = compute_query_tfidf(query_tf, df, N)
    if not query_tfidf:
        print("No query terms found in corpus.")
        return []

    # compute cosine similarity with all docs
    scores = {}
    for doc_id in tfidf:
        sim = cosine_similarity(query_tfidf, tfidf[doc_id], magnitudes[doc_id])
        if sim > alpha:
            scores[doc_id] = sim

    # sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # get doc numbers for display (matching the gold standard format)
    results = []
    for doc_id, score in ranked:
        filename = doc_names[doc_id]
        # extract doc number from filename like speech_3.txt -> 3
        doc_num = filename.replace('speech_', '').replace('.txt', '')
        results.append((doc_num, score))
    
    return results


# ---- MAIN CLI ----

def main():
    # load stopwords
    stopwords = load_stopwords(STOPWORDS_FILE)
    
    # build or load index
    if os.path.exists(INDEX_FILE):
        choice = input("Index file found. Load existing index? (y/n): ").strip().lower()
        if choice == 'y':
            tf, df, doc_names = load_index(INDEX_FILE)
        else:
            tf, df, doc_names = build_index(SPEECHES_DIR, stopwords)
            save_index(tf, df, doc_names, INDEX_FILE)
    else:
        tf, df, doc_names = build_index(SPEECHES_DIR, stopwords)
        save_index(tf, df, doc_names, INDEX_FILE)
    
    N = len(tf)
    
    # compute tfidf and magnitudes
    print("Computing TF-IDF weights...")
    tfidf = compute_tfidf(tf, df, N)
    magnitudes = compute_doc_magnitudes(tfidf)
    print("Ready!\n")

    # CLI loop
    while True:
        print("=" * 50)
        query = input("Enter query (or 'quit' to exit): ").strip()
        
        if query.lower() == 'quit':
            print("Bye!")
            break
        
        if not query:
            continue
        
        results = search(query, tfidf, df, magnitudes, doc_names, stopwords)
        
        if not results:
            print("No documents found above threshold.")
        else:
            print(f"\nResults for: '{query}'")
            print(f"Total documents retrieved: {len(results)}")
            print(f"Document IDs: {set(r[0] for r in results)}")
            print("\nTop 10 ranked results:")
            print(f"{'Rank':<6} {'Doc':<10} {'Score':<10} {'File'}")
            print("-" * 45)
            for rank, (doc_num, score) in enumerate(results[:10], 1):
                # find filename
                fname = ""
                for did, dname in doc_names.items():
                    if dname == f"speech_{doc_num}.txt":
                        fname = dname
                        break
                print(f"{rank:<6} {doc_num:<10} {score:.4f}     {fname}")
        print()


if __name__ == "__main__":
    main()
