"""
VSM Web App - Flask Backend
IR Assignment 2
"""

from flask import Flask, request, jsonify, send_from_directory
import os
import vsm

app = Flask(__name__, static_folder='static')

# ---- load everything on startup ----
print("Loading stopwords and index...")
stopwords = vsm.load_stopwords('stopwords.txt')
tf, df, doc_names = vsm.load_index('vsm_index.json')
N = len(tf)
tfidf = vsm.compute_tfidf(tf, df, N)
magnitudes = vsm.compute_doc_magnitudes(tfidf)
print("Ready! Server starting...")


def get_snippet(doc_num, query_terms, length=300):
    """Get a text snippet from a speech, trying to include query terms if possible."""
    path = f'./speeches/Trump Speechs/Trump Speechs/speech_{doc_num}.txt'
    try:
        with open(path, 'r', errors='ignore') as f:
            text = f.read()
    except:
        return ""
    
    # try to find a section with query terms in it
    text_lower = text.lower()
    best_pos = 0
    for term in query_terms:
        pos = text_lower.find(term.lower())
        if pos != -1:
            best_pos = max(0, pos - 50)
            break
    
    snippet = text[best_pos: best_pos + length]
    # clean up
    snippet = snippet.replace('\n', ' ').strip()
    if best_pos > 0:
        snippet = "..." + snippet
    if best_pos + length < len(text):
        snippet = snippet + "..."
    return snippet


def get_title(doc_num):
    """Get the first line (title) of a speech."""
    path = f'./speeches/Trump Speechs/Trump Speechs/speech_{doc_num}.txt'
    try:
        with open(path, 'r', errors='ignore') as f:
            first_line = f.readline().strip()
        return first_line if first_line else f"Speech {doc_num}"
    except:
        return f"Speech {doc_num}"


# ---- routes ----

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': 'Empty query', 'results': [], 'count': 0})

    # run VSM search
    results = vsm.search(query, tfidf, df, magnitudes, doc_names, stopwords)

    # get query terms for snippet highlighting
    query_tokens = vsm.preprocess(query, stopwords)

    # build response
    response_results = []
    for doc_num, score in results:
        response_results.append({
            'doc_id': doc_num,
            'score': round(score, 4),
            'title': get_title(doc_num),
            'snippet': get_snippet(doc_num, query_tokens),
            'filename': f'speech_{doc_num}.txt'
        })

    return jsonify({
        'query': query,
        'count': len(results),
        'results': response_results
    })


@app.route('/stats')
def stats():
    """Return basic corpus stats for display."""
    return jsonify({
        'total_docs': N,
        'total_terms': len(df),
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
