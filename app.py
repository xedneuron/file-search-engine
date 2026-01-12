import os
import time
from flask import Flask, render_template, request, abort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- SEARCH ENGINE LOGIC (Same as before) ---
class LocalFileSearch:
    def __init__(self, root_dir):
        self.root_dir = os.path.abspath(root_dir)
        self.documents = []
        self.metadata = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.is_indexed = False

    def index_files(self):
        print(f"Indexing files in {self.root_dir}...")
        self.documents = []
        self.metadata = []
        
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if filename.endswith(('.txt', '.md', '.py', '.json', '.csv', '.log')):
                    filepath = os.path.join(dirpath, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                            for line_num, line_content in enumerate(lines):
                                clean_line = line_content.strip()
                                if len(clean_line) > 2:
                                    self.documents.append(clean_line)
                                    self.metadata.append({
                                        'file': filepath,
                                        'rel_path': os.path.relpath(filepath, self.root_dir),
                                        'line': line_num + 1,
                                        'content': clean_line
                                    })
                    except Exception:
                        pass
        
        if self.documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            self.is_indexed = True
            print(f"Indexed {len(self.documents)} lines.")
        else:
            print("No documents found.")

    def search(self, query, top_k=10, threshold=0.1):
        if not self.is_indexed: return []
        
        try:
            query_vec = self.vectorizer.transform([query])
            cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            related_docs_indices = cosine_similarities.argsort()[:-top_k-1:-1]
            
            results = []
            for index in related_docs_indices:
                score = cosine_similarities[index]
                if score > threshold:
                    match = self.metadata[index]
                    results.append({
                        'score': int(score * 100),
                        'file': match['file'],
                        'rel_path': match['rel_path'],
                        'line': match['line'],
                        'content': match['content']
                    })
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []

# --- FLASK APP ---
app = Flask(__name__)

# Configuration
DATA_DIR = "__data__"
search_engine = LocalFileSearch(DATA_DIR)

# Ensure data dir exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    # Create a dummy file so the index isn't empty on first run
    with open(os.path.join(DATA_DIR, "welcome.txt"), "w") as f:
        f.write("Welcome to the local search engine!\nSearch for text in this directory.\n")

# Index immediately on startup
search_engine.index_files()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_route():
    query = request.form.get('query', '')
    if len(query) < 2:
        return "" # Return nothing if query is too short
    
    results = search_engine.search(query)
    return render_template('results.html', results=results)

@app.route('/view', methods=['GET'])
def view_file():
    # Security: Prevent traversing out of the DATA_DIR
    req_path = request.args.get('path')
    if not req_path:
        return "No file specified."
    
    abs_path = os.path.abspath(req_path)
    if not abs_path.startswith(search_engine.root_dir):
        return abort(403, "Access Denied: Cannot view files outside data directory.")

    try:
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return render_template('viewer.html', content=content, filename=os.path.basename(abs_path))
    except Exception as e:
        return f"Error reading file: {e}"

@app.route('/reindex', methods=['POST'])
def reindex():
    search_engine.index_files()
    return "<span class='text-success'>Index Updated!</span>"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
