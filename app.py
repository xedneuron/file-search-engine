import os
import time
import urllib.parse
from flask import Flask, render_template, request, abort
from search_engine import LocalFileSearch
from markupsafe import Markup

# --- FLASK APP ---
app = Flask(__name__)

# Configuration
DATA_DIR = "__data__"
LSA_COMPONENTS = int(os.getenv("LSA_COMPONENTS", "100"))

search_engine = LocalFileSearch(DATA_DIR, n_components=LSA_COMPONENTS)

# Ensure data dir exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    # Create a dummy file so the index isn't empty on first run
    with open(os.path.join(DATA_DIR, "welcome.txt"), "w") as f:
        f.write("Welcome to the local search engine!\nSearch for text in this directory.\n")

# Index immediately on startup (builds both TF-IDF and LSA)
search_engine.index_files()

@app.route('/')
def index():
    # Provide current mode to the template so the UI can reflect it
    return render_template('index.html', current_mode=search_engine.use_lsa, components=search_engine.n_components)

@app.route('/mode/data-search')
def mode_data_search():
    return render_template('_data_search.html')

@app.route('/mode/rag')
def mode_rag():
    return render_template('_rag.html')

@app.route('/rag-search', methods=['POST'])
def rag_search():
    # Your RAG search implementation here
    pass

@app.route('/reindex', methods=['POST'])
def reindex():
    """Toggle between Fast (TF-IDF) and Smart (LSA) modes, optionally rebuild index on demand."""
    req_json = request.get_json(silent=True)
    
    # Check if user wants to rebuild the index (e.g., new files added)
    rebuild = False
    if req_json:
        use_lsa = req_json.get('use_lsa')
        rebuild = req_json.get('rebuild', False)
    else:
        use_lsa = request.form.get('use_lsa')
        rebuild = request.form.get('rebuild', False)
    
    # Toggle use_lsa flag (instant, no reindex needed)
    if use_lsa is not None:
        search_engine.use_lsa = str(use_lsa).lower() in ("1", "true", "yes", "on")
    
    # Optionally rebuild the index if data changed
    if rebuild:
        search_engine.index_files()
    
    mode_name = "Smart (LSA)" if search_engine.use_lsa else "Fast (TF-IDF)"
    return render_template('_mode_controls.html', current_mode=search_engine.use_lsa, components=search_engine.n_components)

@app.route('/search', methods=['POST'])
def search_route():
    query = request.form.get('query', '')
    if len(query) < 2:
        return ""
    
    results = search_engine.search(query, top_k=10)
    # Convert scores to percentage for template display
    for result in results:
        result['score'] = int(result['score'] * 100)
        # Provide a URL-safe, encoded version of the matched content so
        # `hx-get` attributes won't break when `res.content` contains
        # spaces, quotes, or other special characters.
        try:
            result['line_param'] = urllib.parse.quote(result.get('content', ''), safe='')
        except Exception:
            result['line_param'] = ''

    return render_template('results.html', results=results)

@app.route('/view', methods=['GET'])
def view_file():
    # Security: Prevent traversing out of the DATA_DIR
    req_path = request.args.get('path')
    if not req_path:
        return "No file specified."
    
    req_line = request.args.get('line')
    print(f"Viewing file: {req_path} with highlight: {req_line if req_line else 'None'}")
    if not req_line:
        return "No line specified."

    abs_path = os.path.abspath(req_path)
    if not abs_path.startswith(search_engine.root_dir):
        return abort(403, "Access Denied: Cannot view files outside data directory.")

    try:
        import re
        with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
            read_text = f.read()
        
        # Escape HTML later
        escaped_text = str(Markup.escape(read_text))
        
        # Build regex pattern from the search term and highlight matches
        pattern = re.escape(str(Markup.escape(req_line)))
        highlighted = re.sub(
            f'({pattern})',
            r'<mark class="scroll-target bg-yellow-200">\1</mark>',
            escaped_text,
            flags=re.IGNORECASE
        )
        
        # Wrap in Markup so Jinja2 doesn't escape the <mark> tags
        content = Markup(highlighted)
        return render_template('viewer.html', content=content, filename=os.path.basename(abs_path))
    except Exception as e:
        return f"Error reading file: {e}"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
