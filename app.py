import os
import time
import urllib.parse
import uuid
from flask import Flask, render_template, request, abort, Response, stream_with_context
from search_engine import LocalFileSearch
from markupsafe import Markup

# --- FLASK APP ---
app = Flask(__name__)

# Configuration
DATA_DIR = "__data__"
LSA_COMPONENTS = int(os.getenv("LSA_COMPONENTS", "100"))

search_engine = LocalFileSearch(DATA_DIR, n_components=LSA_COMPONENTS)

# Store RAG pipeline state for each session
rag_sessions = {}

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

@app.route('/rag-query', methods=['POST'])
def rag_search():
    query = request.form.get('rag-query', '').strip()
    session_id = request.form.get('session-id', '')
    
    if len(query) < 2 and not session_id:
        return ""
    
    from query_ollama import QueryModel, RAG
    
    # Initialize new session if needed
    if not session_id:
        session_id = str(uuid.uuid4())
        model = QueryModel(model="mistral:7b-instruct")
        rag = RAG(search_engine, model)
        rag_sessions[session_id] = {
            'iterator': rag.run_prompt(query),
            'query': query,
            'response_stream': None,
            'files': None,
            'done': False
        }
    
    session = rag_sessions.get(session_id)
    if not session:
        return '<div class="text-red-600">Session expired</div>'
    
    # Get next step from the iterator
    try:
        step, files, response_stream = next(session['iterator'])
        
        # Check if this is the final step (has response_stream)
        is_final = response_stream is not None
        
        if is_final:
            session['response_stream'] = response_stream
            session['files'] = files
            session['done'] = True
        
        # Build the response HTML
        html = f'<div class="text-gray-600 text-sm p-2 bg-blue-50 rounded mb-2 border-l-4 border-blue-400">📍 {step}</div>'
        
        
        
        # If final step, add response streaming area and the hidden session ID for polling
        if is_final:
            html += f'''
            <div class="p-3 bg-white border rounded mt-3 mb-3">
                <strong class="text-gray-700">🤖 Response:</strong>
                <div class="text-gray-800 mt-2 whitespace-pre-wrap" id="rag-stream-{session_id}">
                </div>
            </div>
            <script>
                (function() {{
                    const stream = new EventSource('/rag-stream?session-id={session_id}');
                    const container = document.getElementById('rag-stream-{session_id}');
                    stream.onmessage = (event) => {{
                        try {{
                            const data = JSON.parse(event.data);
                            if (data.done) {{
                                stream.close();
                            }} else {{
                                container.textContent += data.chunk;
                            }}
                        }} catch (e) {{
                            console.error('Parse error:', e);
                        }}
                    }};
                    stream.onerror = () => {{
                        stream.close();
                    }};
                }})();
            </script>
            '''
            # If we have files, render them
            if files is not None and len(files) > 0:
                file_results = [{'file': f, 'rel_path': f} for f in files]
                html += render_template('rag_sources.html', results=file_results)

            # Clean up session after a delay
            def cleanup():
                time.sleep(60)
                rag_sessions.pop(session_id, None)
            
            import threading
            threading.Thread(target=cleanup, daemon=True).start()
        else:
            # Not final, tell HTMX to poll again
            html += f'''
            <div hx-post="/rag-query"
                 hx-vals='{{"rag-query": "{session['query']}", "session-id": "{session_id}"}}'
                 hx-target="#rag-results"
                 hx-swap="innerHTML"
                 hx-trigger="load delay:500ms"
                 hx-replace-url="false">
            </div>
            '''
        
        return html
        
    except StopIteration:
        session['done'] = True
        return '<div class="text-gray-600 p-2">✅ Pipeline complete</div>'
    except Exception as e:
        return f'<div class="p-3 bg-red-100 border border-red-400 rounded text-red-700">❌ Error: {str(e)}</div>'


@app.route('/rag-stream')
def rag_stream():
    """Stream the response from the RAG pipeline."""
    session_id = request.args.get('session-id', '')
    session = rag_sessions.get(session_id)
    
    if not session or not session['response_stream']:
        return 'data: {"done": true}\n\n'
    
    def generate():
        import json
        try:
            for chunk in session['response_stream']:
                # Properly escape and JSON encode the chunk
                data = json.dumps({"chunk": chunk, "done": False})
                yield f'data: {data}\n\n'
                time.sleep(0.01)  # Small delay to allow streaming feel
        except Exception as e:
            data = json.dumps({"chunk": f"Error: {str(e)}", "done": True})
            yield f'data: {data}\n\n'
        finally:
            yield 'data: {"done": true}\n\n'
    
    return Response(generate(), mimetype='text/event-stream')

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
