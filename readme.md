# Simple Local File Search (Flask + HTMX) 🔎

A lightweight Flask app that provides a fast, read-only search interface over a local `__data__` directory. Results are returned lazily and rendered quickly using HTMX + Jinja templates. Files can be viewed safely (no directory traversal) and the app is intended for small/local use and demos.

---

## Features ✅

- **Fast (TF-IDF) Mode**: Exact term matching with morphological stemming (fly/flying match)
- **Smart (LSA) Mode**: Semantic search with:
  - Latent Semantic Analysis (LSA) for topic-based matching
  - WordNet synonym expansion (e.g., search "cocoa" finds "chocolate")
  - Category refinement: results are re-ranked by semantic similarity to the query
- Lazily loaded results with HTMX for snappy UI and minimal JS
- File viewer with directory-traversal protections (read-only)
- Minimal dependencies — easy to run and extend

## Quick Start 🔧

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run the app:

```bash
# Enable LSA (optional) and set number of components, then run Flask
export LSA_ENABLED=1
export LSA_COMPONENTS=50
FLASK_APP=main.py FLASK_ENV=development flask run
```

3. Open your browser at http://127.0.0.1:5000 and search the `__data__` files.

## Usage & Structure 🗂️

- `__data__/` — sample text files used for searching (e.g. `sms.txt`, `coffee.txt`)
- `main.py` / `app.py` — Flask application entry points
- `search_engine.py` — search and indexing logic
- `templates/` — Jinja templates, including HTMX partials

Tips:
- The UI uses HTMX endpoints so results update without full page reloads.
- File viewing is read-only and sanitized against path traversal.

## Development Notes 🛠️

- Small, modular codebase — add more file parsers or change ranking in `search_engine.py`.
- Tests are not included but can be added easily (unit tests for search functions).

## Search Modes — Fast vs Smart ⚡🤖

The UI provides two modes to control the search behavior:

- **Fast (TF-IDF)** — Exact term matching with stemming. Fast and precise.
  - Matches: "fly" ↔ "flying" (via stem normalization)
  - Matches: "run" ↔ "running" ↔ "ran"
  - Does NOT match: "cocoa" ↔ "chocolate" (different terms)
  
- **Smart (LSA + Semantics)** — Intelligent semantic search. Slower but more aware.
  - Matches: all of the above, PLUS
  - Matches: "cocoa" ↔ "chocolate" (semantic synonymy via WordNet)
  - Matches: "flying" ↔ "airborne" ↔ "aviation" (related topics via LSA)
  - Results are re-ranked by semantic category to reduce false positives

**How Smart Mode Works:**
1. **Query Expansion**: Adds WordNet synonyms to the search query (e.g., "cocoa" → "cocoa cacao chocolate")
2. **LSA Matching**: Finds semantically related documents using latent topic analysis
3. **Category Refinement**: Boosts results that share semantic categories with the query (e.g., both food-related)

Toggle modes from the left panel—switching is instant (no reindex needed). You can also control LSA via environment variable `LSA_COMPONENTS` (default: 100).

## Security & Limitations ⚠️

- Intended for local use only. Do not expose to untrusted networks without additional hardening.
- It performs simple text search (no advanced ranking or tokenization).

## License

MIT — feel free to reuse and modify.

---

If you'd like, I can add a sample `docker-compose.yml`, example tests, or a small CONTRIBUTING section next.