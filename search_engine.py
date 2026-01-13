import os
import numpy as np
import re
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet
import nltk

warnings.filterwarnings('ignore', category=UserWarning)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

class LocalFileSearch:
    """Local file search with TF-IDF and LSA (Truncated SVD) support.

    - Always indexes both TF-IDF and LSA matrices.
    - Uses stemming to handle word morphology (fly/flying/flew are the same stem).
    - Expands queries with WordNet synonyms for semantic awareness.
    - Refines results by semantic category (WordNet synsets).
    - `use_lsa` flag controls which space is used for search (instant toggle, no reindex needed).
    """
    def __init__(self, root_dir, valid_extensions=None, use_lsa=False, n_components=30):
        self.root_dir = os.path.abspath(root_dir)
        self.valid_extensions = valid_extensions or ['.txt', '.md', '.py', '.json', '.csv', '.log']
        self.documents = []
        self.metadata = []
        
        # Stemmer for morphological normalization
        self.stemmer = SnowballStemmer('english')
        
        # Use simple TF-IDF with stemming support via custom tokenizer
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda text: [self.stemmer.stem(token) 
                                   for token in re.findall(r'\b\w+\b', text.lower()) 
                                   if len(token) > 2],
            stop_words='english',
            min_df=1,
            max_df=0.95,
            max_features=500,
            lowercase=True
        )
        self.tfidf_matrix = None
        self.tfidf_reduced = None
        self.svd_pipeline = None
        self.document_clusters = None  # For semantic category refinement

        # LSA configuration: moderate components for good semantic compression
        self.n_components = int(n_components) if n_components else 30
        
        # Flag to control search behavior (Fast = TF-IDF only, Smart = LSA)
        self.use_lsa = use_lsa

    def _is_valid_file(self, filename):
        return any(filename.endswith(ext) for ext in self.valid_extensions)

    def _get_synonyms(self, word):
        """Get synonyms of a word using WordNet."""
        synonyms = set()
        try:
            for synset in wordnet.synsets(word):
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower():
                        synonyms.add(self.stemmer.stem(synonym))
        except Exception:
            pass
        return synonyms

    def _expand_query(self, query):
        """Expand query with synonyms for semantic awareness."""
        tokens = re.findall(r'\b\w+\b', query.lower())
        expanded = set(tokens)
        
        # Add synonyms for each token
        for token in tokens:
            if len(token) > 3:  # Only expand longer words
                synonyms = self._get_synonyms(token)
                expanded.update(synonyms)
        
        return ' '.join(expanded)

    def _get_result_category(self, result_text):
        """Get WordNet synsets (categories) for a result."""
        tokens = re.findall(r'\b\w+\b', result_text.lower())
        categories = set()
        for token in tokens[:5]:  # Sample first 5 tokens
            if len(token) > 3:
                try:
                    synsets = wordnet.synsets(token)
                    for synset in synsets[:1]:  # Take the most common synset
                        categories.add(synset.lexname())  # e.g., 'noun.food'
                except Exception:
                    pass
        return categories

    def _refine_results_by_category(self, results, query_categories, threshold_boost=0.15):
        """Refine results: boost scores for results in same semantic category as query."""
        if not query_categories:
            return results
        
        refined = []
        for result in results:
            result_cats = self._get_result_category(result['content'])
            # Check if result shares a category with the query
            if result_cats & query_categories:  # intersection
                result['score'] *= (1 + threshold_boost)  # Boost score
            refined.append(result)
        
        # Re-sort by boosted scores
        refined.sort(key=lambda x: x['score'], reverse=True)
        return refined

    def index_files(self):
        """Walk directory, collect text lines, and build both TF-IDF and LSA matrices unconditionally."""
        print(f"Indexing files in {self.root_dir}...")
        self.documents = []
        self.metadata = []

        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if self._is_valid_file(filename):
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

        if not self.documents:
            print("No valid text data found to index.")
            return

        # Always build TF-IDF matrix
        print(f"Vectorizing {len(self.documents)} lines with TF-IDF...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

        # Always fit LSA on the TF-IDF matrix
        max_components = max(1, min(self.n_components, self.tfidf_matrix.shape[1], self.tfidf_matrix.shape[0]))
        try:
            print(f"Fitting LSA (n_components={max_components})...")
            svd = TruncatedSVD(n_components=max_components, random_state=42)
            self.svd_pipeline = make_pipeline(svd, Normalizer(copy=False))
            self.tfidf_reduced = self.svd_pipeline.fit_transform(self.tfidf_matrix)
            print("LSA fit complete.")
        except Exception as e:
            print(f"LSA fitting failed: {e}")
            self.tfidf_reduced = None
            self.svd_pipeline = None

        print(f"Indexing complete. Both TF-IDF and LSA ready.")

    def search(self, query, top_k=5, threshold=None):
        """Search using either TF-IDF (Fast) or LSA (Smart) space based on self.use_lsa flag.
        
        For Smart mode: expands query with synonyms and refines by semantic category.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            threshold: Minimum similarity score. If None, uses mode-specific defaults:
                      - LSA (Smart): 0.4 (semantic search, boosted by category)
                      - TF-IDF (Fast): 0.1 (strict term matching with stemming)
        """
        if self.tfidf_matrix is None:
            print("Index is empty. Run index_files() first.")
            return []

        # Use mode-specific default thresholds if not provided
        if threshold is None:
            threshold = 0.4 if self.use_lsa else 0.1

        try:
            # For Smart mode: expand query with synonyms
            search_query = query
            query_categories = set()
            if self.use_lsa:
                search_query = self._expand_query(query)
                # Get categories for the original query
                query_tokens = re.findall(r'\b\w+\b', query.lower())
                for token in query_tokens:
                    if len(token) > 3:
                        try:
                            synsets = wordnet.synsets(token)
                            if synsets:
                                query_categories.add(synsets[0].lexname())
                        except Exception:
                            pass

            query_vec = self.vectorizer.transform([search_query])

            # Use LSA or TF-IDF space based on the flag
            if self.use_lsa and (self.tfidf_reduced is not None) and (self.svd_pipeline is not None):
                query_reduced = self.svd_pipeline.transform(query_vec)
                cosine_similarities = cosine_similarity(query_reduced, self.tfidf_reduced).flatten()
            else:
                cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

            related_docs_indices = cosine_similarities.argsort()[:-top_k-1:-1]

            results = []
            for index in related_docs_indices:
                score = float(cosine_similarities[index])
                if score > threshold:
                    md = self.metadata[index]
                    results.append({
                        'score': score,
                        'file': md['file'],
                        'rel_path': md['rel_path'],
                        'line': md['line'],
                        'content': md['content']
                    })

            # Refine results by semantic category for Smart mode
            if self.use_lsa and query_categories:
                results = self._refine_results_by_category(results, query_categories, threshold_boost=0.2)

            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []



if __name__ == "__main__":
    TARGET_DIR = "__data__"

    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        with open(os.path.join(TARGET_DIR, "test.txt"), "w") as f:
            f.write("Python is a great language for data science.\n")
            f.write("Machine learning uses vectorization.\n")
            f.write("Search engines use cosine similarity.\n")
        print(f"Created {TARGET_DIR} with dummy data.")

    engine = LocalFileSearch(TARGET_DIR, use_lsa=False, n_components=30)
    engine.index_files()

    print("\n--- Search Engine Ready (Type 'exit' to quit) ---")
    while True:
        user_query = input("\nEnter search query: ")
        if user_query.lower() in ['exit', 'quit']:
            break

        matches = engine.search(user_query, top_k=5)

        if matches:
            print(f"\nFound {len(matches)} matches:")
            print("-" * 60)
            for match in matches:
                relevance = match['score'] * 100
                print(f"File: {match['file']} (Line {match['line']})")
                print(f"Relevance: {relevance:.1f}%")
                print(f"Content: \"{match['content']}\"")
                print("-" * 60)
        else:
            print("No matches found.")
