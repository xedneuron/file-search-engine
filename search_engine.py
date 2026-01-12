import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LocalFileSearch:
    def __init__(self, root_dir, valid_extensions=None):
        """
        Initialize the search engine.
        :param root_dir: The directory to search in.
        :param valid_extensions: A list of file extensions to include (e.g., ['.txt', '.md']).
                                 If None, defaults to common text formats.
        """
        self.root_dir = root_dir
        self.valid_extensions = valid_extensions or ['.txt', '.md', '.py', '.json', '.csv', '.log']
        self.documents = []  # Stores the actual text of the lines
        self.metadata = []   # Stores (filepath, line_number)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None

    def _is_valid_file(self, filename):
        return any(filename.endswith(ext) for ext in self.valid_extensions)

    def index_files(self):
        """
        Walks through the directory, reads files, and builds the TF-IDF index.
        """
        print(f"Indexing files in {self.root_dir}...")
        
        # 1. Collect all lines from files
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                if self._is_valid_file(filename):
                    filepath = os.path.join(dirpath, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                            for line_num, line_content in enumerate(lines):
                                clean_line = line_content.strip()
                                # Skip empty lines or very short lines to reduce noise
                                if len(clean_line) > 2:
                                    self.documents.append(clean_line)
                                    self.metadata.append({
                                        'file': filepath,
                                        'line_num': line_num + 1,
                                        'content': clean_line
                                    })
                    except Exception as e:
                        print(f"Skipping {filepath} due to error: {e}")

        if not self.documents:
            print("No valid text data found to index.")
            return

        # 2. Vectorize the documents (lines)
        # This converts text into numerical vectors based on word importance
        print(f"Vectorizing {len(self.documents)} lines...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        print("Indexing complete.")

    def search(self, query, top_k=5, threshold=0.1):
        """
        Searches the index for the query.
        :param query: The search string.
        :param top_k: Number of top results to return.
        :param threshold: Minimum similarity score (0 to 1) to be considered a match.
        """
        if self.tfidf_matrix is None:
            print("Index is empty. Run index_files() first.")
            return []

        # Convert query to the same vector space
        query_vec = self.vectorizer.transform([query])

        # Calculate cosine similarity between query and all lines
        # flatten() is used to convert the result into a 1D array
        cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get indices of the top_k most similar lines
        # argsort returns indices that would sort the array, we take the last k and reverse them
        related_docs_indices = cosine_similarities.argsort()[:-top_k-1:-1]

        results = []
        for index in related_docs_indices:
            score = cosine_similarities[index]
            if score > threshold:
                result_data = self.metadata[index]
                results.append({
                    'score': score,
                    'file': result_data['file'],
                    'line': result_data['line_num'],
                    'content': result_data['content']
                })

        return results

# --- Main Execution ---
if __name__ == "__main__":
    # Configuration
    TARGET_DIR = "__data__"
    
    # Create dummy data if directory doesn't exist (for testing purposes)
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        with open(os.path.join(TARGET_DIR, "test.txt"), "w") as f:
            f.write("Python is a great language for data science.\n")
            f.write("Machine learning uses vectorization.\n")
            f.write("Search engines use cosine similarity.\n")
        print(f"Created {TARGET_DIR} with dummy data.")

    # Initialize and Run
    engine = LocalFileSearch(TARGET_DIR)
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
                # Calculate relevance percentage
                relevance = match['score'] * 100
                print(f"File: {match['file']} (Line {match['line']})")
                print(f"Relevance: {relevance:.1f}%")
                print(f"Content: \"{match['content']}\"")
                print("-" * 60)
        else:
            print("No matches found.")
