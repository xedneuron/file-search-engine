import json
from typing import List, Tuple, Optional
import ollama

from search_engine import LocalFileSearch



class QueryModel:
    """Simple Ollama HTTP client that supports streaming generation.

    Expects an Ollama server at `host` (default http://localhost:11434).
    """
    def __init__(self, model: str = "llama2"):
        """Initialize the QueryModel with the specified model.
        
        Args:
            model: The name of the Ollama model to use (default: llama2)
        """
        self.model = model
    
    def generate_stream(self, prompt: str):
        """Stream the response from the model for the given prompt.
        
        Args:
            prompt: The input prompt to send to the model
            
        Yields:
            Streamed response chunks from the model
        """
        response = ollama.generate(model=self.model, prompt=prompt, stream=True)
        for chunk in response:
            yield chunk.get("response", "")
    
    def prompt(self, prompt: str) -> str:
        """Generate a complete response from the model for the given prompt.
        
        Args:
            prompt: The input prompt to send to the model
            
        Returns:
            The complete response from the model
        """
        response = ollama.generate(model=self.model, prompt=prompt, stream=False)
        return response.get("response", "")


class RAG:
    """Retrieval-Augmented Generation helper using `LocalFileSearch`.

    Args:
        search: An instance of LocalFileSearch for retrieving relevant documents
        model: An instance of QueryModel for generating responses
    
    - run_prompt is an iterable that yields each current step in the response generation,
        like "Searching Files...", "Checking Relevance...", etc. but will also eventually 
        yield the model's generated_stream iterable and an iterable of files used.

    - it will search for results, agentically check relevance, and then generate a response, yielding
        each step along the way.
    """
    def __init__(self, search: LocalFileSearch, model: QueryModel):
        self.search = search
        self.model = model
        self.history = []

        self.search.index_files()

    def run_prompt(self, query: str, top_k: int = 5):
        """Run a RAG pipeline that yields steps, files, and response stream.
        
        Args:
            query: The user's query
            top_k: Number of top results to retrieve
            
        Yields:
            Tuples of (step_description, relevant_files, response_stream or None)
        """
        # Step 1: Search for relevant files
        yield ("Searching files...", None, None)
        results = self.search.search(query, top_k=top_k)
        
        # Step 2: Extract unique files from results
        yield ("Extracting relevant files...", None, None)
        relevant_files = self._extract_files_from_results(results)
        
        # Step 3: Check relevance of retrieved documents
        yield ("Checking relevance...", None, None)
        relevant_docs = self._check_relevance(query, results)
        
        # Step 4: Build context from relevant documents
        yield ("Building context...", relevant_docs, None)
        context = self._build_context(relevant_docs)
        
        # Step 5: Generate response with streaming
        yield ("Generating response...", None, 
               self._generate_response(query, context))
        # response_stream = self._generate_response(query, context)
        
        
        # Update history
        self.history.append({
            "query": query,
            "files": relevant_files,
            "context": context
        })

    def _extract_files_from_results(self, results: List[Tuple]) -> List[str]:
        """Extract unique file paths from search results.
        
        Args:
            results: List of (file_path, line_content) tuples from search
            
        Returns:
            List of unique file paths
        """
        files = list(set(item["rel_path"] for item in results))   
        return sorted(files)

    def _check_relevance(self, query: str, results: List[dict]) -> List[dict]:
        """Use the model to check relevance of retrieved documents.
        
        Args:
            query: The user's query
            results: List of result dictionaries from search
            
        Returns:
            Filtered list of relevant documents
        """
        if not results:
            return []
        
        # Build a prompt to check relevance
        docs_text = "\n".join([f"- {result['content']}" for result in results])
        relevance_prompt = f"""Given the query: "{query}"
        
Below are retrieved documents. Mark each as RELEVANT or IRRELEVANT based on whether it helps answer the query.

Documents:
{docs_text}

For each document, output RELEVANT or IRRELEVANT on a new line."""
        
        # Get relevance judgment from model
        judgment = self.model.prompt(relevance_prompt)
        
        # Parse relevance judgments (simple approach: assume one judgment per line)
        judgments = judgment.strip().split('\n')
        relevant_results = []
        for i, result in enumerate(results):
            if i < len(judgments) and "RELEVANT" in judgments[i].upper():
                relevant_results.append(result)
        
        return relevant_results if relevant_results else results[:3]  # Fallback to top 3

    def _build_context(self, relevant_docs: List[dict]) -> str:
        """Build a context string from relevant documents.
        
        Args:
            relevant_docs: List of result dictionaries from search
            
        Returns:
            A formatted context string
        """
        if not relevant_docs:
            return ""
        
        context = "Context from relevant files:\n\n"
        for result in relevant_docs:
            context += f"From {result['rel_path']} (line {result['line']}):\n{result['content']}\n\n"
        
        return context

    def _generate_response(self, query: str, context: str):
        """Generate a response using the model with context.
        
        Args:
            query: The user's query
            context: The context from relevant documents
            
        Yields:
            Response chunks from the model stream
        """
        prompt = f"""{context}

Based on the context above, answer the following query:
{query}"""
        
        # Stream the response
        yield from self.model.generate_stream(prompt)

if __name__ == "__main__":
    # Example usage
    search = LocalFileSearch(root_dir="__data__", use_lsa=True)
    model = QueryModel(model="mistral:7b-instruct")
    rag = RAG(search, model)
    
    query = "So who is Tabby? Tell me what she's like."
    for step, files, response_stream in rag.run_prompt(query):
        print(step)
        if files is not None:
            print("Relevant files:", files)
        if response_stream is not None:
            print("Response:")
            for chunk in response_stream:
                print(chunk, end='', flush=True)
            print()