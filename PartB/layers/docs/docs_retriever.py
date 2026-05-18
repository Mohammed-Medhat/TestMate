# docs_retriever.py - Layer 1: Documentation Retriever (RAGFix Paper)
"""
External Knowledge Layer from RAGFix (NSF/IEEE 2025).

Retrieves:
- Official API documentation for target repositories
- Stack Overflow answers for similar errors
- Migration guides and release notes

Uses FAISS for semantic search over documentation embeddings.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import json
import os

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-cpu")

# ============================================================================
# TARGET REPOSITORIES (SWE-bench Lite)
# ============================================================================

TARGET_REPOS = [
    "astropy", "django", "flask", "matplotlib", "pylint",
    "pytest", "requests", "scikit-learn", "seaborn", "sphinx", "sympy"
]

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DocChunk:
    """A chunk of documentation."""
    chunk_id: str
    repo: str
    source: str  # "api_docs", "stackoverflow", "migration_guide"
    title: str
    content: str
    url: Optional[str] = None
    embedding: Optional[List[float]] = None

@dataclass 
class RetrievalResult:
    """Result from documentation search."""
    chunk: DocChunk
    score: float
    relevance: str  # "high", "medium", "low"

# ============================================================================
# DOCUMENTATION RETRIEVER
# ============================================================================

class DocsRetriever:
    """
    Layer 1: Documentation Retriever (RAGFix).
    
    Retrieves external knowledge for bug repair:
    - API documentation
    - Stack Overflow solutions
    - Migration guides
    """
    
    def __init__(self, index_path: str = "docs_index"):
        self.index_path = index_path
        self.chunks: Dict[str, DocChunk] = {}
        self.embeddings = []
        self.chunk_ids = []
        self.index = None
        self.embed_model = None
        
    def _get_embedder(self):
        """Lazy load embedding model."""
        if self.embed_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                print("Warning: sentence-transformers not available")
                return None
        return self.embed_model
    
    def add_chunk(self, chunk: DocChunk):
        """Add documentation chunk to index."""
        self.chunks[chunk.chunk_id] = chunk
        
        # Generate embedding
        embedder = self._get_embedder()
        if embedder and chunk.embedding is None:
            chunk.embedding = embedder.encode(chunk.content).tolist()
        
        if chunk.embedding:
            self.embeddings.append(chunk.embedding)
            self.chunk_ids.append(chunk.chunk_id)
    
    def build_index(self):
        """Build FAISS index from embeddings."""
        if not FAISS_AVAILABLE or not self.embeddings:
            return
        
        embeddings_array = np.array(self.embeddings, dtype='float32')
        dim = embeddings_array.shape[1]
        
        # Use L2 distance index
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings_array)
        
        print(f"Built FAISS index with {len(self.embeddings)} chunks")
    
    def retrieve(self, query: str, top_k: int = 5, repo_filter: str = None) -> List[RetrievalResult]:
        """
        Retrieve relevant documentation for a query.
        
        Args:
            query: The search query (e.g., error message, API name)
            top_k: Number of results to return
            repo_filter: Only return docs from this repo
            
        Returns:
            List of RetrievalResult sorted by relevance
        """
        embedder = self._get_embedder()
        if not embedder or not FAISS_AVAILABLE or self.index is None:
            return []
        
        # Encode query
        query_embedding = embedder.encode(query).astype('float32').reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k * 2, len(self.chunk_ids)))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
                
            chunk_id = self.chunk_ids[idx]
            chunk = self.chunks.get(chunk_id)
            
            if chunk is None:
                continue
            
            # Apply repo filter
            if repo_filter and chunk.repo != repo_filter:
                continue
            
            # Convert distance to similarity score
            score = 1.0 / (1.0 + dist)
            
            # Determine relevance
            if score > 0.7:
                relevance = "high"
            elif score > 0.5:
                relevance = "medium"
            else:
                relevance = "low"
            
            results.append(RetrievalResult(chunk=chunk, score=score, relevance=relevance))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def retrieve_for_error(self, error_message: str, repo: str) -> List[RetrievalResult]:
        """Retrieve docs specifically for an error message."""
        # Clean error message
        clean_error = error_message.split('\n')[0][:200]
        return self.retrieve(f"{repo} {clean_error}", top_k=3, repo_filter=repo)
    
    def retrieve_for_api(self, api_name: str, repo: str) -> List[RetrievalResult]:
        """Retrieve API documentation."""
        return self.retrieve(f"{repo} API {api_name}", top_k=3, repo_filter=repo)
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def save(self, path: str = None):
        """Save index to disk."""
        path = path or self.index_path
        os.makedirs(path, exist_ok=True)
        
        # Save chunks
        chunks_data = {
            cid: {
                'chunk_id': c.chunk_id,
                'repo': c.repo,
                'source': c.source,
                'title': c.title,
                'content': c.content,
                'url': c.url
            }
            for cid, c in self.chunks.items()
        }
        with open(f"{path}/chunks.json", 'w') as f:
            json.dump(chunks_data, f)
        
        # Save embeddings
        if self.embeddings:
            np.save(f"{path}/embeddings.npy", np.array(self.embeddings))
        
        # Save chunk IDs
        with open(f"{path}/chunk_ids.json", 'w') as f:
            json.dump(self.chunk_ids, f)
        
        # Save FAISS index
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, f"{path}/faiss.index")
        
        print(f"Saved docs index to {path}")
    
    def load(self, path: str = None):
        """Load index from disk."""
        path = path or self.index_path
        
        if not os.path.exists(path):
            print(f"Index path {path} does not exist")
            return
        
        # Load chunks
        with open(f"{path}/chunks.json", 'r') as f:
            chunks_data = json.load(f)
        
        for cid, data in chunks_data.items():
            self.chunks[cid] = DocChunk(**data)
        
        # Load embeddings
        if os.path.exists(f"{path}/embeddings.npy"):
            self.embeddings = np.load(f"{path}/embeddings.npy").tolist()
        
        # Load chunk IDs
        with open(f"{path}/chunk_ids.json", 'r') as f:
            self.chunk_ids = json.load(f)
        
        # Load FAISS index
        if FAISS_AVAILABLE and os.path.exists(f"{path}/faiss.index"):
            self.index = faiss.read_index(f"{path}/faiss.index")
        
        print(f"Loaded docs index: {len(self.chunks)} chunks")

# ============================================================================
# DOCUMENTATION INGESTER
# ============================================================================

class DocsIngester:
    """Ingests documentation from various sources."""
    
    def __init__(self, retriever: DocsRetriever):
        self.retriever = retriever
        self.chunk_counter = 0
    
    def ingest_markdown(self, content: str, repo: str, source: str, title: str, 
                        chunk_size: int = 500):
        """Ingest markdown documentation."""
        # Simple chunking by paragraphs
        paragraphs = content.split('\n\n')
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if current_size + len(para) > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                self._add_chunk(chunk_text, repo, source, title)
                current_chunk = []
                current_size = 0
            
            current_chunk.append(para)
            current_size += len(para)
        
        # Save remaining
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            self._add_chunk(chunk_text, repo, source, title)
    
    def ingest_stackoverflow(self, question: str, answer: str, repo: str, tags: List[str]):
        """Ingest Stack Overflow Q&A."""
        content = f"Question: {question}\n\nAnswer: {answer}"
        self._add_chunk(content, repo, "stackoverflow", f"SO: {tags[0] if tags else 'python'}")
    
    def _add_chunk(self, content: str, repo: str, source: str, title: str):
        """Add a chunk to the retriever."""
        self.chunk_counter += 1
        chunk = DocChunk(
            chunk_id=f"{repo}_{source}_{self.chunk_counter}",
            repo=repo,
            source=source,
            title=title,
            content=content
        )
        self.retriever.add_chunk(chunk)

# ============================================================================
# UTILITY
# ============================================================================

def create_demo_index() -> DocsRetriever:
    """Create a demo index with sample documentation."""
    retriever = DocsRetriever()
    ingester = DocsIngester(retriever)
    
    # Sample Django docs
    ingester.ingest_markdown(
        """
        # Django QuerySet API Reference
        
        ## filter()
        Returns a new QuerySet containing objects that match the given lookup parameters.
        
        The lookup parameters should be in the format described in Field lookups.
        Multiple parameters are joined via AND in the underlying SQL statement.
        
        ## exclude()
        Returns a new QuerySet containing objects that do not match the given lookup parameters.
        """,
        repo="django",
        source="api_docs",
        title="QuerySet API"
    )
    
    # Sample StackOverflow
    ingester.ingest_stackoverflow(
        question="Django filter with empty list raises error",
        answer="When using __in with an empty list, Django returns an empty QuerySet. "
               "Make sure to check if the list is empty before filtering.",
        repo="django",
        tags=["django", "queryset", "filter"]
    )
    
    retriever.build_index()
    return retriever

# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    print("Creating demo docs index...")
    retriever = create_demo_index()
    
    # Test retrieval
    results = retriever.retrieve("Django QuerySet filter empty list", top_k=3)
    print(f"\nFound {len(results)} results:")
    for r in results:
        print(f"  [{r.relevance}] {r.chunk.title}: {r.chunk.content[:100]}...")
