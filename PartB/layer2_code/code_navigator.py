# code_navigator.py - Layer 2: Codebase Navigator (KGCompass Paper)
"""
Internal Knowledge Layer from KGCompass (arXiv 2025).

Navigates repository structure using:
- BM25 sparse retrieval for keyword matching
- Vector embeddings for semantic similarity
- 2-hop graph traversal for related code

Integrates with graph_rag.py for the full KGCompass pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import os
import re

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CodeLocation:
    """A location in the codebase."""
    file_path: str
    start_line: int
    end_line: int
    content: str
    symbol_type: str  # "function", "class", "method", "variable"
    symbol_name: str
    
@dataclass
class SearchResult:
    """Result from code search."""
    location: CodeLocation
    score: float
    match_type: str  # "bm25", "semantic", "graph"

# ============================================================================
# BM25 IMPLEMENTATION (Lightweight)
# ============================================================================

class SimpleBM25:
    """Lightweight BM25 implementation for code search."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []
        self.doc_metadata: List[Dict] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0
        self.term_freqs: Dict[str, Dict[int, int]] = {}  # term -> {doc_id: freq}
        self.doc_freqs: Dict[str, int] = {}  # term -> num_docs_containing
        self.n_docs: int = 0
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Convert camelCase and snake_case
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = text.replace('_', ' ')
        # Lowercase and split
        tokens = text.lower().split()
        # Remove short tokens
        return [t for t in tokens if len(t) > 1]
    
    def add_document(self, content: str, metadata: Dict = None):
        """Add a document to the index."""
        doc_id = len(self.documents)
        self.documents.append(content)
        self.doc_metadata.append(metadata or {})
        
        tokens = self._tokenize(content)
        self.doc_lengths.append(len(tokens))
        
        # Count term frequencies
        term_counts = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1
        
        for term, freq in term_counts.items():
            if term not in self.term_freqs:
                self.term_freqs[term] = {}
            self.term_freqs[term][doc_id] = freq
            
            if term not in self.doc_freqs:
                self.doc_freqs[term] = 0
            self.doc_freqs[term] += 1
        
        self.n_docs = len(self.documents)
        self.avg_doc_length = sum(self.doc_lengths) / self.n_docs if self.n_docs > 0 else 0
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search for documents matching query."""
        import math
        
        query_tokens = self._tokenize(query)
        scores = {}
        
        for token in query_tokens:
            if token not in self.term_freqs:
                continue
            
            df = self.doc_freqs.get(token, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            
            for doc_id, tf in self.term_freqs[token].items():
                doc_len = self.doc_lengths[doc_id]
                
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                score = idf * numerator / denominator
                
                scores[doc_id] = scores.get(doc_id, 0) + score
        
        # Sort by score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

# ============================================================================
# CODE NAVIGATOR
# ============================================================================

class CodeNavigator:
    """
    Layer 2: Codebase Navigator (KGCompass).
    
    Provides:
    - BM25 keyword search
    - Semantic vector search
    - Symbol lookup
    - File tree navigation
    """
    
    def __init__(self, repo_path: str = None):
        self.repo_path = Path(repo_path) if repo_path else None
        self.bm25 = SimpleBM25()
        self.locations: Dict[int, CodeLocation] = {}
        self.file_tree: Dict[str, List[str]] = {}  # dir -> files
        self.symbols: Dict[str, CodeLocation] = {}  # symbol_name -> location
        
    def index_repository(self, repo_path: str):
        """Index all Python files in a repository."""
        self.repo_path = Path(repo_path)
        
        python_files = list(self.repo_path.rglob("*.py"))
        print(f"Indexing {len(python_files)} Python files...")
        
        for file_path in python_files:
            try:
                self._index_file(file_path)
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")
        
        print(f"Indexed {len(self.locations)} code locations")
        print(f"Found {len(self.symbols)} symbols")
    
    def _index_file(self, file_path: Path):
        """Index a single Python file."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except:
            return
        
        rel_path = str(file_path.relative_to(self.repo_path))
        
        # Update file tree
        dir_path = str(file_path.parent.relative_to(self.repo_path))
        if dir_path not in self.file_tree:
            self.file_tree[dir_path] = []
        self.file_tree[dir_path].append(rel_path)
        
        # Extract functions and classes
        self._extract_symbols(content, rel_path)
    
    def _extract_symbols(self, content: str, file_path: str):
        """Extract function and class definitions."""
        lines = content.split('\n')
        
        # Simple regex patterns for Python
        func_pattern = re.compile(r'^(\s*)def\s+(\w+)\s*\(')
        class_pattern = re.compile(r'^(\s*)class\s+(\w+)')
        
        current_class = None
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check for class
            class_match = class_pattern.match(line)
            if class_match:
                indent, class_name = class_match.groups()
                current_class = class_name if len(indent) == 0 else None
                
                # Find class body end (simplified)
                end_line = self._find_block_end(lines, i)
                class_content = '\n'.join(lines[i:end_line])
                
                location = CodeLocation(
                    file_path=file_path,
                    start_line=i + 1,
                    end_line=end_line,
                    content=class_content[:1000],
                    symbol_type="class",
                    symbol_name=class_name
                )
                
                doc_id = len(self.locations)
                self.locations[doc_id] = location
                self.symbols[class_name] = location
                self.bm25.add_document(class_content, {'doc_id': doc_id})
            
            # Check for function
            func_match = func_pattern.match(line)
            if func_match:
                indent, func_name = func_match.groups()
                
                # Find function body end
                end_line = self._find_block_end(lines, i)
                func_content = '\n'.join(lines[i:end_line])
                
                symbol_type = "method" if current_class and len(indent) > 0 else "function"
                full_name = f"{current_class}.{func_name}" if current_class and symbol_type == "method" else func_name
                
                location = CodeLocation(
                    file_path=file_path,
                    start_line=i + 1,
                    end_line=end_line,
                    content=func_content[:1000],
                    symbol_type=symbol_type,
                    symbol_name=full_name
                )
                
                doc_id = len(self.locations)
                self.locations[doc_id] = location
                self.symbols[full_name] = location
                self.bm25.add_document(func_content, {'doc_id': doc_id})
            
            i += 1
    
    def _find_block_end(self, lines: List[str], start: int) -> int:
        """Find the end of a Python block (function/class)."""
        if start >= len(lines):
            return start + 1
        
        # Get starting indent
        first_line = lines[start]
        start_indent = len(first_line) - len(first_line.lstrip())
        
        for i in range(start + 1, min(start + 200, len(lines))):
            line = lines[i]
            if not line.strip():
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= start_indent and line.strip():
                return i
        
        return min(start + 50, len(lines))
    
    # ========================================================================
    # SEARCH METHODS
    # ========================================================================
    
    def search_bm25(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search using BM25 keyword matching."""
        results = []
        
        for doc_id, score in self.bm25.search(query, top_k):
            if doc_id in self.locations:
                results.append(SearchResult(
                    location=self.locations[doc_id],
                    score=score,
                    match_type="bm25"
                ))
        
        return results
    
    def search_symbol(self, symbol_name: str) -> Optional[CodeLocation]:
        """Find a symbol by exact name."""
        return self.symbols.get(symbol_name)
    
    def search_file(self, file_pattern: str) -> List[str]:
        """Find files matching a pattern."""
        matches = []
        pattern = file_pattern.lower()
        
        for dir_path, files in self.file_tree.items():
            for file_path in files:
                if pattern in file_path.lower():
                    matches.append(file_path)
        
        return matches[:20]
    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """Get content of a specific file."""
        if not self.repo_path:
            return None
        
        full_path = self.repo_path / file_path
        if full_path.exists():
            try:
                return full_path.read_text(encoding='utf-8', errors='ignore')
            except:
                return None
        return None
    
    # ========================================================================
    # REPO MAP (Compressed Tree View)
    # ========================================================================
    
    def get_repo_map(self, max_files: int = 50) -> str:
        """Generate a compressed tree view of the repository."""
        if not self.file_tree:
            return "No repository indexed."
        
        lines = ["Repository Structure:"]
        file_count = 0
        
        for dir_path in sorted(self.file_tree.keys()):
            if file_count >= max_files:
                lines.append(f"... and {sum(len(f) for f in self.file_tree.values()) - file_count} more files")
                break
            
            files = self.file_tree[dir_path]
            lines.append(f"\n{dir_path}/")
            
            for file_path in sorted(files)[:5]:
                lines.append(f"  {Path(file_path).name}")
                file_count += 1
            
            if len(files) > 5:
                lines.append(f"  ... and {len(files) - 5} more files")
        
        return '\n'.join(lines)
    
    def get_stats(self) -> Dict:
        """Get indexing statistics."""
        return {
            'files': sum(len(f) for f in self.file_tree.values()),
            'directories': len(self.file_tree),
            'symbols': len(self.symbols),
            'locations': len(self.locations),
            'functions': sum(1 for l in self.locations.values() if l.symbol_type in ['function', 'method']),
            'classes': sum(1 for l in self.locations.values() if l.symbol_type == 'class')
        }

# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
        print(f"Indexing repository: {repo_path}")
        
        navigator = CodeNavigator()
        navigator.index_repository(repo_path)
        
        print("\nStats:", navigator.get_stats())
        print("\nRepo Map:")
        print(navigator.get_repo_map())
        
        # Test search
        if len(sys.argv) > 2:
            query = sys.argv[2]
            print(f"\nSearching for: {query}")
            results = navigator.search_bm25(query, top_k=5)
            for r in results:
                print(f"  [{r.score:.2f}] {r.location.symbol_name} in {r.location.file_path}")
    else:
        print("Usage: python code_navigator.py <repo_path> [search_query]")
