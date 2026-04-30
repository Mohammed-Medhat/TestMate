# ast_parser_complete.py - Parse repository and build Graph-RAG
"""
Complete AST parser following KGCompass architecture.

Usage:
    parser = KGCompassParser()
    graph = parser.parse_repository("./path/to/repo")
    graph.save("graph.pkl")
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set
from final_graph_rag import (
    KGCompassGraphRAG,
    FileNode,
    ClassNode,
    FunctionNode,
    IssueNode
)

class KGCompassParser:
    """
    Parse Python repository into KGCompass graph structure.
    
    Builds:
    - File nodes
    - Class nodes (with CONTAIN edges)
    - Function nodes (with CONTAIN edges)
    - CALLS edges (function → function)
    - INHERIT edges (class → class)
    """
    
    def __init__(self):
        self.graph = KGCompassGraphRAG()
        self.function_calls = {}  # Track calls for second pass
        self.class_inheritance = {}  # Track inheritance
        
    def parse_repository(self, repo_path: str, exclude_dirs: List[str] = None):
        """
        Parse entire repository.
        
        Args:
            repo_path: Path to repository root
            exclude_dirs: Directories to skip (e.g., ['venv', 'tests'])
        
        Returns:
            KGCompassGraphRAG instance
        """
        if exclude_dirs is None:
            exclude_dirs = ['venv', 'env', '.venv', '__pycache__', '.git', 'node_modules']
        
        repo_path = Path(repo_path)
        
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        print(f"📂 Parsing repository: {repo_path}")
        
        # Find all Python files
        python_files = []
        for py_file in repo_path.rglob("*.py"):
            # Skip excluded directories
            if any(excluded in py_file.parts for excluded in exclude_dirs):
                continue
            python_files.append(py_file)
        
        print(f"📄 Found {len(python_files)} Python files")
        
        # First pass: Parse files and extract nodes
        for i, py_file in enumerate(python_files, 1):
            print(f"  [{i}/{len(python_files)}] Parsing {py_file.name}...", end='\r')
            self._parse_file(py_file, repo_path)
        
        print(f"\n✅ Parsed {len(python_files)} files")
        
        # Second pass: Build edges
        print("🔗 Building edges...")
        self._build_call_edges()
        self._build_inheritance_edges()
        
        print(f"✅ Graph construction complete!")
        print(f"📊 Stats: {self.graph.get_stats()}")
        
        return self.graph
    
    def _parse_file(self, file_path: Path, repo_root: Path):
        """Parse single Python file"""
        
        rel_path = str(file_path.relative_to(repo_root))
        
        # Add File node
        file_node = FileNode(
            file_path=rel_path,
            language="python"
        )
        self.graph.add_file_node(file_node)
        
        # Read and parse
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code, filename=str(file_path))
        except Exception as e:
            print(f"\n⚠️  Failed to parse {rel_path}: {e}")
            return
        
        # Extract top-level classes and functions
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                self._extract_class(node, rel_path, source_code)
            elif isinstance(node, ast.FunctionDef):
                self._extract_function(node, rel_path, source_code, class_name=None)
    
    def _extract_class(self, ast_node: ast.ClassDef, file_path: str, source_code: str):
        """Extract class and its methods"""
        
        class_name = ast_node.name
        
        # Extract method names
        methods = [
            item.name for item in ast_node.body 
            if isinstance(item, ast.FunctionDef)
        ]
        
        # Extract parent classes (for INHERIT edges)
        parent_classes = [
            base.id if isinstance(base, ast.Name) else None
            for base in ast_node.bases
        ]
        parent_classes = [p for p in parent_classes if p]  # Remove None
        
        # Create Class node
        class_node = ClassNode(
            name=class_name,
            file_path=file_path,
            methods=methods,
            parent_class=parent_classes[0] if parent_classes else None,
            docstring=ast.get_docstring(ast_node)
        )
        self.graph.add_class_node(class_node)
        
        # File CONTAINS Class
        self.graph.add_contain_edge(file_path, class_name)
        
        # Store inheritance for later
        if parent_classes:
            self.class_inheritance[class_name] = parent_classes
        
        # Extract methods
        for item in ast_node.body:
            if isinstance(item, ast.FunctionDef):
                self._extract_function(item, file_path, source_code, class_name=class_name)
    
    def _extract_function(self, ast_node: ast.FunctionDef, file_path: str, 
                         source_code: str, class_name: str = None):
        """Extract function/method"""
        
        func_name = ast_node.name
        
        # Build full name (Class.method or just function)
        if class_name:
            full_name = f"{class_name}.{func_name}"
        else:
            full_name = func_name
        
        # Extract signature
        args = []
        for arg in ast_node.args.args:
            args.append(arg.arg)
        signature = f"def {func_name}({', '.join(args)}):"
        
        # Extract function body (try to get source)
        try:
            # Get line numbers
            line_start = ast_node.lineno
            line_end = ast_node.end_lineno or line_start
            
            # Extract source lines
            lines = source_code.split('\n')
            body_lines = lines[line_start-1:line_end]
            body = '\n'.join(body_lines)
        except:
            body = f"# Function {func_name}"
        
        # Create Function node
        func_node = FunctionNode(
            name=full_name,
            signature=signature,
            body=body[:2000],  # Limit to 2000 chars
            file_path=file_path,
            class_name=class_name,
            line_start=ast_node.lineno,
            line_end=ast_node.end_lineno or ast_node.lineno,
            docstring=ast.get_docstring(ast_node)
        )
        self.graph.add_function_node(func_node)
        
        # Hierarchy: File/Class CONTAINS Function
        container = class_name if class_name else file_path
        self.graph.add_contain_edge(container, full_name)
        
        # Extract function calls (for CALLS edges)
        calls = self._extract_calls(ast_node)
        self.function_calls[full_name] = calls
    
    def _extract_calls(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract function calls from AST"""
        
        calls = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                # Handle different call types
                if isinstance(node.func, ast.Name):
                    # Simple call: func()
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # Method call: obj.method()
                    calls.append(node.func.attr)
        
        return list(set(calls))  # Remove duplicates
    
    def _build_call_edges(self):
        """Build CALLS edges (second pass)"""
        
        added = 0
        
        for caller, callees in self.function_calls.items():
            for callee in callees:
                # Check if both exist in graph
                if (caller in self.graph.graph.nodes() and 
                    callee in self.graph.graph.nodes()):
                    self.graph.add_calls_edge(caller, callee)
                    added += 1
        
        print(f"  Added {added} CALLS edges")
    
    def _build_inheritance_edges(self):
        """Build INHERIT edges (second pass)"""
        
        added = 0
        
        for child, parents in self.class_inheritance.items():
            for parent in parents:
                # Check if both exist
                if (child in self.graph.graph.nodes() and 
                    parent in self.graph.graph.nodes()):
                    self.graph.add_inherit_edge(child, parent)
                    added += 1
        
        print(f"  Added {added} INHERIT edges")
    
    def add_issue(self, issue_dict: Dict):
        """
        Add Issue node (bug report).
        
        Format:
        {
            'id': 'BUG-001',
            'title': 'NoneType error in extract_text_from_pdf',
            'description': 'When file_or_path is None...',
            'referenced_files': ['utils.py']
        }
        """
        issue = IssueNode(
            issue_id=issue_dict['id'],
            title=issue_dict['title'],
            description=issue_dict['description'],
            referenced_files=issue_dict.get('referenced_files', [])
        )
        
        self.graph.add_issue_node(issue)
        
        # Add REFERENCE edges to mentioned files
        for file_path in issue.referenced_files:
            if file_path in self.graph.graph.nodes():
                self.graph.add_reference_edge(issue.issue_id, file_path)

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Parse repository
    parser = KGCompassParser()
    
    # Example: Parse current directory
    repo_path = "."  # Or provide path as argument
    
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    
    print(f"Parsing repository: {repo_path}")
    print("=" * 60)
    
    graph = parser.parse_repository(repo_path)
    
    # Save graph
    output_path = "graph.pkl"
    graph.save(output_path)
    print(f"\n💾 Graph saved to {output_path}")
    
    # Print statistics
    print("\n📊 Final Statistics:")
    stats = graph.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Example: Add an issue
    print("\n🐛 Adding example issue...")
    parser.add_issue({
        'id': 'ISSUE-001',
        'title': 'Function fails on null input',
        'description': 'The function crashes when input is None',
        'referenced_files': ['utils.py']
    })
    
    # Test retrieval
    if 'ISSUE-001' in graph.graph.nodes():
        print("\n🔍 Testing retrieval (top 5 candidates)...")
        try:
            top_candidates = graph.retrieve_top_20_candidates('ISSUE-001', max_candidates=5)
            
            for i, candidate in enumerate(top_candidates, 1):
                print(f"\n{i}. {candidate['function']}")
                print(f"   Score: {candidate['score']:.3f}")
                print(f"   Distance: {candidate['path_distance']} hops")
        except Exception as e:
            print(f"⚠️  Retrieval failed: {e}")
            print("   (This is normal if graph is too small)")
    
    print("\n✅ Done!")