# ============================================================
# build_knowledge_graph.py - Build Graph from SWE-bench Repos
# ============================================================
"""
This script builds the Knowledge Graph for Graph-RAG training.
It parses repositories, extracts code structure, and creates:
- Function nodes with embeddings
- Class nodes
- File nodes  
- CALLS edges between functions
- CONTAIN edges (file → class → function)
- TRACES_TO edges to requirements (if available)
"""

import os
import sys
import ast
import pickle
import networkx as nx
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict
from collections import defaultdict

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from layer2_code.graph_rag import (
    KGCompassGraphRAG,
    IssueNode,
    FileNode,
    ClassNode,
    FunctionNode,
    RequirementNode
)

# ============================================================
# Configuration
# ============================================================

REPOS_DIR = Path("D:/TestMate/TestMate/PartB/repos")
OUTPUT_DIR = Path("D:/TestMate/TestMate/PartB/graphs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# CFG Path Extractor - THE "DEPTH" UPGRADE! (Research-Grade)
# ============================================================

class CFGPathExtractor(ast.NodeVisitor):
    """
    Traces execution paths to find 'Unhappy Paths' and complex logic.
    Returns structured paths: ['IF amount < 0 -> RAISE ValueError', ...]
    
    This is the KEY innovation that makes Graph-RAG "research-grade"!
    Standard RAG sees code. FixMate sees EXECUTION PATHS.
    """
    def __init__(self):
        self.paths = []
        
    def extract(self, node) -> list:
        """Extract all execution paths from a function."""
        self.paths = []
        if isinstance(node, ast.FunctionDef):
            self._explore(node.body, [])
        return self.paths

    def _explore(self, statements, current_conditions):
        """Recursively explore all branches."""
        if not statements:
            # End of path without explicit return
            if current_conditions:
                self.paths.append(current_conditions + ["RETURNS (implicit)"])
            return

        stmt = statements[0]
        remaining = statements[1:]
        
        if isinstance(stmt, ast.If):
            # Branch 1: Condition is True
            try:
                cond = ast.unparse(stmt.test)
            except:
                cond = "condition"
            self._explore(stmt.body + remaining, current_conditions + [f"IF ({cond})"])
            
            # Branch 2: Condition is False (Else/Elif)
            if stmt.orelse:
                self._explore(stmt.orelse + remaining, current_conditions + [f"NOT ({cond})"])
            else:
                # No else - continue with remaining
                self._explore(remaining, current_conditions + [f"NOT ({cond})"])
            
        elif isinstance(stmt, ast.Raise):
            # Path End: Exception (UNHAPPY PATH!)
            try:
                exc = ast.unparse(stmt.exc) if stmt.exc else "Error"
            except:
                exc = "Exception"
            self.paths.append(current_conditions + [f"RAISES {exc}"])
            
        elif isinstance(stmt, ast.Return):
            # Path End: Return
            try:
                val = ast.unparse(stmt.value) if stmt.value else "None"
            except:
                val = "value"
            self.paths.append(current_conditions + [f"RETURNS {val}"])
            
        elif isinstance(stmt, ast.For):
            # Loop: show iteration
            try:
                iter_target = ast.unparse(stmt.target)
                iter_src = ast.unparse(stmt.iter)
            except:
                iter_target, iter_src = "item", "iterable"
            self._explore(stmt.body + remaining, current_conditions + [f"FOR {iter_target} IN {iter_src}"])
            
        elif isinstance(stmt, ast.While):
            # While loop
            try:
                cond = ast.unparse(stmt.test)
            except:
                cond = "condition"
            self._explore(stmt.body + remaining, current_conditions + [f"WHILE ({cond})"])
            
        elif isinstance(stmt, ast.Try):
            # Try/Except block
            self._explore(stmt.body + remaining, current_conditions + ["TRY"])
            for handler in stmt.handlers:
                exc_type = handler.type.id if handler.type and hasattr(handler.type, 'id') else "Exception"
                self._explore(handler.body + remaining, current_conditions + [f"EXCEPT {exc_type}"])
            
        else:
            # Continue linear flow
            self._explore(remaining, current_conditions)
    
    def format_paths(self, max_paths: int = 5) -> list:
        """Format paths as readable strings, prioritizing complex/unhappy paths."""
        # Sort by complexity (longer paths = more conditions = more important)
        sorted_paths = sorted(self.paths, key=len, reverse=True)
        
        # Prioritize paths with RAISES (unhappy paths are critical!)
        raises_paths = [p for p in sorted_paths if any("RAISES" in step for step in p)]
        other_paths = [p for p in sorted_paths if p not in raises_paths]
        
        prioritized = raises_paths + other_paths
        
        formatted = []
        for path in prioritized[:max_paths]:
            formatted.append(" -> ".join(path))
        
        return formatted

# ============================================================
# AST Parser - Extract Code Structure
# ============================================================

class CodeParser:
    """Parse Python files and extract code structure."""
    
    def __init__(self):
        self.functions = []
        self.classes = []
        self.calls = []  # (caller, callee) tuples
        
    def parse_file(self, file_path: Path) -> dict:
        """Parse a Python file and extract all code entities."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
        except Exception as e:
            return None
            
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        
        result = {
            'file_path': str(file_path),
            'functions': [],
            'classes': [],
            'calls': []
        }
        
        # Extract all function definitions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func = self._extract_function(node, source, str(file_path))
                result['functions'].append(func)
                
            elif isinstance(node, ast.ClassDef):
                cls = self._extract_class(node, str(file_path))
                result['classes'].append(cls)
        
        # Extract call relationships
        result['calls'] = self._extract_calls(tree, result['functions'])
        
        return result
    
    def _extract_function(self, node: ast.FunctionDef, source: str, file_path: str) -> dict:
        """Extract function details from AST node, INCLUDING execution paths."""
        lines = source.split('\n')
        start = node.lineno - 1
        end = node.end_lineno if hasattr(node, 'end_lineno') else start + 10
        body = '\n'.join(lines[start:end])
        
        # Get docstring
        docstring = ast.get_docstring(node) or ""
        
        # Get signature
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        signature = f"def {node.name}({', '.join(args)})"
        
        # NEW: Extract CFG execution paths (THE DEPTH UPGRADE!)
        cfg_extractor = CFGPathExtractor()
        cfg_extractor.extract(node)
        formatted_paths = cfg_extractor.format_paths(max_paths=5)
        
        return {
            'name': node.name,
            'signature': signature,
            'body': body[:500],  # Limit body size
            'docstring': docstring[:200],
            'file_path': file_path,
            'line_start': node.lineno,
            'line_end': end,
            'class_name': None,  # Will be set for methods
            'paths': formatted_paths  # CFG execution paths!
        }
    
    def _extract_class(self, node: ast.ClassDef, file_path: str) -> dict:
        """Extract class details from AST node."""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(item.name)
        
        parent = None
        if node.bases:
            for base in node.bases:
                if isinstance(base, ast.Name):
                    parent = base.id
                    break
        
        return {
            'name': node.name,
            'file_path': file_path,
            'methods': methods,
            'parent_class': parent,
            'docstring': ast.get_docstring(node) or ""
        }
    
    def _extract_calls(self, tree: ast.AST, functions: list) -> list:
        """Extract function call relationships."""
        calls = []
        func_names = {f['name'] for f in functions}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                caller = node.name
                
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            callee = child.func.id
                            if callee in func_names and callee != caller:
                                calls.append((caller, callee))
                        elif isinstance(child.func, ast.Attribute):
                            callee = child.func.attr
                            if callee in func_names and callee != caller:
                                calls.append((caller, callee))
        
        return list(set(calls))

# ============================================================
# Graph Builder
# ============================================================

class KnowledgeGraphBuilder:
    """Build Knowledge Graph from parsed code."""
    
    def __init__(self):
        self.graph = KGCompassGraphRAG()
        self.parser = CodeParser()
        self.stats = defaultdict(int)
    
    def build_from_repo(self, repo_path: Path):
        """Build graph from a repository."""
        print(f"\n📁 Processing repository: {repo_path.name}")
        
        # Find all Python files
        python_files = list(repo_path.rglob("*.py"))
        print(f"   Found {len(python_files)} Python files")
        
        for py_file in python_files:
            # Skip test files (but NOT the testgen folder - we want to parse that!)
            filename = py_file.name.lower()
            if filename.startswith('test_') or filename.endswith('_test.py'):
                continue
            
            result = self.parser.parse_file(py_file)
            if not result:
                continue
            
            # Add file node
            file_node = FileNode(
                file_path=result['file_path'],
                language="python"
            )
            self.graph.add_file_node(file_node)
            self.stats['files'] += 1
            
            # Add class nodes
            for cls_data in result['classes']:
                cls_node = ClassNode(
                    name=cls_data['name'],
                    file_path=cls_data['file_path'],
                    methods=cls_data['methods'],
                    parent_class=cls_data['parent_class'],
                    docstring=cls_data['docstring']
                )
                self.graph.add_class_node(cls_node)
                self.graph.add_contain_edge(result['file_path'], cls_data['name'])
                self.stats['classes'] += 1
            
            # Add function nodes WITH CFG PATHS!
            for func_data in result['functions']:
                func_node = FunctionNode(
                    name=func_data['name'],
                    signature=func_data['signature'],
                    body=func_data['body'],
                    file_path=func_data['file_path'],
                    class_name=func_data['class_name'],
                    line_start=func_data['line_start'],
                    line_end=func_data['line_end'],
                    docstring=func_data['docstring'],
                    paths=func_data.get('paths', [])  # CFG paths stored!
                )
                self.graph.add_function_node(func_node)
                self.graph.add_contain_edge(result['file_path'], func_data['name'])
                self.stats['functions'] += 1
            
            # Add CALLS edges
            for caller, callee in result['calls']:
                self.graph.add_calls_edge(caller, callee)
                self.stats['calls'] += 1
        
        print(f"   ✅ Added: {self.stats['files']} files, {self.stats['classes']} classes, "
              f"{self.stats['functions']} functions, {self.stats['calls']} call relationships")
    
    def add_issue(self, issue_id: str, title: str, description: str, files: list = None):
        """Add an issue node to the graph."""
        issue = IssueNode(
            issue_id=issue_id,
            title=title,
            description=description,
            referenced_files=files or []
        )
        self.graph.add_issue_node(issue)
        
        # Add reference edges to files
        if files:
            for f in files:
                if f in self.graph.graph.nodes():
                    self.graph.add_reference_edge(issue_id, f)
        
        self.stats['issues'] += 1
    
    def add_requirement(self, req_id: str, description: str, source: str, priority: str = "medium"):
        """Add a requirement node (from Team A)."""
        req = RequirementNode(
            req_id=req_id,
            description=description,
            source=source,
            priority=priority
        )
        self.graph.add_requirement_node(req)
        self.stats['requirements'] += 1
    
    def save(self, output_path: Path):
        """Save the graph to pickle file."""
        self.graph.save(str(output_path))
        print(f"\n💾 Saved graph to: {output_path}")
        print(f"   Stats: {dict(self.stats)}")
    
    def get_stats(self):
        return {
            'total_nodes': self.graph.graph.number_of_nodes(),
            'total_edges': self.graph.graph.number_of_edges(),
            **dict(self.stats)
        }

# ============================================================
# Main - Build Graph from Available Repos
# ============================================================

def main():
    print("="*60)
    print("KNOWLEDGE GRAPH BUILDER FOR GRAPH-RAG")
    print("="*60)
    
    builder = KnowledgeGraphBuilder()
    
    # Always build from current project directories
    project_root = Path("D:/TestMate/TestMate/PartB")
    
    # Parse layer2_code (Graph RAG code)
    if (project_root / "layer2_code").exists():
        builder.build_from_repo(project_root / "layer2_code")
    
    # Parse testgen (test generation code)
    if (project_root / "testgen").exists():
        builder.build_from_repo(project_root / "testgen")
    
    # Parse layer1_docs if exists
    if (project_root / "layer1_docs").exists():
        builder.build_from_repo(project_root / "layer1_docs")
    
    # Check for repos directory (additional repos if available)
    if REPOS_DIR.exists():
        repos = [d for d in REPOS_DIR.iterdir() if d.is_dir()]
        print(f"\nFound {len(repos)} extra repositories in {REPOS_DIR}")
        
        for repo in repos[:5]:  # Limit to first 5 repos
            builder.build_from_repo(repo)
    
    # Add some sample requirements (Team A integration)
    print("\n📋 Adding sample requirements...")
    builder.add_requirement(
        "REQ-001",
        "System must handle null inputs gracefully without crashing",
        "requirements.md",
        "high"
    )
    builder.add_requirement(
        "REQ-002", 
        "All API endpoints must validate input parameters",
        "requirements.md",
        "high"
    )
    builder.add_requirement(
        "REQ-003",
        "Error messages must be user-friendly and informative",
        "requirements.md",
        "medium"
    )
    
    # Save the graph
    output_path = OUTPUT_DIR / "knowledge_graph.pkl"
    builder.save(output_path)
    
    # Print final stats
    stats = builder.get_stats()
    print("\n" + "="*60)
    print("GRAPH BUILD COMPLETE")
    print("="*60)
    print(f"Total Nodes: {stats['total_nodes']}")
    print(f"Total Edges: {stats['total_edges']}")
    print(f"Files: {stats['files']}")
    print(f"Classes: {stats['classes']}")  
    print(f"Functions: {stats['functions']}")
    print(f"Calls: {stats['calls']}")
    print(f"Requirements: {stats['requirements']}")
    print(f"\n✅ Graph saved to: {output_path}")
    print("📤 Upload this file to Kaggle as a dataset!")

if __name__ == "__main__":
    main()
