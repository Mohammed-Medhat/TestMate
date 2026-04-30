# graph_schema.py - Define your graph structure
from pydantic import BaseModel
from typing import List, Optional
import networkx as nx
from sentence_transformers import SentenceTransformer
import pickle

class FunctionNode(BaseModel):
    """Represents a function in the codebase"""
    name: str
    file_path: str
    line_start: int
    line_end: int
    signature: str
    body: str
    docstring: Optional[str] = None
    embedding: Optional[List[float]] = None

class RequirementNode(BaseModel):
    """Represents a functional requirement"""
    req_id: str
    description: str
    priority: str = "medium"
    embedding: Optional[List[float]] = None

class TestNode(BaseModel):
    """Represents a test case"""
    test_name: str
    target_function: str
    status: str = "pending"  # pending, passed, failed
    error_msg: Optional[str] = None

class CodeGraph:
    """Main graph structure for Graph-RAG"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def add_function(self, func: FunctionNode):
        """Add function node with embedding"""
        # Generate embedding from function body + docstring
        text = f"{func.signature}\n{func.docstring or ''}\n{func.body}"
        func.embedding = self.encoder.encode(text).tolist()
        
        self.graph.add_node(
            func.name,
            type='function',
            data=func.dict()
        )
    
    def add_requirement(self, req: RequirementNode):
        """Add requirement node with embedding"""
        req.embedding = self.encoder.encode(req.description).tolist()
        
        self.graph.add_node(
            req.req_id,
            type='requirement',
            data=req.dict()
        )
    
    def add_test(self, test: TestNode):
        """Add test node"""
        self.graph.add_node(
            test.test_name,
            type='test',
            data=test.dict()
        )
    
    def add_call_edge(self, caller: str, callee: str):
        """Function A calls Function B"""
        self.graph.add_edge(caller, callee, type='CALLS')
    
    def add_implements_edge(self, func_name: str, req_id: str, confidence: float):
        """Function implements Requirement"""
        self.graph.add_edge(func_name, req_id, type='IMPLEMENTS', confidence=confidence)
    
    def add_tests_edge(self, test_name: str, func_name: str):
        """Test covers Function"""
        self.graph.add_edge(test_name, func_name, type='TESTS')
    
    def add_fails_at_edge(self, test_name: str, func_name: str, error_msg: str):
        """Test fails at Function"""
        self.graph.add_edge(test_name, func_name, type='FAILS_AT', error=error_msg)
    
    def save(self, path: str):
        """Save graph to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self.graph, f)
    
    def load(self, path: str):
        """Load graph from disk"""
        with open(path, 'rb') as f:
            self.graph = pickle.load(f)
    
    def get_stats(self):
        """Quick validation"""
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'functions': len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'function']),
            'requirements': len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'requirement']),
            'tests': len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'test'])
        }

# Quick test
if __name__ == "__main__":
    graph = CodeGraph()
    
    # Add a function
    func = FunctionNode(
        name="extract_text_from_pdf",
        file_path="utils.py",
        line_start=10,
        line_end=25,
        signature="def extract_text_from_pdf(file_or_path):",
        body="pdf = fitz.open(file_or_path)\ntext = ''\nfor page in pdf:\n    text += page.get_text()\nreturn text",
        docstring="Extracts text from PDF files"
    )
    graph.add_function(func)
    
    # Add a requirement
    req = RequirementNode(
        req_id="REQ-001",
        description="System must extract text from PDF documents",
        priority="high"
    )
    graph.add_requirement(req)
    
    # Link them
    graph.add_implements_edge("extract_text_from_pdf", "REQ-001", confidence=0.95)
    
    print(graph.get_stats())
    # Output: {'total_nodes': 2, 'total_edges': 1, 'functions': 1, 'requirements': 1, 'tests': 0}