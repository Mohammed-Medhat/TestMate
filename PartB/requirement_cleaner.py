# requirement_cleaner.py - Clean pre-provided requirements
"""
Assumes requirements come in a structured format like:
- JSON file with requirements
- CSV with columns: id, description, priority
- Text file with markdown format

No extraction needed - just cleaning and formatting.
"""

import json
import re
from typing import List, Dict
from final_graph_rag import RequirementNode, KGCompassGraphRAG

class RequirementCleaner:
    """
    Clean and format requirements that are already provided.
    
    Use cases:
    1. Requirements from team's extraction module
    2. Requirements from project docs (already structured)
    3. Requirements from benchmark datasets
    """
    
    def __init__(self, graph: KGCompassGraphRAG):
        self.graph = graph
    
    # ========================================================================
    # OPTION 1: Load from JSON (Most Common)
    # ========================================================================
    
    def load_from_json(self, json_path: str):
        """
        Load requirements from JSON file.
        
        Expected format:
        {
            "requirements": [
                {
                    "id": "REQ-001",
                    "description": "System must handle null inputs",
                    "priority": "high",
                    "source": "README.md"
                },
                ...
            ]
        }
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        requirements = data.get('requirements', [])
        
        for req_dict in requirements:
            # Clean and add
            cleaned_req = self._clean_requirement(req_dict)
            self.graph.add_requirement_node(cleaned_req)
        
        return len(requirements)
    
    # ========================================================================
    # OPTION 2: Load from List/Dict (From Team's Module)
    # ========================================================================
    
    def load_from_list(self, requirements: List[Dict]):
        """
        Load requirements from Python list (from team's extraction).
        
        Format:
        [
            {'id': 'REQ-001', 'text': '...', 'priority': 'high'},
            {'id': 'REQ-002', 'text': '...', 'priority': 'medium'},
        ]
        """
        for req_dict in requirements:
            cleaned_req = self._clean_requirement(req_dict)
            self.graph.add_requirement_node(cleaned_req)
        
        return len(requirements)
    
    # ========================================================================
    # OPTION 3: Load from CSV (Benchmark Datasets)
    # ========================================================================
    
    def load_from_csv(self, csv_path: str):
        """
        Load from CSV file.
        
        Columns: id, description, priority, source
        """
        import csv
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            requirements = list(reader)
        
        for req_dict in requirements:
            cleaned_req = self._clean_requirement(req_dict)
            self.graph.add_requirement_node(cleaned_req)
        
        return len(requirements)
    
    # ========================================================================
    # CLEANING LOGIC
    # ========================================================================
    
    def _clean_requirement(self, req_dict: Dict) -> RequirementNode:
        """
        Clean a single requirement.
        
        Handles:
        - Different key names (id vs req_id, text vs description)
        - Missing fields (add defaults)
        - Formatting issues (strip whitespace, normalize)
        """
        # Normalize keys (handle different formats)
        req_id = self._get_id(req_dict)
        description = self._get_description(req_dict)
        priority = self._get_priority(req_dict)
        source = req_dict.get('source', 'unknown')
        
        # Clean text
        description = self._clean_text(description)
        
        # Detect if orphan (optional)
        is_orphan = req_dict.get('is_orphan', False)
        
        return RequirementNode(
            req_id=req_id,
            description=description,
            priority=priority,
            source=source,
            is_orphan=is_orphan
        )
    
    def _get_id(self, req_dict: Dict) -> str:
        """Handle different ID field names"""
        # Try common field names
        for key in ['id', 'req_id', 'requirement_id', 'number']:
            if key in req_dict:
                return str(req_dict[key])
        
        # Generate ID if missing
        return f"REQ-{hash(str(req_dict)) % 10000:04d}"
    
    def _get_description(self, req_dict: Dict) -> str:
        """Handle different description field names"""
        for key in ['description', 'text', 'content', 'requirement', 'desc']:
            if key in req_dict:
                return str(req_dict[key])
        
        raise ValueError(f"No description field found in {req_dict.keys()}")
    
    def _get_priority(self, req_dict: Dict) -> str:
        """Normalize priority"""
        priority = req_dict.get('priority', 'medium')
        
        # Normalize to: high, medium, low
        priority = priority.lower().strip()
        
        if priority in ['critical', 'urgent', '1', 'p1']:
            return 'high'
        elif priority in ['important', '2', 'p2']:
            return 'medium'
        elif priority in ['optional', 'nice-to-have', '3', 'p3']:
            return 'low'
        
        return priority
    
    def _clean_text(self, text: str) -> str:
        """Clean requirement text"""
        # Strip whitespace
        text = text.strip()
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might break embedding
        # (but keep meaningful punctuation)
        text = re.sub(r'[^\w\s\.,!?-]', '', text)
        
        return text
    
    # ========================================================================
    # LINKING TO CODE (After Cleaning)
    # ========================================================================
    
    def link_to_code(self, threshold: float = 0.7):
        """
        Link requirements to functions via semantic similarity.
        
        Same as before, but works with cleaned requirements.
        """
        import numpy as np
        
        # Get all requirements and functions
        requirements = [n for n in self.graph.graph.nodes() 
                       if self.graph.graph.nodes[n].get('type') == 'requirement']
        
        functions = [n for n in self.graph.graph.nodes() 
                    if self.graph.graph.nodes[n].get('type') == 'function']
        
        links_created = 0
        
        for req_id in requirements:
            req_data = self.graph.graph.nodes[req_id]['data']
            req_emb = np.array(req_data['embedding'])
            
            for func_id in functions:
                func_data = self.graph.graph.nodes[func_id]['data']
                func_emb = np.array(func_data['embedding'])
                
                # Cosine similarity
                similarity = np.dot(req_emb, func_emb) / (
                    np.linalg.norm(req_emb) * np.linalg.norm(func_emb)
                )
                
                if similarity >= threshold:
                    self.graph.add_traces_to_edge(func_id, req_id, similarity)
                    links_created += 1
        
        return links_created

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    from final_graph_rag import KGCompassGraphRAG
    
    graph = KGCompassGraphRAG()
    cleaner = RequirementCleaner(graph)
    
    # Example 1: Load from team's extraction module
    team_requirements = [
        {
            'id': 'REQ-001',
            'text': 'System must handle null PDF inputs gracefully',
            'priority': 'high'
        },
        {
            'id': 'REQ-002',
            'text': 'Application should save data to persistent storage',
            'priority': 'medium'
        }
    ]
    
    cleaner.load_from_list(team_requirements)
    
    # Example 2: Load from JSON file (if team provides this)
    # cleaner.load_from_json('requirements.json')
    
    # Example 3: Load from CSV (benchmark datasets)
    # cleaner.load_from_csv('requirements.csv')
    
    print(f"✅ Loaded {graph.get_stats()['requirement']} requirements")
    
    # After building code graph, link them
    # cleaner.link_to_code(threshold=0.7)