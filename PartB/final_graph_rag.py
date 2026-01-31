# final_graph_rag.py - Complete Graph-RAG from KGCompass + RAGFix + Traceability
"""
Combines best practices from:
- KGCompass: Multi-hop traversal, Issue nodes, 20-node candidates
- RAGFix: External knowledge (Stack Overflow), similarity-based retrieval
- Traceability: Requirement validation, orphan detection

YOUR INNOVATION: Self-healing memory layer (not in any paper!)
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Set
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
import pickle

# ============================================================================
# NODE SCHEMAS (From Papers)
# ============================================================================

class IssueNode(BaseModel):
    """
    From KGCompass: Natural language bug reports.
    
    Starting point for retrieval!
    """
    issue_id: str
    title: str
    description: str  # Natural language bug report
    labels: List[str] = []
    referenced_files: List[str] = []
    embedding: Optional[List[float]] = None

class FileNode(BaseModel):
    """From KGCompass: Repository structure"""
    file_path: str
    language: str = "python"
    imports: List[str] = []
    loc: int = 0

class ClassNode(BaseModel):
    """From KGCompass: CONTAIN hierarchy"""
    name: str
    file_path: str
    methods: List[str] = []
    parent_class: Optional[str] = None  # For INHERIT edge
    docstring: Optional[str] = None

class FunctionNode(BaseModel):
    """From KGCompass: Top 20 candidates"""
    name: str
    signature: str
    body: str
    file_path: str
    class_name: Optional[str] = None
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    embedding: Optional[List[float]] = None
    complexity: Optional[int] = None

class RequirementNode(BaseModel):
    """From Traceability: Requirements from docs"""
    req_id: str
    description: str
    source: str  # "README.md", "docs/api.md"
    priority: str = "medium"
    is_orphan: bool = False  # No code implements it
    embedding: Optional[List[float]] = None

class TestNode(BaseModel):
    """From Traceability: VERIFIES relationship"""
    test_name: str
    test_file: str
    target_function: Optional[str] = None
    verifies_requirement: Optional[str] = None  # Links to requirement!
    status: str = "pending"
    error_message: Optional[str] = None

class StackOverflowNode(BaseModel):
    """
    From RAGFix: External knowledge (YOUR UNIQUE ADDITION).
    
    Memory layer for patterns not in the repo.
    """
    so_id: str
    question: str
    accepted_answer: str
    tags: List[str] = []
    embedding: Optional[List[float]] = None

class PatchMemoryNode(BaseModel):
    """
    YOUR INNOVATION (not in papers): Self-healing layer.
    
    Stores historical repair attempts.
    """
    patch_id: str
    bug_description: str
    code: str
    success: bool
    requirements_violated: List[str] = []
    embedding: Optional[List[float]] = None

# ============================================================================
# COMPLETE GRAPH-RAG SYSTEM
# ============================================================================

class KGCompassGraphRAG:
    """
    Production Graph-RAG combining all 3 papers + your innovation.
    
    Architecture:
    1. KGCompass structure (Issue → File → Class → Function)
    2. RAGFix external knowledge (Stack Overflow nodes)
    3. Traceability validation (Requirement checking)
    4. YOUR innovation (Patch memory + self-healing)
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder="D:/TestMate/huggingface_cache/hub")
        
        # Track statistics
        self.stats = defaultdict(int)
    
    # ========================================================================
    # PART 1: GRAPH CONSTRUCTION (KGCompass Method)
    # ========================================================================
    
    def add_issue_node(self, issue: IssueNode):
        """Starting point for KGCompass retrieval"""
        if not issue.embedding:
            issue.embedding = self.encoder.encode(
                f"{issue.title}\n{issue.description}"
            ).tolist()
        
        self.graph.add_node(
            issue.issue_id,
            type='issue',
            data=issue.dict(),
            embedding=issue.embedding
        )
        self.stats['issue'] += 1
    
    def add_file_node(self, file: FileNode):
        """Repository structure"""
        self.graph.add_node(
            file.file_path,
            type='file',
            data=file.dict()
        )
        self.stats['file'] += 1
    
    def add_class_node(self, cls: ClassNode):
        """Class hierarchy (CONTAIN edge)"""
        self.graph.add_node(
            cls.name,
            type='class',
            data=cls.dict()
        )
        self.stats['class'] += 1
    
    def add_function_node(self, func: FunctionNode):
        """Function (candidate for top 20)"""
        if not func.embedding:
            text = f"{func.signature}\n{func.docstring or ''}\n{func.body[:500]}"
            func.embedding = self.encoder.encode(text).tolist()
        
        self.graph.add_node(
            func.name,
            type='function',
            data=func.dict(),
            embedding=func.embedding
        )
        self.stats['function'] += 1
    
    def add_requirement_node(self, req: RequirementNode):
        """From Traceability paper"""
        if not req.embedding:
            req.embedding = self.encoder.encode(req.description).tolist()
        
        self.graph.add_node(
            req.req_id,
            type='requirement',
            data=req.dict(),
            embedding=req.embedding,
            is_orphan=req.is_orphan
        )
        self.stats['requirement'] += 1
    
    def add_test_node(self, test: TestNode):
        """Test case (VERIFIES edge)"""
        self.graph.add_node(
            test.test_name,
            type='test',
            data=test.dict()
        )
        self.stats['test'] += 1
    
    def add_stackoverflow_node(self, so: StackOverflowNode):
        """RAGFix: External knowledge"""
        if not so.embedding:
            so.embedding = self.encoder.encode(
                f"{so.question}\n{so.accepted_answer}"
            ).tolist()
        
        self.graph.add_node(
            so.so_id,
            type='stackoverflow',
            data=so.dict(),
            embedding=so.embedding
        )
        self.stats['stackoverflow'] += 1
    
    def add_patch_memory_node(self, patch: PatchMemoryNode):
        """YOUR INNOVATION: Self-healing memory"""
        if not patch.embedding:
            patch.embedding = self.encoder.encode(patch.bug_description).tolist()
        
        self.graph.add_node(
            patch.patch_id,
            type='patch_memory',
            data=patch.dict(),
            embedding=patch.embedding,
            success=patch.success
        )
        self.stats['patch_memory'] += 1
    
    # ========================================================================
    # PART 2: EDGE CONSTRUCTION (All Papers)
    # ========================================================================
    
    # KGCompass Edges
    def add_calls_edge(self, caller: str, callee: str):
        """Function → Function"""
        self.graph.add_edge(caller, callee, type='CALLS')
    
    def add_contain_edge(self, container: str, contained: str):
        """File → Class → Function hierarchy"""
        self.graph.add_edge(container, contained, type='CONTAIN')
    
    def add_inherit_edge(self, child: str, parent: str):
        """Class → Class inheritance"""
        self.graph.add_edge(child, parent, type='INHERIT')
    
    def add_reference_edge(self, issue: str, target: str):
        """Issue → Function/File (mentioned in bug report)"""
        self.graph.add_edge(issue, target, type='REFERENCE')
    
    # Traceability Edges
    def add_traces_to_edge(self, func: str, req: str, confidence: float):
        """
        Bidirectional: Code ↔ Requirement
        
        From Traceability paper: Reduces regression bugs by 30%!
        """
        # Forward
        self.graph.add_edge(func, req, type='TRACES_TO', confidence=confidence)
        # Backward
        self.graph.add_edge(req, func, type='TRACED_BY', confidence=confidence)
    
    def add_verifies_edge(self, test: str, req: str):
        """Test → Requirement"""
        self.graph.add_edge(test, req, type='VERIFIES')
    
    # RAGFix Edges
    def add_similar_to_edge(self, node1: str, node2: str, similarity: float):
        """Bug ↔ Bug, Bug ↔ StackOverflow"""
        self.graph.add_edge(node1, node2, type='SIMILAR_TO', similarity=similarity)
    
    # YOUR INNOVATION
    def add_fixed_by_edge(self, func: str, patch: str, success: bool):
        """Function → PatchMemory"""
        edge_type = 'FIXED_BY_GOLDEN' if success else 'FIXED_BY_FAILED'
        self.graph.add_edge(func, patch, type=edge_type)
    
    # ========================================================================
    # PART 3: RETRIEVAL (KGCompass Algorithm)
    # ========================================================================
    
    def retrieve_top_20_candidates(self, issue_node: str, max_candidates: int = 20):
        """
        KGCompass retrieval: Issue → Top 20 Functions
        
        Algorithm:
        1. Start from Issue node
        2. Multi-hop traversal along CALLS, CONTAIN, REFERENCE edges
        3. Score each function: S(f) = semantic_similarity + path_distance
        4. Return top 20
        """
        if issue_node not in self.graph.nodes():
            raise ValueError(f"Issue {issue_node} not in graph")
        
        # Get issue embedding
        issue_data = self.graph.nodes[issue_node]['data']
        issue_embedding = np.array(issue_data['embedding'])
        
        # Collect all reachable functions via multi-hop
        candidates = []
        visited = set()
        queue = [(issue_node, 0)]  # (node, distance)
        
        while queue:
            current, dist = queue.pop(0)
            
            if current in visited:
                continue
            visited.add(current)
            
            # If it's a function, add to candidates
            if self.graph.nodes[current].get('type') == 'function':
                func_data = self.graph.nodes[current]['data']
                func_embedding = np.array(func_data['embedding'])
                
                # Score function (KGCompass formula)
                semantic_sim = np.dot(issue_embedding, func_embedding) / (
                    np.linalg.norm(issue_embedding) * np.linalg.norm(func_embedding)
                )
                
                # Inverse path distance (closer = higher score)
                path_score = 1.0 / (dist + 1)
                
                # Combined score
                score = 0.7 * semantic_sim + 0.3 * path_score
                
                candidates.append({
                    'function': current,
                    'score': score,
                    'semantic_similarity': semantic_sim,
                    'path_distance': dist,
                    'data': func_data
                })
            
            # Expand to neighbors (multi-hop)
            if dist < 5:  # Max 5 hops
                for neighbor in self.graph.neighbors(current):
                    edge_type = self.graph[current][neighbor][0].get('type')
                    
                    # Follow relevant edges (KGCompass strategy)
                    if edge_type in ['CALLS', 'CONTAIN', 'REFERENCE', 'TRACES_TO']:
                        queue.append((neighbor, dist + 1))
        
        # Sort by score, return top 20
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:max_candidates]
    
    def two_hop_traversal(self, seed_function: str) -> Dict:
        """
        KGCompass 2-Hop Traversal Query - The Heart of KGCompass!
        
        Given a "Seed Node" (suspected buggy function), retrieves its neighborhood:
        
        1. Incoming CALLS edges: Who calls this function? 
           → To trace where bad data came from
        
        2. Outgoing CALLS edges: Who does this function call?
           → To check if a dependency is actually the problem
        
        3. Parent CONTAINS nodes: The Class or Module it belongs to
           → To understand the structural context
        
        Returns full neighborhood context for the LLM.
        
        Cypher-equivalent query:
        ```cypher
        MATCH (caller:Function)-[:CALLS]->(seed:Function {name: $seed_function})
        MATCH (seed)-[:CALLS]->(callee:Function)
        MATCH (parent)-[:CONTAIN]->(seed)
        RETURN caller, seed, callee, parent
        ```
        """
        if seed_function not in self.graph.nodes():
            raise ValueError(f"Seed function '{seed_function}' not found in graph")
        
        seed_node = self.graph.nodes[seed_function]
        if seed_node.get('type') != 'function':
            raise ValueError(f"'{seed_function}' is not a function node")
        
        neighborhood = {
            'seed': {
                'name': seed_function,
                'data': seed_node.get('data', {}),
                'type': seed_node.get('type')
            },
            'callers': [],      # Functions that call the seed (incoming CALLS)
            'callees': [],      # Functions the seed calls (outgoing CALLS)
            'parent': None,     # Class or Module containing the seed
            'siblings': [],     # Other functions in the same parent
            'second_hop': {
                'caller_callers': [],   # Who calls the callers (2-hop incoming)
                'callee_callees': []    # Who the callees call (2-hop outgoing)
            }
        }
        
        # ====================================================================
        # HOP 1: Direct Neighbors
        # ====================================================================
        
        # 1a. INCOMING CALLS: Who calls this function?
        for predecessor in self.graph.predecessors(seed_function):
            edges = self.graph[predecessor][seed_function]
            for edge_key, edge_data in edges.items():
                if edge_data.get('type') == 'CALLS':
                    caller_node = self.graph.nodes[predecessor]
                    neighborhood['callers'].append({
                        'name': predecessor,
                        'type': caller_node.get('type'),
                        'data': caller_node.get('data', {})
                    })
        
        # 1b. OUTGOING CALLS: Who does this function call?
        for successor in self.graph.successors(seed_function):
            edges = self.graph[seed_function][successor]
            for edge_key, edge_data in edges.items():
                if edge_data.get('type') == 'CALLS':
                    callee_node = self.graph.nodes[successor]
                    neighborhood['callees'].append({
                        'name': successor,
                        'type': callee_node.get('type'),
                        'data': callee_node.get('data', {})
                    })
        
        # 1c. PARENT CONTAINS: Class or Module containing this function
        for predecessor in self.graph.predecessors(seed_function):
            edges = self.graph[predecessor][seed_function]
            for edge_key, edge_data in edges.items():
                if edge_data.get('type') == 'CONTAIN':
                    parent_node = self.graph.nodes[predecessor]
                    neighborhood['parent'] = {
                        'name': predecessor,
                        'type': parent_node.get('type'),
                        'data': parent_node.get('data', {})
                    }
                    
                    # Get sibling functions (other functions in same parent)
                    for sibling in self.graph.successors(predecessor):
                        sibling_edges = self.graph[predecessor][sibling]
                        for s_key, s_data in sibling_edges.items():
                            if s_data.get('type') == 'CONTAIN' and sibling != seed_function:
                                sibling_node = self.graph.nodes[sibling]
                                if sibling_node.get('type') == 'function':
                                    neighborhood['siblings'].append({
                                        'name': sibling,
                                        'type': sibling_node.get('type'),
                                        'data': sibling_node.get('data', {})
                                    })
                    break  # Only get first parent
        
        # ====================================================================
        # HOP 2: Extended Neighborhood
        # ====================================================================
        
        # 2a. Who calls the callers? (2-hop incoming)
        for caller in neighborhood['callers']:
            caller_name = caller['name']
            for predecessor in self.graph.predecessors(caller_name):
                edges = self.graph[predecessor][caller_name]
                for edge_key, edge_data in edges.items():
                    if edge_data.get('type') == 'CALLS':
                        caller_caller_node = self.graph.nodes[predecessor]
                        neighborhood['second_hop']['caller_callers'].append({
                            'name': predecessor,
                            'via': caller_name,
                            'type': caller_caller_node.get('type'),
                            'data': caller_caller_node.get('data', {})
                        })
        
        # 2b. Who do the callees call? (2-hop outgoing)
        for callee in neighborhood['callees']:
            callee_name = callee['name']
            for successor in self.graph.successors(callee_name):
                edges = self.graph[callee_name][successor]
                for edge_key, edge_data in edges.items():
                    if edge_data.get('type') == 'CALLS':
                        callee_callee_node = self.graph.nodes[successor]
                        neighborhood['second_hop']['callee_callees'].append({
                            'name': successor,
                            'via': callee_name,
                            'type': callee_callee_node.get('type'),
                            'data': callee_callee_node.get('data', {})
                        })
        
        return neighborhood
    
    def format_neighborhood_context(self, neighborhood: Dict) -> str:
        """
        Format the 2-hop neighborhood into a prompt-friendly string.
        
        This gives the LLM all the context it needs to understand:
        - Where the bug might have originated (callers)
        - What might actually be broken (callees/dependencies)
        - The structural context (class, siblings)
        """
        seed = neighborhood['seed']
        seed_data = seed.get('data', {})
        
        context = f"""=== 2-HOP NEIGHBORHOOD ANALYSIS ===
        
🎯 SEED FUNCTION: {seed['name']}
   File: {seed_data.get('file_path', 'unknown')}
   Lines: {seed_data.get('line_start', '?')}-{seed_data.get('line_end', '?')}
   Signature: {seed_data.get('signature', 'N/A')}

"""
        
        # Parent context
        if neighborhood['parent']:
            parent = neighborhood['parent']
            parent_data = parent.get('data', {})
            context += f"""📦 PARENT ({parent['type'].upper()}): {parent['name']}
   File: {parent_data.get('file_path', parent_data.get('name', 'unknown'))}

"""
        
        # Callers - where bad data might come from
        if neighborhood['callers']:
            context += "📥 INCOMING CALLS (potential sources of bad data):\n"
            for i, caller in enumerate(neighborhood['callers'][:5], 1):
                caller_data = caller.get('data', {})
                context += f"   {i}. {caller['name']} → calls → {seed['name']}\n"
                context += f"      File: {caller_data.get('file_path', 'unknown')}\n"
            context += "\n"
        else:
            context += "📥 INCOMING CALLS: None (entry point function)\n\n"
        
        # Callees - dependencies that might be the actual problem
        if neighborhood['callees']:
            context += "📤 OUTGOING CALLS (dependencies - might be the real problem):\n"
            for i, callee in enumerate(neighborhood['callees'][:5], 1):
                callee_data = callee.get('data', {})
                context += f"   {i}. {seed['name']} → calls → {callee['name']}\n"
                context += f"      File: {callee_data.get('file_path', 'unknown')}\n"
            context += "\n"
        else:
            context += "📤 OUTGOING CALLS: None (leaf function)\n\n"
        
        # Siblings - related functions in same class/module
        if neighborhood['siblings']:
            context += "👥 SIBLING FUNCTIONS (same parent, might share bugs):\n"
            for i, sibling in enumerate(neighborhood['siblings'][:5], 1):
                context += f"   {i}. {sibling['name']}\n"
            context += "\n"
        
        # Second hop - extended context
        second_hop = neighborhood['second_hop']
        if second_hop['caller_callers']:
            context += "🔄 2-HOP CALLERS (indirect sources):\n"
            for i, cc in enumerate(second_hop['caller_callers'][:3], 1):
                context += f"   {i}. {cc['name']} → {cc['via']} → {seed['name']}\n"
            context += "\n"
        
        if second_hop['callee_callees']:
            context += "🔄 2-HOP CALLEES (deep dependencies):\n"
            for i, cc in enumerate(second_hop['callee_callees'][:3], 1):
                context += f"   {i}. {seed['name']} → {cc['via']} → {cc['name']}\n"
            context += "\n"
        
        return context
    
    def retrieve_external_knowledge(self, bug_description: str, k: int = 3):
        """
        RAGFix: Retrieve from Stack Overflow nodes.
        
        Finds k most similar SO posts.
        """
        bug_embedding = self.encoder.encode(bug_description)
        
        so_candidates = []
        
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'stackoverflow':
                so_data = self.graph.nodes[node]['data']
                so_embedding = np.array(so_data['embedding'])
                
                similarity = np.dot(bug_embedding, so_embedding) / (
                    np.linalg.norm(bug_embedding) * np.linalg.norm(so_embedding)
                )
                
                so_candidates.append({
                    'so_id': node,
                    'similarity': similarity,
                    'data': so_data
                })
        
        so_candidates.sort(key=lambda x: x['similarity'], reverse=True)
        return so_candidates[:k]
    
    def check_requirement_violations(self, func_name: str, proposed_patch: str):
        """
        Traceability: Ensure patch doesn't violate requirements.
        
        Returns list of potentially violated requirements.
        """
        violations = []
        
        # Find requirements this function traces to
        for neighbor in self.graph.neighbors(func_name):
            edges = self.graph[func_name][neighbor]
            for edge_key, edge_data in edges.items():
                if edge_data.get('type') == 'TRACES_TO':
                    req_id = neighbor
                    req_data = self.graph.nodes[req_id]['data']
                    
                    # Check semantic similarity of patch vs requirement
                    patch_embedding = self.encoder.encode(proposed_patch)
                    req_embedding = np.array(req_data['embedding'])
                    
                    similarity = np.dot(patch_embedding, req_embedding) / (
                        np.linalg.norm(patch_embedding) * np.linalg.norm(req_embedding)
                    )
                    
                    # If very dissimilar, might violate
                    if similarity < 0.3:
                        violations.append({
                            'req_id': req_id,
                            'description': req_data['description'],
                            'risk_score': 1.0 - similarity
                        })
        
        return violations
    
    def retrieve_patch_history(self, bug_description: str, k: int = 3):
        """
        YOUR INNOVATION: Self-healing memory retrieval.
        
        Finds similar bugs from history (golden + failed patches).
        """
        bug_embedding = self.encoder.encode(bug_description)
        
        patches = []
        
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'patch_memory':
                patch_data = self.graph.nodes[node]['data']
                patch_embedding = np.array(patch_data['embedding'])
                
                similarity = np.dot(bug_embedding, patch_embedding) / (
                    np.linalg.norm(bug_embedding) * np.linalg.norm(patch_embedding)
                )
                
                patches.append({
                    'patch_id': node,
                    'similarity': similarity,
                    'success': patch_data['success'],
                    'code': patch_data['code'],
                    'data': patch_data
                })
        
        patches.sort(key=lambda x: x['similarity'], reverse=True)
        return patches[:k]
    
    # ========================================================================
    # PART 4: PROMPT FORMATTING (KGCompass Style)
    # ========================================================================
    
    def format_kgcompass_prompt(self, issue: str, top_20_functions: list,
                                external_knowledge: list = None,
                                patch_history: list = None):
        """
        KGCompass prompt with reasoning chains.
        
        Structure:
        1. Bug description (Issue)
        2. Reasoning chain (graph paths)
        3. Top 20 candidate functions
        4. External knowledge (RAGFix)
        5. Historical patches (YOUR innovation)
        """
        issue_data = self.graph.nodes[issue]['data']
        
        prompt = f"""=== BUG REPAIR TASK ===

ISSUE: {issue_data['title']}
{issue_data['description']}

=== REASONING CHAIN ===
(From KGCompass: Multi-hop graph traversal)

"""
        
        # Add top 20 candidates with reasoning
        for i, candidate in enumerate(top_20_functions[:20], 1):
            func_data = candidate['data']
            prompt += f"""
## Candidate {i}: {func_data['name']} (Score: {candidate['score']:.3f})
File: {func_data['file_path']}:{func_data['line_start']}
Semantic Similarity: {candidate['semantic_similarity']:.3f}
Graph Distance: {candidate['path_distance']} hops

```python
{func_data['signature']}
{func_data['body'][:300]}...
```
"""
        
        # Add external knowledge (RAGFix)
        if external_knowledge:
            prompt += "\n=== EXTERNAL KNOWLEDGE (Stack Overflow) ===\n"
            for so in external_knowledge:
                prompt += f"""
Similar Issue: {so['data']['question'][:200]}...
Solution: {so['data']['accepted_answer'][:200]}...
"""
        
        # Add patch history (YOUR innovation)
        if patch_history:
            prompt += "\n=== HISTORICAL PATCHES (Self-Healing Memory) ===\n"
            
            golden = [p for p in patch_history if p['success']]
            failed = [p for p in patch_history if not p['success']]
            
            if golden:
                prompt += "\n✅ SUCCESSFUL PATCHES (Use similar logic):\n"
                for p in golden[:2]:
                    prompt += f"- {p['code'][:150]}...\n"
            
            if failed:
                prompt += "\n❌ FAILED ATTEMPTS (Avoid these patterns):\n"
                for p in failed[:2]:
                    prompt += f"- {p['code'][:150]}... (Failed)\n"
        
        prompt += "\n\nGENERATE THE FIX:\n"
        
        return prompt
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def get_stats(self):
        return dict(self.stats)
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.graph, f)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            self.graph = pickle.load(f)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    graph = KGCompassGraphRAG()
    
    # Example: Build a small graph
    
    # 1. Add Issue (starting point)
    issue = IssueNode(
        issue_id="ISSUE-001",
        title="NoneType error in PDF extraction",
        description="When file_or_path is None, extract_text_from_pdf crashes with AttributeError",
        referenced_files=["utils.py"]
    )
    graph.add_issue_node(issue)
    
    # 2. Add File
    file = FileNode(file_path="utils.py", language="python")
    graph.add_file_node(file)
    graph.add_reference_edge("ISSUE-001", "utils.py")
    
    # 3. Add Function
    func = FunctionNode(
        name="extract_text_from_pdf",
        signature="def extract_text_from_pdf(file_or_path):",
        body="pdf = fitz.open(file_or_path)\nreturn text",
        file_path="utils.py",
        line_start=10,
        line_end=18
    )
    graph.add_function_node(func)
    graph.add_contain_edge("utils.py", "extract_text_from_pdf")
    
    # 4. Add Requirement (Traceability)
    req = RequirementNode(
        req_id="REQ-001",
        description="System must handle null file inputs gracefully",
        source="README.md",
        priority="high"
    )
    graph.add_requirement_node(req)
    graph.add_traces_to_edge("extract_text_from_pdf", "REQ-001", confidence=0.92)
    
    # 5. Retrieve top 20 candidates (KGCompass)
    top_20 = graph.retrieve_top_20_candidates("ISSUE-001")
    
    # 6. Format prompt
    prompt = graph.format_kgcompass_prompt("ISSUE-001", top_20)
    
    print(prompt[:1000])
    print("\n✅ Graph-RAG from papers implemented!")
    print(f"Stats: {graph.get_stats()}")