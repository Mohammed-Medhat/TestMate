# ============================================================
# ONE-CELL GRAPH BUILDER - Just Run It!
# ============================================================
# Clones your repo + DataDog from GitHub, builds graph, done!
# Works on Kaggle AND Colab - no uploads needed!
# ============================================================

import subprocess, os, ast, pickle, networkx as nx
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from collections import defaultdict

# --- Step 1: Clone repos ---
REPOS = {
    "TestMate": "https://github.com/Mohammed-Medhat/TestMate.git",
    "DataDog": "https://github.com/DataDog/integrations-core.git",
}
CLONE_BASE = "/tmp/fixmate_repos"
os.makedirs(CLONE_BASE, exist_ok=True)

for name, url in REPOS.items():
    dest = os.path.join(CLONE_BASE, name)
    if not os.path.exists(dest):
        print(f"📥 Cloning {name}...")
        subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True)
        print(f"   ✅ {name} cloned!")
    else:
        print(f"✅ {name} already cloned")

# Directories to parse
DIRS_TO_PARSE = [
    (os.path.join(CLONE_BASE, "TestMate", "PartB"), "FixMate"),
    (os.path.join(CLONE_BASE, "DataDog"), "DataDog__integrations-core"),
]
OUTPUT_PATH = "knowledge_graph.pkl"

# --- Step 2: Node Classes ---
@dataclass
class FunctionNode:
    name: str; signature: str; body: str; file_path: str
    class_name: Optional[str] = None; line_start: int = 0; line_end: int = 0
    docstring: Optional[str] = None; paths: List[str] = field(default_factory=list)
    def dict(self): return self.__dict__

@dataclass
class ClassNode:
    name: str; file_path: str; methods: List[str] = field(default_factory=list)
    parent_class: Optional[str] = None; docstring: Optional[str] = None
    def dict(self): return self.__dict__

@dataclass
class FileNode:
    path: str; imports: List[str] = field(default_factory=list)
    def dict(self): return self.__dict__

# --- Step 3: CFG Path Extractor ---
class CFGPathExtractor:
    def __init__(self, max_paths=5):
        self.max_paths = max_paths; self.paths = []
    
    def trace_paths(self, func_node):
        self.paths = []
        self._trace(func_node.body, [])
        unhappy = sorted([p for p in self.paths if 'RAISES' in p], key=len, reverse=True)
        happy = sorted([p for p in self.paths if 'RAISES' not in p], key=len, reverse=True)
        return (unhappy + happy)[:self.max_paths]
    
    def _trace(self, nodes, path):
        for n in nodes:
            if isinstance(n, ast.If):
                try: c = ast.unparse(n.test)[:50]
                except: c = "cond"
                self._trace(n.body, path + [f"IF ({c})"])
                if n.orelse: self._trace(n.orelse, path + [f"ELSE ({c})"])
            elif isinstance(n, ast.Raise):
                try: e = ast.unparse(n.exc)[:30] if n.exc else "Exception"
                except: e = "Exception"
                self.paths.append(" -> ".join(path + [f"RAISES {e}"]))
            elif isinstance(n, ast.Return):
                try: v = ast.unparse(n.value)[:30] if n.value else "None"
                except: v = "value"
                self.paths.append(" -> ".join(path + [f"RETURN {v}"]))
            elif isinstance(n, ast.For): self._trace(n.body, path + ["FOR loop"])
            elif isinstance(n, ast.While): self._trace(n.body, path + ["WHILE loop"])
            elif isinstance(n, ast.Try):
                self._trace(n.body, path + ["TRY"])
                for h in n.handlers:
                    try: etype = h.type.id if hasattr(h.type, 'id') else "Exception"
                    except: etype = "Exception"
                    self._trace(h.body, path + [f"EXCEPT {etype}"])

# --- Step 4: Parser ---
class CodeParser:
    def __init__(self): self.cfg = CFGPathExtractor()
    
    def parse_file(self, fpath):
        try:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
            tree = ast.parse(content)
            result = {'file_path': fpath, 'imports': [], 'classes': [], 'functions': []}
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    for a in n.names: result['imports'].append(a.name)
                elif isinstance(n, ast.ImportFrom) and n.module:
                    result['imports'].append(n.module)
            for n in tree.body:
                if isinstance(n, ast.ClassDef):
                    methods = [i.name for i in n.body if isinstance(i, ast.FunctionDef)]
                    parent = None
                    if n.bases:
                        try: parent = ast.unparse(n.bases[0])
                        except: pass
                    result['classes'].append({'name': n.name, 'file_path': fpath, 'methods': methods,
                                              'parent_class': parent, 'docstring': ast.get_docstring(n)})
                elif isinstance(n, ast.FunctionDef):
                    try: args = ast.unparse(n.args)
                    except: args = ""
                    try: body = ast.unparse(n)
                    except: body = ""
                    paths = self.cfg.trace_paths(n)
                    result['functions'].append({
                        'name': n.name, 'signature': f"def {n.name}({args})", 'body': body[:2000],
                        'file_path': fpath, 'class_name': None, 'line_start': n.lineno,
                        'line_end': n.end_lineno or n.lineno, 'docstring': ast.get_docstring(n),
                        'paths': paths
                    })
            return result
        except: return None

# --- Step 5: Build Graph ---
print("\n" + "="*60)
print("🔨 BUILDING KNOWLEDGE GRAPH WITH CFG PATHS")
print("="*60)

graph = nx.MultiDiGraph()
parser = CodeParser()
stats = defaultdict(int)

for dir_path, name in DIRS_TO_PARSE:
    if not os.path.exists(dir_path):
        print(f"⚠️ Skipping {name}: not found at {dir_path}")
        continue
    
    py_files = [f for f in Path(dir_path).rglob("*.py")
                if not any(x in str(f) for x in ['__pycache__', '.git', 'test_', '_test.py'])]
    print(f"\n📁 {name}: {len(py_files)} Python files")
    
    for i, fpath in enumerate(py_files):
        if i % 200 == 0: print(f"   Processing {i}/{len(py_files)}...")
        result = parser.parse_file(str(fpath))
        if not result: continue
        
        graph.add_node(result['file_path'], type='file',
                       data=FileNode(result['file_path'], result['imports']).dict())
        stats['files'] += 1
        
        for c in result['classes']:
            graph.add_node(c['name'], type='class', data=ClassNode(**c).dict())
            graph.add_edge(result['file_path'], c['name'], type='CONTAINS')
            stats['classes'] += 1
        
        for f in result['functions']:
            fn = FunctionNode(name=f['name'], signature=f['signature'], body=f['body'],
                              file_path=f['file_path'], class_name=f.get('class_name'),
                              line_start=f['line_start'], line_end=f['line_end'],
                              docstring=f.get('docstring'), paths=f.get('paths', []))
            graph.add_node(f['name'], type='function', data=fn.dict())
            graph.add_edge(result['file_path'], f['name'], type='CONTAINS')
            stats['functions'] += 1

# --- Step 6: Save & Verify ---
with open(OUTPUT_PATH, 'wb') as f: pickle.dump(graph, f)

print(f"\n{'='*60}")
print(f"✅ GRAPH BUILT SUCCESSFULLY!")
print(f"   Nodes: {graph.number_of_nodes()}")
print(f"   Edges: {graph.number_of_edges()}")
print(f"   Files: {stats['files']}, Classes: {stats['classes']}, Functions: {stats['functions']}")

paths_count = 0
for nid, data in graph.nodes(data=True):
    if data.get('type') == 'function':
        p = data.get('data', {}).get('paths', [])
        if p:
            paths_count += 1
            if paths_count <= 3:
                print(f"\n   📊 {nid}: {p[0]}")
print(f"\n   Functions with CFG paths: {paths_count}")
print(f"\n📥 Download: {OUTPUT_PATH}")

try:
    from IPython.display import FileLink
    display(FileLink(OUTPUT_PATH))
except: pass
