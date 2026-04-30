import os
import warnings
import logging

# Must be before ANY other imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = "testmate_rag.db"
EMBED_MODEL = "all-MiniLM-L6-v2"  # small, fast, good enough

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS test_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_signature TEXT,     -- e.g. "CaseInsensitiveDict.__eq__"
            method_name TEXT,          -- e.g. "__eq__"
            class_name TEXT,           -- e.g. "CaseInsensitiveDict"
            test_code TEXT,
            quality_score REAL,        -- 0-100 composite
            coverage_lines INTEGER,
            passed_mutation INTEGER,   -- 0 or 1
            coverage_pattern TEXT,     -- e.g. "branch:if_not_initialized"
            source_file TEXT,          -- e.g. "structures.py"
            embedding BLOB,            -- serialized numpy array
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bad_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_signature TEXT,
            method_name TEXT,
            class_name TEXT,
            test_code TEXT,
            failure_reason TEXT,
            failure_category TEXT,
            survived_mutants TEXT,
            source_file TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS probe_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            import_path TEXT,
            class_name TEXT,
            method_name TEXT,
            probed_value TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(import_path, class_name, method_name)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS source_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT,
            method_name TEXT,
            pattern_type TEXT,
            pattern_description TEXT,
            dependencies TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS method_dependencies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT,
            method_name TEXT,
            depends_on_method TEXT,
            dependency_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def embed(text: str) -> np.ndarray:
    return get_embedder().encode(text, normalize_embeddings=True)

def store_example(target_signature, method_name, class_name,
                  test_code, quality_score, coverage_lines,
                  passed_mutation, source_file, coverage_pattern=None):
    conn = sqlite3.connect(DB_PATH)

    # Check if better example already exists
    existing = conn.execute(
        "SELECT id, quality_score FROM test_examples WHERE target_signature = ?",
        (target_signature,)
    ).fetchone()

    emb = embed(f"{target_signature} {method_name} {test_code[:200]}")
    emb_bytes = emb.astype(np.float32).tobytes()

    if existing:
        if quality_score > existing[1]:
            # Replace with better example
            conn.execute("""
                UPDATE test_examples
                SET test_code=?, quality_score=?, coverage_lines=?,
                    passed_mutation=?, coverage_pattern=?, embedding=?
                WHERE id=?
            """, (test_code, quality_score, coverage_lines,
                  int(passed_mutation), coverage_pattern, emb_bytes, existing[0]))
            print(f"   📚 RAG updated: {target_signature} ({existing[1]:.1f} → {quality_score:.1f})")
        else:
            print(f"   📚 RAG kept existing: {target_signature} (existing {existing[1]:.1f} >= new {quality_score:.1f})")
    else:
        conn.execute("""
            INSERT INTO test_examples
            (target_signature, method_name, class_name, test_code,
             quality_score, coverage_lines, passed_mutation, coverage_pattern,
             source_file, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (target_signature, method_name, class_name, test_code,
              quality_score, coverage_lines, int(passed_mutation),
              coverage_pattern, source_file, emb_bytes))
        print(f"   📚 RAG stored: {target_signature} (quality {quality_score:.1f})")

    conn.commit()
    conn.close()

def store_bad_example(target_signature, method_name, class_name,
                      test_code, failure_reason, source_file,
                      failure_category=None, survived_mutants=None):
    """Store a failed test attempt so the model learns what NOT to do."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        # Ensure table exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bad_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_signature TEXT,
                method_name TEXT,
                class_name TEXT,
                test_code TEXT,
                failure_reason TEXT,
                failure_category TEXT,
                survived_mutants TEXT,
                source_file TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Store up to 3 bad examples per target — avoid flooding
        existing_count = conn.execute(
            "SELECT COUNT(*) FROM bad_examples WHERE target_signature = ?",
            (target_signature,)
        ).fetchone()[0]

        if existing_count < 3:
            survived_mutants_str = json.dumps(survived_mutants) if survived_mutants else None
            conn.execute("""
                INSERT INTO bad_examples
                (target_signature, method_name, class_name, test_code, failure_reason,
                 failure_category, survived_mutants, source_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (target_signature, method_name, class_name,
                  test_code, failure_reason[:300], failure_category,
                  survived_mutants_str, source_file))
            conn.commit()
    except Exception:
        pass
    finally:
        if conn:
            conn.close()

def retrieve_bad_examples(method_name: str, class_name: str) -> list:
    """Retrieve failed test examples for this method type."""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Match by method name — same method type across classes fails similarly
        rows = conn.execute(
            """SELECT test_code, failure_reason, failure_category, survived_mutants
               FROM bad_examples
               WHERE method_name = ? OR class_name = ?
               ORDER BY created_at DESC LIMIT 3""",
            (method_name, class_name or "")
        ).fetchall()
        conn.close()
        result = []
        for r in rows:
            ex = {
                "test_code": r[0],
                "failure_reason": r[1],
                "failure_category": r[2],
            }
            if r[3]:
                try:
                    ex["survived_mutants"] = json.loads(r[3])
                except:
                    ex["survived_mutants"] = []
            result.append(ex)
        return result
    except Exception:
        return []

def retrieve_similar(method_name: str, class_name: str, top_k: int = 3,
                     source_file: str = None) -> list:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT target_signature, test_code, quality_score, embedding, class_name, method_name, source_file FROM test_examples"
    ).fetchall()
    conn.close()
    
    if not rows:
        return []
    
    query = embed(f"{class_name}.{method_name}" if class_name else method_name)
    
    # Related dunder method groups — similar test patterns
    _RELATED_DUNDERS = {
        "__getitem__": {"__setitem__", "__delitem__", "__contains__"},
        "__setitem__": {"__getitem__", "__delitem__", "__contains__"},
        "__delitem__": {"__getitem__", "__setitem__"},
        "__contains__": {"__getitem__", "__setitem__"},
        "__eq__": {"__ne__", "__hash__"},
        "__ne__": {"__eq__", "__hash__"},
        "__lt__": {"__gt__", "__le__", "__ge__", "__eq__"},
        "__gt__": {"__lt__", "__le__", "__ge__", "__eq__"},
        "__le__": {"__lt__", "__gt__", "__ge__"},
        "__ge__": {"__lt__", "__gt__", "__le__"},
        "__len__": {"__bool__", "__iter__"},
        "__iter__": {"__len__", "__getitem__"},
        "__str__": {"__repr__"},
        "__repr__": {"__str__"},
    }
    related_methods = _RELATED_DUNDERS.get(method_name, set())
    
    scored = []
    for sig, code, score, emb_bytes, cls, meth, src_file in rows:
        # Skip low-quality examples (< 40)
        if score < 40.0:
            continue
        
        # Skip same-file examples to avoid self-reinforcing patterns
        if source_file and src_file and src_file == source_file:
            continue
        
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        embedding_sim = float(np.dot(query, emb))
        
        # Bonus for same class
        class_bonus = 0.2 if (cls or "") == (class_name or "") else 0.0
        
        # Bonus for same method name pattern
        method_bonus = 0.15 if (meth or "") == method_name else 0.0
        
        # Bonus for related dunder methods (e.g., __getitem__ ↔ __setitem__)
        related_bonus = 0.10 if (meth or "") in related_methods else 0.0
        
        # Penalty for cross-class contamination
        cross_class_penalty = 0.0
        if class_name and (cls or "") != class_name and (cls or "") != "":
            cross_class_penalty = 0.3
        
        final_score = (embedding_sim + class_bonus + method_bonus
                       + related_bonus - cross_class_penalty)
        
        scored.append({
            "target_signature": sig,
            "test_code": code,
            "quality_score": score,
            "similarity": final_score,
            "class_name": cls or "",
            "method_name": meth or "",
        })
    
    scored.sort(key=lambda x: (x["similarity"], x["quality_score"]), reverse=True)
    return scored[:top_k]

def store_probe_value(import_path: str, class_name: str, method_name: str, probed_value: str):
    """Store a probed runtime value for persistent cache."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT OR REPLACE INTO probe_cache
            (import_path, class_name, method_name, probed_value)
            VALUES (?, ?, ?, ?)
        """, (import_path, class_name or "", method_name, probed_value))
        conn.commit()
        conn.close()
    except Exception:
        pass

def retrieve_probe_value(import_path: str, class_name: str, method_name: str) -> str:
    """Retrieve cached probed value if available."""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT probed_value FROM probe_cache WHERE import_path=? AND class_name=? AND method_name=?",
            (import_path, class_name or "", method_name)
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None

def store_source_pattern(class_name: str, method_name: str, pattern_type: str,
                        pattern_description: str, dependencies: str = None):
    """Store source code pattern for matching with test patterns."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT OR REPLACE INTO source_patterns
            (class_name, method_name, pattern_type, pattern_description, dependencies)
            VALUES (?, ?, ?, ?, ?)
        """, (class_name, method_name, pattern_type, pattern_description, dependencies))
        conn.commit()
        conn.close()
    except Exception:
        pass

def retrieve_source_patterns(class_name: str, method_name: str) -> list:
    """Retrieve source patterns for a method."""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT pattern_type, pattern_description, dependencies FROM source_patterns WHERE class_name=? AND method_name=?",
            (class_name, method_name)
        ).fetchall()
        conn.close()
        return [{"pattern_type": r[0], "description": r[1], "dependencies": r[2]} for r in rows]
    except Exception:
        return []

def store_method_dependency(class_name: str, method_name: str,
                           depends_on_method: str, dependency_type: str):
    """Store method dependency for test generation."""
    try:
        conn = sqlite3.connect(DB_PATH)
        # Check if already exists
        existing = conn.execute(
            "SELECT id FROM method_dependencies WHERE class_name=? AND method_name=? AND depends_on_method=?",
            (class_name, method_name, depends_on_method)
        ).fetchone()
        if not existing:
            conn.execute("""
                INSERT INTO method_dependencies
                (class_name, method_name, depends_on_method, dependency_type)
                VALUES (?, ?, ?, ?)
            """, (class_name, method_name, depends_on_method, dependency_type))
        conn.commit()
        conn.close()
    except Exception:
        pass

def retrieve_method_dependencies(class_name: str, method_name: str) -> list:
    """Retrieve methods that must be called before this method."""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT depends_on_method, dependency_type FROM method_dependencies WHERE class_name=? AND method_name=?",
            (class_name, method_name)
        ).fetchall()
        conn.close()
        return [{"method": r[0], "type": r[1]} for r in rows]
    except Exception:
        return []
