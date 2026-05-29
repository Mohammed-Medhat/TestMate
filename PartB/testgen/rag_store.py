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
import time
from collections import OrderedDict
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
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_test_examples_quality_src "
        "ON test_examples(quality_score, source_file)"
    )

    # Tier 4 — self-healing columns. ALTER TABLE is idempotent: we check which
    # columns already exist via PRAGMA table_info and only add the missing ones.
    def _add_column_if_missing(table: str, col: str, col_def: str) -> None:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if col not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")

    for _table in ("test_examples", "bad_examples"):
        _add_column_if_missing(_table, "use_count",     "INTEGER DEFAULT 0")
        _add_column_if_missing(_table, "success_count", "INTEGER DEFAULT 0")
        _add_column_if_missing(_table, "last_used_at",  "TIMESTAMP")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_test_examples_last_used "
        "ON test_examples(last_used_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_bad_examples_last_used "
        "ON bad_examples(last_used_at)"
    )

    # Startup decay: gently penalise test_examples rows that haven't been
    # touched in 30+ days (5% quality decay per run, falls below the 40-floor
    # in ~10 dormant runs). Keeps the audit trail while pushing stale entries
    # out of retrieval naturally.
    conn.execute("""
        UPDATE test_examples
        SET quality_score = quality_score * 0.95
        WHERE last_used_at IS NULL
           OR last_used_at < datetime('now', '-30 days')
    """)

    # Hard-prune bad_examples that were shown 3+ times but never helped
    # and haven't been touched in 60 days — they actively waste prompt tokens.
    conn.execute("""
        DELETE FROM bad_examples
        WHERE last_used_at IS NOT NULL
          AND last_used_at < datetime('now', '-60 days')
          AND success_count = 0
          AND use_count >= 3
    """)

    conn.commit()
    conn.close()

def embed(text: str) -> np.ndarray:
    return get_embedder().encode(text, normalize_embeddings=True)

# Cache query embeddings inside retrieve_similar — same (class, method) pair
# is queried many times across the autonomous_loop retries for a single target.
# OrderedDict + (emb, timestamp) gives us bounded LRU + TTL expiry without an
# external dep. Old entries expire after 1h; cache caps at 512 entries.
_QUERY_EMB_CACHE: "OrderedDict[str, tuple[np.ndarray, float]]" = OrderedDict()
_QUERY_EMB_CACHE_MAX = 512
_QUERY_EMB_CACHE_TTL_SEC = 3600


def _cached_query_embed(key: str) -> np.ndarray:
    now = time.time()
    entry = _QUERY_EMB_CACHE.get(key)
    if entry is not None:
        emb, ts = entry
        if now - ts <= _QUERY_EMB_CACHE_TTL_SEC:
            # Mark as most-recently-used.
            _QUERY_EMB_CACHE.move_to_end(key)
            return emb
        # Expired — drop and re-encode.
        del _QUERY_EMB_CACHE[key]
    emb = embed(key)
    _QUERY_EMB_CACHE[key] = (emb, now)
    # Evict oldest entries past the size cap.
    while len(_QUERY_EMB_CACHE) > _QUERY_EMB_CACHE_MAX:
        _QUERY_EMB_CACHE.popitem(last=False)
    return emb

# ── Tier 4: self-healing retrieval tracking ───────────────────────────────
# Keyed by target_signature; maps to the list of row IDs (and for bad examples,
# the failure_category) that were retrieved for that target during this process.
# Cleared by the feedback functions after they flush to the DB.
_RECENT_GOOD_RETRIEVAL: "dict[str, list[int]]" = {}
_RECENT_BAD_RETRIEVAL:  "dict[str, list[tuple[int, str]]]" = {}


def _now_ts() -> str:
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def feedback_test_accepted(target_signature: str) -> None:
    """Call when a test is accepted (passes BES gate + pytest).
    Marks retrieved good examples as successes."""
    ids = _RECENT_GOOD_RETRIEVAL.pop(target_signature, [])
    if not ids:
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        now = _now_ts()
        for row_id in ids:
            conn.execute(
                "UPDATE test_examples SET use_count = use_count + 1, "
                "success_count = success_count + 1, last_used_at = ? WHERE id = ?",
                (now, row_id),
            )
        conn.commit()
        conn.close()
    except Exception:
        pass


def feedback_test_rejected(target_signature: str) -> None:
    """Call when a test is rejected (BES fail, pytest fail, tautology, etc.).
    Marks retrieved good examples as used-but-not-successful."""
    ids = _RECENT_GOOD_RETRIEVAL.pop(target_signature, [])
    if not ids:
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        now = _now_ts()
        for row_id in ids:
            conn.execute(
                "UPDATE test_examples SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
                (now, row_id),
            )
        conn.commit()
        conn.close()
    except Exception:
        pass


def feedback_bad_helped(target_signature: str) -> None:
    """Call when a test is accepted after warnings were shown.
    The bad examples for this target successfully steered the model away."""
    entries = _RECENT_BAD_RETRIEVAL.pop(target_signature, [])
    if not entries:
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        now = _now_ts()
        for row_id, _ in entries:
            conn.execute(
                "UPDATE bad_examples SET use_count = use_count + 1, "
                "success_count = success_count + 1, last_used_at = ? WHERE id = ?",
                (now, row_id),
            )
        conn.commit()
        conn.close()
    except Exception:
        pass


def feedback_bad_didnt_help(target_signature: str, failure_category: str) -> None:
    """Call when a test is rejected. Counts use for all retrieved bad examples;
    only increments success for those whose failure_category did NOT match
    (those warnings weren't on trial — they still helped by staying relevant)."""
    entries = _RECENT_BAD_RETRIEVAL.pop(target_signature, [])
    if not entries:
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        now = _now_ts()
        for row_id, cat in entries:
            if cat == failure_category:
                # Warning shown, model repeated the same mistake — no success
                conn.execute(
                    "UPDATE bad_examples SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
                    (now, row_id),
                )
            else:
                # Unrelated warning — treat as mildly helpful (didn't hurt)
                conn.execute(
                    "UPDATE bad_examples SET use_count = use_count + 1, "
                    "success_count = success_count + 1, last_used_at = ? WHERE id = ?",
                    (now, row_id),
                )
        conn.commit()
        conn.close()
    except Exception:
        pass


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

def retrieve_bad_examples(method_name: str, class_name: str,
                          target_signature: str = "") -> list:
    """Retrieve failed test examples for this method type.

    If target_signature is provided, populates _RECENT_BAD_RETRIEVAL so
    feedback_bad_helped / feedback_bad_didnt_help can update the trust stats.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            """SELECT id, test_code, failure_reason, failure_category, survived_mutants,
                      use_count, success_count
               FROM bad_examples
               WHERE method_name = ? OR class_name = ?
               ORDER BY created_at DESC LIMIT 10""",
            (method_name, class_name or "")
        ).fetchall()
        conn.close()
        result = []
        tracked_ids: list[tuple[int, str]] = []
        for r in rows:
            row_id, code, reason, cat, mutants_json, use_count, success_count = r
            # Trust gate: if shown 5+ times with <30% success rate, skip it —
            # the warning isn't landing and wastes prompt tokens.
            if use_count >= 5:
                trust = (success_count + 1) / (use_count + 2)
                if trust < 0.3:
                    continue
            ex = {
                "id": row_id,
                "test_code": code,
                "failure_reason": reason,
                "failure_category": cat,
            }
            if mutants_json:
                try:
                    ex["survived_mutants"] = json.loads(mutants_json)
                except Exception:
                    ex["survived_mutants"] = []
            result.append(ex)
            tracked_ids.append((row_id, cat or ""))
            if len(result) >= 3:
                break
        if target_signature and tracked_ids:
            _RECENT_BAD_RETRIEVAL[target_signature] = tracked_ids
        return result
    except Exception:
        return []

def retrieve_similar(method_name: str, class_name: str, top_k: int = 3,
                     source_file: str = None,
                     target_signature: str = "") -> list:
    """Retrieve semantically similar high-quality test examples.

    If target_signature is provided, populates _RECENT_GOOD_RETRIEVAL so
    feedback_test_accepted / feedback_test_rejected can update trust stats.
    """
    conn = sqlite3.connect(DB_PATH)
    # Hard filters pushed into SQL:
    #   quality_score < 40   → already filtered at query level
    #   source_file == ours  → filtered at query level when source_file provided
    if source_file:
        rows = conn.execute(
            "SELECT id, target_signature, test_code, quality_score, embedding, "
            "class_name, method_name, source_file, use_count, success_count "
            "FROM test_examples "
            "WHERE quality_score >= 40 AND (source_file IS NULL OR source_file != ?)",
            (source_file,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, target_signature, test_code, quality_score, embedding, "
            "class_name, method_name, source_file, use_count, success_count "
            "FROM test_examples "
            "WHERE quality_score >= 40"
        ).fetchall()
    conn.close()

    if not rows:
        return []

    query = _cached_query_embed(f"{class_name}.{method_name}" if class_name else method_name)

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
    for row_id, sig, code, score, emb_bytes, cls, meth, src_file, use_count, success_count in rows:
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

        # Trust-factor tiebreak: Laplace-smoothed empirical success rate.
        # 8/10 success → full boost; never-used → neutral (0.5); always-fail → slight penalty.
        trust_factor = (success_count + 1) / (use_count + 2)
        adjusted_quality = score * (0.5 + trust_factor)  # [0.5*score .. 1.5*score]
        quality_tiebreak = 0.001 * (adjusted_quality - score)

        final_score = (embedding_sim + class_bonus + method_bonus
                       + related_bonus - cross_class_penalty + quality_tiebreak)

        scored.append({
            "id": row_id,
            "target_signature": sig,
            "test_code": code,
            "quality_score": score,
            "similarity": final_score,
            "class_name": cls or "",
            "method_name": meth or "",
        })

    scored.sort(key=lambda x: (x["similarity"], x["quality_score"]), reverse=True)
    top = scored[:top_k]

    if target_signature and top:
        _RECENT_GOOD_RETRIEVAL[target_signature] = [r["id"] for r in top]

    return top

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
