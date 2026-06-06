# ============================================================
# TestMate — End-to-End Pipeline Evaluation  (Kaggle T4)
# ============================================================
# Conditions:
#   E2 — PartB tests (no reqs)   + base Qwen APR
#   E3 — PartB tests (no reqs)   + TestMate APR  (LoRA)
#   E4 — PartB tests (desc reqs) + TestMate APR  (LoRA)  ← full pipeline
#   E5 — E4 + gap-fill second pass                       ← best mode
#
# Usage (Kaggle notebook):
#   !pip install -q transformers peft bitsandbytes accelerate pytest
#   !git clone https://github.com/<user>/TestMate /kaggle/working/TestMate
#   %cd /kaggle/working/TestMate/PartB/testgen/training
#   !python pipeline_eval_kaggle.py \
#       --condition E4 \
#       --dataset   /kaggle/input/testmate/humaneval_eval_dataset.json \
#       --partc_adapter /kaggle/input/testmate-partc/adapter   # E3/E4/E5 only
#
# Outputs → /kaggle/working/pipeline_results/{CONDITION}/
#   results.json    per-sample metrics (detection + apparent_repair)
#   summary.json    aggregate rates
#   artifacts/      {id}_test.py  +  {id}_repaired.py  for local hidden-test eval
# ============================================================

from __future__ import annotations

import os
import sys
import ast
import re
import gc
import json
import time
import shutil
import argparse
import tempfile
import subprocess
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("pipeline_eval")

# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--condition",     required=True, choices=["E2","E3","E4","E5"])
    p.add_argument("--dataset",       required=True, help="Path to humaneval_eval_dataset.json")
    p.add_argument("--partc_adapter", default=None,  help="LoRA adapter dir for PartC (E3/E4/E5)")
    p.add_argument("--limit",         type=int, default=None, help="Evaluate only first N samples")
    p.add_argument("--output_dir",    default="/kaggle/working/pipeline_results")
    return p.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_dataset(path: str, limit: Optional[int]) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    valid = [s for s in data
             if s.get("buggy_code") and s.get("tests") and s.get("description") and s.get("entry_point")]
    if limit:
        valid = valid[:limit]
    log.info("Loaded %d valid samples", len(valid))
    return valid


# ── Model loading ─────────────────────────────────────────────────────────────

def load_base_qwen(model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True,garbage_collection_threshold:0.6")
    log.info("Loading %s (4-bit NF4)…", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map={"": 0},
        trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float16,
    )
    model.eval()
    import torch as _t
    log.info("Loaded — %.1f GB VRAM", _t.cuda.memory_allocated() / 1e9)
    return model, tokenizer


def attach_lora(model, adapter_path: str):
    from peft import PeftModel
    posix = Path(adapter_path).resolve().as_posix()
    log.info("Attaching LoRA adapter from %s", posix)
    return PeftModel.from_pretrained(model, posix)


def free_gpu(model):
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log.info("GPU memory freed")


# ── Part A (inline): description → requirements ───────────────────────────────

def description_to_requirements(description: str) -> list[dict]:
    """Treat the full problem description as one labelled requirement."""
    return [{"label": 1, "text": description.strip()}]


# ── Part B: test generation ───────────────────────────────────────────────────

_TESTGEN_SYSTEM = "You write production-quality pytest test files. Output ONLY Python code."

_TESTGEN_PROMPT = """\
You are a senior Python test engineer. Generate a complete pytest test file for the function described below.

{req_section}FUNCTION (import it as: from solution import {entry_point}):
```python
{signature}
```

Rules:
- `from solution import {entry_point}` must be the first import.
- Use pytest only — no unittest.
- Cover: happy path, edge cases, boundary values, error/exception cases.
- Each test function must include at least one assert statement.
- Output ONLY the Python code, no markdown fences.

Generate the complete test file:"""


def _extract_signature(buggy_code: str, entry_point: str) -> str:
    """Pull the function definition + docstring from buggy_code (bug-free structure)."""
    try:
        tree = ast.parse(buggy_code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == entry_point:
                # Reconstruct header + docstring only (drop body after docstring)
                lines = buggy_code.splitlines()
                start = node.lineno - 1
                end   = node.end_lineno
                fn_src = "\n".join(lines[start:end])
                # Keep just up to the first non-docstring statement
                try:
                    fn_tree = ast.parse(fn_src)
                    fn_node = fn_tree.body[0]
                    sig_lines = [lines[start]]  # def line
                    if (fn_node.body and isinstance(fn_node.body[0], ast.Expr)
                            and isinstance(fn_node.body[0].value, ast.Constant)):
                        sig_end = fn_node.body[0].end_lineno
                        sig_lines = lines[start: start + sig_end]
                    return "\n".join(sig_lines) + "\n    ..."
                except Exception:
                    return fn_src[:600]
    except Exception:
        pass
    return f"def {entry_point}(...):\n    ..."


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _generate_tests(model, tokenizer, sample: dict,
                    requirements: list[dict]) -> tuple[str, float]:
    import torch
    entry_point = sample["entry_point"]
    signature   = _extract_signature(sample["buggy_code"], entry_point)

    req_section = ""
    if requirements:
        req_text = "\n".join(f"  • {r['text']}" for r in requirements if r.get("label") == 1)
        if req_text:
            req_section = f"SPECIFICATION:\n{req_text}\n\n"

    prompt = _TESTGEN_PROMPT.format(
        req_section=req_section,
        entry_point=entry_point,
        signature=signature,
    )
    messages = [
        {"role": "system", "content": _TESTGEN_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=1024, do_sample=True,
            temperature=0.3, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _strip_fences(raw), elapsed


# ── Gap-fill (E5): extra tests for uncovered requirements ─────────────────────

_GAPFILL_PROMPT = """\
You are a Python test engineer. The following requirements were NOT covered by the initial test suite.
Write ADDITIONAL pytest test functions (no class, no imports — just the test functions) that specifically target these requirements.

Uncovered requirements:
{uncovered}

Function:
```python
{signature}
```

The function is imported as: from solution import {entry_point}

Output ONLY the additional test functions (no imports, no markdown):"""


def _gapfill_tests(model, tokenizer, sample: dict,
                   uncovered_reqs: list[dict], existing_test: str) -> str:
    import torch
    entry_point = sample["entry_point"]
    signature   = _extract_signature(sample["buggy_code"], entry_point)
    uncovered   = "\n".join(f"  • {r['text']}" for r in uncovered_reqs)

    prompt = _GAPFILL_PROMPT.format(
        uncovered=uncovered, signature=signature, entry_point=entry_point,
    )
    messages = [
        {"role": "system", "content": _TESTGEN_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=512, do_sample=True,
            temperature=0.5, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    extra = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    extra = _strip_fences(extra)
    return existing_test + "\n\n# --- gap-fill ---\n" + extra


def _check_req_coverage(requirements: list[dict], test_code: str) -> list[dict]:
    """Return requirements whose keywords don't appear in the test code."""
    noise = {"the","system","shall","must","accept","return","provide","support",
             "allow","when","that","from","with","this","have","will","each",
             "only","than","into","and","for","not","any","all","its","may",
             "can","are","was","has","been","also","does","use","used","set","get"}
    test_lower = test_code.lower()
    uncovered = []
    for r in requirements:
        if r.get("label") != 1:
            continue
        words = re.findall(r"\b[A-Za-z][a-zA-Z_]{2,}\b", r["text"])
        kws   = [w.lower() for w in words if w.lower() not in noise][:6]
        if not kws:
            continue
        hits = sum(1 for kw in kws if kw in test_lower)
        if hits < min(2, len(kws)):
            uncovered.append(r)
    return uncovered


# ── Sandbox runner ────────────────────────────────────────────────────────────

def _run_sandbox(source_code: str, test_code: str, timeout: int = 30) -> dict:
    """Write solution.py + test_solution.py in a tmpdir, run pytest, return result."""
    tmpdir = tempfile.mkdtemp(prefix="tm_pipeline_")
    try:
        (Path(tmpdir) / "solution.py").write_text(source_code, encoding="utf-8")
        (Path(tmpdir) / "test_solution.py").write_text(test_code, encoding="utf-8")
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "test_solution.py",
             "-v", "--tb=short", "--no-header", "-q"],
            capture_output=True, text=True, cwd=tmpdir, timeout=timeout,
        )
        out = r.stdout + r.stderr
        passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
        failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
        errors = int(m.group(1)) if (m := re.search(r"(\d+) error",  out)) else 0
        total  = passed + failed + errors
        return {
            "returncode":  r.returncode,
            "passed":      passed,
            "failed":      failed,
            "errors":      errors,
            "total":       total,
            "all_passed":  r.returncode == 0 and total > 0,
            "any_failed":  failed > 0 or errors > 0,
            "output":      out[-2000:],
        }
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "passed": 0, "failed": 0, "errors": 1,
                "total": 0, "all_passed": False, "any_failed": False,
                "output": "TIMEOUT"}
    except Exception as exc:
        return {"returncode": -1, "passed": 0, "failed": 0, "errors": 1,
                "total": 0, "all_passed": False, "any_failed": False,
                "output": str(exc)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Part C: APR loop ──────────────────────────────────────────────────────────

_APR_SYSTEM = ("Expert APR agent. Output ONLY the complete fixed Python function(s). "
               "No line numbers, no explanation, no markdown.")

_APR_PROMPT = """\
ISSUE: {description}

TRACE:
{trace}

BUGGY CODE:
```python
{buggy_code}
```

Output ONLY the fixed Python function(s):"""

_APR_RETRY_PROMPT = """\
ISSUE: {description}

TRACE:
{trace}

BUGGY CODE:
```python
{buggy_code}
```

PREVIOUS ATTEMPT FAILED. Try a completely different approach.
Output ONLY the fixed Python function(s):"""


def _extract_clean_code(raw: str) -> str:
    for marker in ("BUGGY CODE", "FIXED CODE", "ISSUE:", "TRACE:"):
        if marker in raw:
            raw = raw[:raw.index(marker)]
    m = re.search(r"```(?:python)?(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = raw.split("\n")
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith(("def ", "class ", "@")):
            start = i
            break
    return "\n".join(lines[start:]).strip()


def _ast_patch(original: str, model_output: str) -> Optional[str]:
    clean = _extract_clean_code(model_output)
    try:
        new_ast = ast.parse(clean)
    except SyntaxError:
        return None
    new_fns = {n.name: n for n in new_ast.body
               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    if not new_fns:
        return None
    try:
        orig_ast = ast.parse(original)
    except SyntaxError:
        return clean
    modified = False
    for i, node in enumerate(orig_ast.body):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in new_fns:
            orig_ast.body[i] = new_fns[node.name]
            modified = True
    return ast.unparse(orig_ast) if modified else None


def _generate_repair(model, tokenizer, sample: dict, fail_output: str,
                     attempt: int, temps: list = (0.2, 0.5, 0.8)) -> str:
    import torch
    trace_lines = [l.strip() for l in fail_output.splitlines()
                   if any(kw in l for kw in ("E   ", "FAILED", "Error", "assert"))]
    trace = "\n".join(trace_lines[:5]) or "AssertionError: test failed."
    tmpl  = _APR_RETRY_PROMPT if attempt > 1 else _APR_PROMPT
    prompt = tmpl.format(
        description=sample.get("description", "Fix the bug."),
        trace=trace, buggy_code=sample["buggy_code"],
    )
    messages = [
        {"role": "system", "content": _APR_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    temp   = temps[min(attempt - 1, len(temps) - 1)]
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=1024, do_sample=True,
            temperature=temp, top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_apr(model, tokenizer, sample: dict, test_code: str,
            max_attempts: int = 3) -> dict:
    """
    Run the APR loop on one sample using the provided test_code (PartB-generated).
    Returns a dict with apparent_repaired, attempts_used, repaired_code.
    """
    t0     = time.time()
    result = {"apparent_repaired": False, "attempts_used": 0,
              "repaired_code": sample["buggy_code"], "apr_output": ""}

    # Confirm the bug is detected by the generated tests
    detection = _run_sandbox(sample["buggy_code"], test_code)
    if not detection["any_failed"]:
        result["apr_output"] = "Bug not detected by generated tests — APR skipped."
        return result

    fail_output  = detection["output"]
    current_code = sample["buggy_code"]

    for attempt in range(1, max_attempts + 1):
        raw        = _generate_repair(model, tokenizer, sample, fail_output, attempt)
        patched    = _ast_patch(current_code, raw) or _extract_clean_code(raw)
        if not patched:
            continue
        check      = _run_sandbox(patched, test_code)
        result["attempts_used"] = attempt
        if check["all_passed"]:
            result.update(apparent_repaired=True,
                          repaired_code=patched,
                          apr_output=check["output"])
            break
        fail_output  = check["output"]
        current_code = patched

    result["time_seconds"] = time.time() - t0
    return result


# ── Per-sample evaluation ─────────────────────────────────────────────────────

@dataclass
class SampleResult:
    id:                  str  = ""
    condition:           str  = ""
    test_generated:      bool = False
    test_syntax_valid:   bool = False
    test_compile_ok:     bool = False
    test_has_assertions: bool = False
    test_detects_bug:    bool = False
    detection_passed:    int  = 0
    detection_failed:    int  = 0
    detection_total:     int  = 0
    apparent_repaired:   bool = False
    attempts_used:       int  = 0
    req_coverage_pct:    float = 0.0
    uncovered_reqs:      int  = 0
    gen_time_s:          float = 0.0
    apr_time_s:          float = 0.0
    error:               str  = ""

    def to_dict(self):
        return asdict(self)


def _check_test_syntax(test_code: str) -> tuple[bool, bool, bool]:
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return False, False, False
    has_fn = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name.startswith("test_")
        for n in ast.walk(tree)
    )
    has_assert = any(isinstance(n, ast.Assert) for n in ast.walk(tree))
    return True, has_fn, has_assert


def evaluate_sample(sample: dict, partb_model, partb_tok,
                    partc_model, partc_tok, condition: str,
                    artifact_dir: Path) -> SampleResult:
    res           = SampleResult(id=sample["id"], condition=condition)
    use_reqs      = condition in ("E4", "E5")
    use_lora_apr  = condition in ("E3", "E4", "E5")
    requirements  = description_to_requirements(sample["description"]) if use_reqs else []

    # ── Part B: generate tests ─────────────────────────────────────────
    try:
        test_code, gen_t = _generate_tests(partb_model, partb_tok, sample, requirements)
        res.gen_time_s   = gen_t
        res.test_generated = bool(test_code.strip())
    except Exception as exc:
        res.error = f"TestGen failed: {exc}"
        log.warning("[%s] testgen error: %s", sample["id"], exc)
        return res

    syntax_ok, has_fn, has_assert = _check_test_syntax(test_code)
    res.test_syntax_valid   = syntax_ok
    res.test_compile_ok     = syntax_ok and has_fn
    res.test_has_assertions = has_assert

    # ── Requirement coverage (E4/E5) ──────────────────────────────────
    if use_reqs and requirements:
        uncov = _check_req_coverage(requirements, test_code)
        total = len([r for r in requirements if r.get("label") == 1])
        covered = total - len(uncov)
        res.req_coverage_pct = round(covered / total * 100, 1) if total else 0.0
        res.uncovered_reqs   = len(uncov)

        # E5: gap-fill pass
        if condition == "E5" and uncov:
            try:
                test_code = _gapfill_tests(partb_model, partb_tok, sample, uncov, test_code)
                uncov2 = _check_req_coverage(requirements, test_code)
                covered2 = total - len(uncov2)
                res.req_coverage_pct = round(covered2 / total * 100, 1)
                res.uncovered_reqs   = len(uncov2)
            except Exception as exc:
                log.warning("[%s] gap-fill error: %s", sample["id"], exc)

    # ── Save generated test artifact ──────────────────────────────────
    (artifact_dir / f"{sample['id']}_test.py").write_text(test_code, encoding="utf-8")

    # ── Bug detection ─────────────────────────────────────────────────
    if not syntax_ok:
        return res
    det = _run_sandbox(sample["buggy_code"], test_code)
    res.detection_passed = det["passed"]
    res.detection_failed = det["failed"]
    res.detection_total  = det["total"]
    res.test_detects_bug = det["any_failed"]

    if not res.test_detects_bug:
        (artifact_dir / f"{sample['id']}_repaired.py").write_text(
            sample["buggy_code"], encoding="utf-8")
        return res

    # ── Part C: APR ──────────────────────────────────────────────────
    apr_model = partc_model if use_lora_apr else partb_model
    apr_tok   = partc_tok   if use_lora_apr else partb_tok
    try:
        apr = run_apr(apr_model, apr_tok, sample, test_code)
        res.apparent_repaired = apr["apparent_repaired"]
        res.attempts_used     = apr["attempts_used"]
        res.apr_time_s        = apr.get("time_seconds", 0.0)
        (artifact_dir / f"{sample['id']}_repaired.py").write_text(
            apr["repaired_code"], encoding="utf-8")
    except Exception as exc:
        res.error = f"APR failed: {exc}"
        log.warning("[%s] APR error: %s", sample["id"], exc)
        (artifact_dir / f"{sample['id']}_repaired.py").write_text(
            sample["buggy_code"], encoding="utf-8")

    return res


# ── Summary ───────────────────────────────────────────────────────────────────

def _summarize(results: list[SampleResult], condition: str) -> dict:
    n = len(results)
    if n == 0:
        return {}
    gen_ok    = sum(r.test_generated    for r in results)
    syntax_ok = sum(r.test_syntax_valid for r in results)
    compile_ok= sum(r.test_compile_ok   for r in results)
    detected  = sum(r.test_detects_bug  for r in results)
    apparent  = sum(r.apparent_repaired for r in results)
    req_covs  = [r.req_coverage_pct     for r in results if r.req_coverage_pct > 0]

    return {
        "condition":             condition,
        "n_samples":             n,
        "test_generation_rate":  round(gen_ok    / n * 100, 2),
        "syntax_valid_rate":     round(syntax_ok / n * 100, 2),
        "compile_ok_rate":       round(compile_ok/ n * 100, 2),
        "test_detection_rate":   round(detected  / n * 100, 2),
        "apparent_repair_rate":  round(apparent  / n * 100, 2),
        "avg_req_coverage_pct":  round(sum(req_covs) / len(req_covs), 2) if req_covs else None,
        "avg_gen_time_s":        round(sum(r.gen_time_s for r in results) / n, 2),
        "avg_apr_time_s":        round(sum(r.apr_time_s for r in results) / n, 2),
        "note": ("apparent_repair_rate requires local hidden-test eval "
                 "via pipeline_eval_local.py for TRUE repair rate"),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    condition = args.condition

    out_dir      = Path(args.output_dir) / condition
    artifact_dir = out_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(exist_ok=True)

    samples = load_dataset(args.dataset, args.limit)

    # Load PartB model (base Qwen, no LoRA — consistent with ablation scripts)
    partb_model, partb_tok = load_base_qwen()

    # Load PartC model (base + LoRA adapter for E3/E4/E5; reuse base for E2)
    partc_model, partc_tok = None, None
    if condition in ("E3", "E4", "E5"):
        if not args.partc_adapter:
            raise ValueError("--partc_adapter is required for conditions E3/E4/E5")
        partc_model = attach_lora(partb_model, args.partc_adapter)
        partc_tok   = partb_tok
        log.info("PartC LoRA attached — using shared base model with adapter swap")

    results: list[SampleResult] = []
    for i, sample in enumerate(samples):
        log.info("[%d/%d] %s", i + 1, len(samples), sample["id"])
        r = evaluate_sample(
            sample, partb_model, partb_tok,
            partc_model or partb_model, partc_tok or partb_tok,
            condition, artifact_dir,
        )
        results.append(r)
        if (i + 1) % 20 == 0:
            _checkpoint(results, out_dir, condition)

    _checkpoint(results, out_dir, condition)
    summary = _summarize(results, condition)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    log.info("=== %s COMPLETE ===", condition)
    for k, v in summary.items():
        log.info("  %-30s %s", k, v)
    log.info("Artifacts saved to %s", artifact_dir)


def _checkpoint(results: list[SampleResult], out_dir: Path, condition: str):
    data = [r.to_dict() for r in results]
    (out_dir / "results.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    log.info("Checkpoint saved (%d samples)", len(results))


if __name__ == "__main__":
    main()
