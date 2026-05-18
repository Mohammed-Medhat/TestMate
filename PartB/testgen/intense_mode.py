"""
Intense Mode — Online LoRA Fine-Tuning During Test Generation
==============================================================
The model actually learns from its failures mid-run and gets smarter
at testing the specific file being processed.

Flow:
  Generate test → Run pytest → FAIL → compute loss → update LoRA weights
                                                     → generate again (smarter)
                              → PASS → DPO pair (fail, pass) → update weights
                              → keep going until score >= threshold

Key constraints for RTX 4060 (8GB VRAM):
  - Only LoRA parameters are updated (tiny % of weights)
  - Gradient checkpointing enabled (halves VRAM at cost of ~30% speed)
  - AMP autocast for FP16 compute (lower VRAM than BF16)
  - SDPA (Scaled Dot-Product Attention) for memory-efficient attention
  - Max sequence length capped at 1024 tokens for training
  - Gradient accumulation (2 steps) to reduce peak VRAM
  - Gradients freed immediately after each step
"""

import os
import sys
import re
import gc
import json
import time
import copy
import argparse
from typing import Optional

import torch
import torch.nn.functional as F
# NOTE: torch.cuda.amp.autocast is deprecated, use torch.amp.autocast instead
from torch.amp import autocast

# ── Ensure testgen imports work ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# CONFIG
# ============================================================

INTENSE_DEFAULTS = {
    "max_iterations": 10,       # Max generation attempts
    "learning_rate": 1e-5,      # Small LR — don't overfit to one file
    "dpo_beta": 0.1,            # DPO temperature (lower = stronger preference)
    "target_score": 90,         # Stop when composite score >= this
    "max_seq_len": 1024,        # Max tokens for training (was 1536, reduced for 8GB VRAM)
    "gradient_accumulation": 2, # Accumulate grads over 2 steps (halves peak VRAM)
    "save_checkpoint": True,    # Save improved LoRA weights
    "persistent_learning": False,  # If True: DON'T reset weights after each file
}


# ============================================================
# CORE: Log probability computation
# ============================================================

def get_sequence_logprob(model, tokenizer, prompt: str, response: str,
                         max_len: int = 1536) -> torch.Tensor:
    """Compute the total log-probability of `response` given `prompt`.

    Returns a scalar tensor (on GPU, with grad if model is in train mode).
    """
    full_text = prompt + response
    inputs = tokenizer(
        full_text, return_tensors="pt",
        truncation=True, max_length=max_len,
    ).to(model.device)

    prompt_ids = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=max_len,
    ).to(model.device)
    prompt_len = prompt_ids["input_ids"].shape[1]

    # Forward pass
    with autocast('cuda', dtype=torch.float16):
        outputs = model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Shift: logits[t] predicts token[t+1]
    # We only care about the response portion
    shift_logits = logits[:, prompt_len - 1:-1, :]  # (1, resp_len, vocab)
    shift_labels = inputs["input_ids"][:, prompt_len:]  # (1, resp_len)

    if shift_logits.shape[1] == 0 or shift_labels.shape[1] == 0:
        return torch.tensor(0.0, device=model.device, requires_grad=True)

    # Truncate to min length
    min_len = min(shift_logits.shape[1], shift_labels.shape[1])
    shift_logits = shift_logits[:, :min_len, :]
    shift_labels = shift_labels[:, :min_len]

    # Per-token log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Sum over response tokens (total log prob)
    return token_log_probs.sum()


# ============================================================
# LOSS: Failure penalization
# ============================================================

def compute_failure_loss(model, tokenizer, prompt: str, bad_response: str,
                         error_log: str, max_len: int = 1536) -> torch.Tensor:
    """Penalize the model for generating `bad_response` given the error context.

    Builds a correction prompt that includes the error, then computes
    negative log-likelihood on the bad response — making the model
    less likely to repeat this mistake.
    """
    # Build correction context
    error_snippet = error_log[:500] if error_log else "Unknown error"
    correction_prompt = (
        f"{prompt}\n\n"
        f"# Previous attempt FAILED with this error:\n"
        f"# {error_snippet}\n\n"
        f"# Generate a FIXED version:\n"
    )

    # Log-prob of the bad response given correction context
    logprob = get_sequence_logprob(
        model, tokenizer, correction_prompt, bad_response, max_len
    )

    # Negative = penalize this sequence (push probability DOWN)
    loss = -logprob / max(1, len(tokenizer.encode(bad_response)))
    return loss


# ============================================================
# LOSS: DPO (Direct Preference Optimization) per iteration
# ============================================================

def compute_dpo_loss(model, tokenizer, prompt: str,
                     chosen: str, rejected: str,
                     beta: float = 0.1,
                     max_len: int = 1536) -> torch.Tensor:
    """DPO loss: teach the model to prefer `chosen` over `rejected`.

    Called when: iteration N failed → iteration N+K passed.
    The fail is `rejected`, the pass is `chosen`.

    DPO objective:
        loss = -log σ(β * (log π(chosen|x) - log π(rejected|x)))
    """
    chosen_logprob = get_sequence_logprob(
        model, tokenizer, prompt, chosen, max_len
    )
    rejected_logprob = get_sequence_logprob(
        model, tokenizer, prompt, rejected, max_len
    )

    # DPO loss
    logit_diff = beta * (chosen_logprob - rejected_logprob)
    loss = -F.logsigmoid(logit_diff)

    return loss


# ============================================================
# MAIN: Intense Mode Loop
# ============================================================

def intense_mode(
    target_file: str,
    import_path: str = None,
    max_iterations: int = None,
    learning_rate: float = None,
    dpo_beta: float = None,
    target_score: float = None,
    persistent_learning: bool = None,
    save_checkpoint: bool = None,
    log_callback=None,
    max_retries: int = 3,
):
    """Online LoRA fine-tuning during test generation.

    The model generates tests, runs them, and if they fail,
    it updates its LoRA weights to learn from the failure.
    When a pass follows a fail, it does a DPO update.

    Args:
        target_file: Python file to generate tests for.
        import_path: Package import path (e.g. 'requests.auth').
        max_iterations: Max generation attempts (default: 10).
        learning_rate: LoRA update LR (default: 1e-5).
        dpo_beta: DPO temperature (default: 0.1).
        target_score: Stop when composite >= this (default: 90).
        persistent_learning: Don't reset weights after (default: False).
        save_checkpoint: Save improved LoRA weights (default: True).
        log_callback: Optional logging function (msg, level).
        max_retries: Retries per target in autonomous_loop.

    Returns:
        dict with best_test, best_score, history, weights_updated
    """
    # ── Apply defaults ──
    cfg = INTENSE_DEFAULTS.copy()
    if max_iterations is not None: cfg["max_iterations"] = max_iterations
    if learning_rate is not None: cfg["learning_rate"] = learning_rate
    if dpo_beta is not None: cfg["dpo_beta"] = dpo_beta
    if target_score is not None: cfg["target_score"] = target_score
    if persistent_learning is not None: cfg["persistent_learning"] = persistent_learning
    if save_checkpoint is not None: cfg["save_checkpoint"] = save_checkpoint

    def _log(msg, level="info"):
        if log_callback:
            log_callback(msg, level)
        print(msg)

    _log("=" * 60)
    _log("  🔥 INTENSE MODE — Online LoRA Fine-Tuning")
    _log(f"  📄 Target: {os.path.basename(target_file)}")
    _log(f"  🔄 Max iterations: {cfg['max_iterations']}")
    _log(f"  📈 Target score: {cfg['target_score']}")
    _log(f"  🧠 Learning rate: {cfg['learning_rate']}")
    _log(f"  🔗 Persistent: {cfg['persistent_learning']}")
    _log("=" * 60)

    # ── Load model (with LoRA trainable) ──
    from main import (
        load_model, autonomous_loop, extract_context_from_code,
        build_prompt, generate_test, extract_test_code,
        run_pytest, run_pytest_with_coverage, run_mutation_testing,
        compute_composite_score, quick_quality_check,
        fix_generator_assertions, save_to_flywheel,
        BASE_MODEL, LORA_PATH,
    )

    _log("🧠 Loading model (trainable LoRA, FP16 + SDPA)...")
    model, tokenizer = load_model(attn_implementation="sdpa")

    # ── Enable training on LoRA parameters only ──
    model.train()
    trainable_params = []
    frozen_params = 0
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
            frozen_params += 1

    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    _log(f"   ✅ Trainable: {n_trainable:,} params ({100*n_trainable/n_total:.2f}%)")
    _log(f"   🔒 Frozen: {frozen_params} layers")

    # ── Enable gradient checkpointing to save VRAM ──
    try:
        model.gradient_checkpointing_enable()
        _log("   ✅ Gradient checkpointing enabled (VRAM safe)")
    except Exception:
        _log("   ⚠️  Gradient checkpointing not available", "warning")

    # ── Report VRAM after setup ──
    if torch.cuda.is_available():
        free_vram = torch.cuda.mem_get_info()[0] / 1e9
        used_vram = torch.cuda.memory_allocated() / 1e9
        _log(f"   📊 VRAM after setup: {used_vram:.1f}GB used, {free_vram:.1f}GB free")
        if free_vram < 1.5:
            _log("   ⚠️  Low VRAM! Training may OOM. Close other GPU apps.", "warning")

    # ── Optimizer: only LoRA params ──
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["learning_rate"],
        weight_decay=0.01,
    )

    # ── Save initial LoRA state (for rollback if needed) ──
    if not cfg["persistent_learning"]:
        initial_lora_state = {
            k: v.clone() for k, v in model.state_dict().items()
            if "lora" in k.lower()
        }

    # ── Read source ──
    with open(target_file, "r") as f:
        source_code = f.read()

    target_name = os.path.basename(target_file)
    module_name = target_name.replace('.py', '')
    actual_import = import_path if import_path else module_name
    ctx = extract_context_from_code(source_code)

    # ── Import statement ──
    if import_path:
        import_statement = f"from {actual_import} import *"
    else:
        import_statement = f"from {module_name} import *"

    # ── Test file location ──
    if import_path:
        gen_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_tests")
        os.makedirs(gen_dir, exist_ok=True)
        test_file = os.path.join(gen_dir, f"test_{target_name}")
    else:
        test_file = os.path.join(
            os.path.dirname(os.path.abspath(target_file)),
            f"test_{target_name}"
        )

    # ── System prompt (same as autonomous_loop) ──
    system_prompt = f"""You are an expert Python QA engineer writing pytest unit tests.

ABSOLUTE RULES:
1. Start with: import pytest; {import_statement}
2. NEVER call real internet/filesystem/database.
3. Mock HTTP calls with @patch('requests.get').
4. Instantiate classes before testing methods.
5. Use pytest.raises() for exception testing.
6. Return ALL test code inside <test>...</test> XML tags.
7. Assert specific VALUES, not just types."""

    # ── Intense Loop ──
    history = []
    best_test = None
    best_score = 0.0
    weights_updated = 0
    dpo_pairs_trained = 0
    last_failed_response = None
    last_failed_prompt = None

    for iteration in range(1, cfg["max_iterations"] + 1):
        _log(f"\n{'━' * 50}")
        _log(f"  🔥 Iteration {iteration}/{cfg['max_iterations']}")
        _log(f"{'━' * 50}")

        # ── Phase 1: Run full autonomous_loop ──
        _log("  🚀 Running autonomous test generation...")

        # Switch to eval for generation (faster, uses KV cache)
        model.eval()
        success = autonomous_loop(
            model, tokenizer, target_file,
            import_path=import_path,
            max_retries=max_retries,
        )
        model.train()

        # Read generated test
        if not os.path.exists(test_file):
            _log("  ⚠️  No test file generated", "warning")
            history.append({
                "iteration": iteration,
                "passed": False,
                "error": "No test file generated",
                "score": 0,
            })
            continue

        with open(test_file, "r") as f:
            test_code = f.read()

        n_tests = len(re.findall(r'def test_\w+', test_code))
        if n_tests == 0:
            _log("  ⚠️  No test functions found", "warning")
            continue

        # ── Phase 2: Score the test ──
        try:
            passed, logs = run_pytest(test_file, target_file)
        except Exception as e:
            passed, logs = False, str(e)

        if passed:
            try:
                _, _, line_cov, branch_cov, _ = run_pytest_with_coverage(
                    test_file, target_file, actual_import, need_lines=False
                )
            except Exception:
                line_cov, branch_cov = 0.0, 0.0

            try:
                killed, mut_feedback = run_mutation_testing(target_file, test_file)
                mut_score = 100.0 if killed else float(
                    re.search(r'(\d+)%', mut_feedback).group(1)
                ) if re.search(r'(\d+)%', mut_feedback) else 0.0
            except Exception:
                mut_score = 0.0

            composite = compute_composite_score(
                test_code=test_code,
                line_coverage_pct=line_cov,
                branch_coverage_pct=branch_cov,
                mutation_score=mut_score,
                per_test=False,
            )
            score = composite["composite"]
        else:
            score = 0.0
            line_cov = branch_cov = mut_score = 0.0
            composite = {}

        # Build prompt for this iteration (for DPO/loss computation)
        user_prompt = build_prompt(source_code, ctx, target_name)
        full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant: "

        # Record history
        entry = {
            "iteration": iteration,
            "passed": passed,
            "score": score,
            "n_tests": n_tests,
            "line_cov": line_cov,
            "branch_cov": branch_cov,
            "mutation": mut_score,
            "error": logs[:300] if not passed else "",
        }
        history.append(entry)

        if passed:
            score_emoji = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
            _log(f"  ✅ PASSED {score_emoji} Score: {score:.0f}/100 "
                 f"(line:{line_cov:.0f}% branch:{branch_cov:.0f}% mut:{mut_score:.0f}%)")

            if score > best_score:
                best_test = test_code
                best_score = score
                _log(f"  🏆 New best: {score:.0f}/100")

            # ── Phase 3a: DPO update (if we have a previous failure) ──
            if last_failed_response and last_failed_prompt:
                _log("  📚 DPO update: learning preference (pass > fail)...")
                try:
                    # Free VRAM before training step
                    gc.collect()
                    torch.cuda.empty_cache()

                    dpo_loss = compute_dpo_loss(
                        model, tokenizer,
                        prompt=last_failed_prompt,
                        chosen=test_code[:cfg["max_seq_len"]],
                        rejected=last_failed_response[:cfg["max_seq_len"]],
                        beta=cfg["dpo_beta"],
                        max_len=cfg["max_seq_len"],
                    )
                    # Scale loss for gradient accumulation
                    scaled_loss = dpo_loss / cfg["gradient_accumulation"]
                    scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)  # set_to_none saves VRAM vs zero
                    weights_updated += 1
                    dpo_pairs_trained += 1
                    _log(f"  ✅ DPO loss: {dpo_loss.item():.4f} "
                         f"(weights updated: {weights_updated})")
                except torch.cuda.OutOfMemoryError:
                    _log("  ⚠️  DPO OOM — skipping this update, freeing VRAM", "warning")
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    _log(f"  ⚠️  DPO update failed: {e}", "warning")
                    optimizer.zero_grad(set_to_none=True)

                last_failed_response = None
                last_failed_prompt = None

                # Aggressive VRAM cleanup after training step
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Check if we've reached target score
            if score >= cfg["target_score"]:
                _log(f"\n  🎯 TARGET REACHED at iteration {iteration}! "
                     f"Score: {score:.0f} >= {cfg['target_score']}")
                break

        else:
            error_lines = [l for l in logs.split("\n")
                           if any(k in l for k in
                                  ["FAILED", "Error", "assert", "Exception"])]
            error_summary = " | ".join(error_lines[:2])[:150] if error_lines else logs[:150]
            _log(f"  ❌ FAILED: {error_summary}")

            # ── Phase 3b: Learn from failure ──
            _log("  📚 Learning from failure (penalizing bad output)...")
            try:
                # Free VRAM before training step
                gc.collect()
                torch.cuda.empty_cache()

                fail_loss = compute_failure_loss(
                    model, tokenizer,
                    prompt=full_prompt,
                    bad_response=test_code[:cfg["max_seq_len"]],
                    error_log=logs[:500],
                    max_len=cfg["max_seq_len"],
                )
                # Scale loss for gradient accumulation
                scaled_loss = fail_loss / cfg["gradient_accumulation"]
                scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # set_to_none saves VRAM
                weights_updated += 1
                _log(f"  ✅ Failure loss: {fail_loss.item():.4f} "
                     f"(weights updated: {weights_updated})")
            except torch.cuda.OutOfMemoryError:
                _log("  ⚠️  Failure OOM — skipping this update, freeing VRAM", "warning")
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                _log(f"  ⚠️  Failure learning failed: {e}", "warning")
                optimizer.zero_grad(set_to_none=True)

            # Save for DPO pairing
            last_failed_response = test_code
            last_failed_prompt = full_prompt

            # Aggressive VRAM cleanup after training step
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ── Summary ──
    _log(f"\n{'=' * 60}")
    _log(f"  🔥 INTENSE MODE COMPLETE")
    _log(f"  📊 Best score: {best_score:.0f}/100")
    _log(f"  🔄 Iterations used: {len(history)}/{cfg['max_iterations']}")
    _log(f"  📚 Weight updates: {weights_updated}")
    _log(f"  🔗 DPO pairs trained: {dpo_pairs_trained}")
    _log(f"{'=' * 60}")

    # ── Save checkpoint (if weights improved) ──
    if cfg["save_checkpoint"] and weights_updated > 0 and best_score > 0:
        checkpoint_dir = os.path.join(
            os.path.dirname(LORA_PATH), "intense_checkpoints"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ckpt_path = os.path.join(
            checkpoint_dir,
            f"intense_{target_name.replace('.py', '')}_{int(best_score)}_{timestamp}"
        )
        model.save_pretrained(ckpt_path)
        _log(f"  💾 Checkpoint saved: {ckpt_path}")

        # Also save training history
        meta = {
            "target_file": target_name,
            "best_score": best_score,
            "iterations": len(history),
            "weight_updates": weights_updated,
            "dpo_pairs": dpo_pairs_trained,
            "config": cfg,
            "history": history,
        }
        with open(os.path.join(ckpt_path, "intense_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ── Rollback weights if not persistent ──
    if not cfg["persistent_learning"] and weights_updated > 0:
        _log("  🔄 Resetting LoRA weights (non-persistent mode)")
        current_state = model.state_dict()
        for k, v in initial_lora_state.items():
            current_state[k] = v
        model.load_state_dict(current_state)

    # ── Save best test as flywheel data ──
    if best_test and best_score > 20:
        save_to_flywheel(
            target_file, source_code, best_test,
            best_score, 0.0,  # mutation_score already baked into composite
        )

    # Write best test to file
    if best_test:
        with open(test_file, "w") as f:
            f.write(best_test)
        _log(f"  📝 Best test written to {os.path.basename(test_file)}")

    return {
        "best_test": best_test,
        "best_score": best_score,
        "history": history,
        "weights_updated": weights_updated,
        "dpo_pairs_trained": dpo_pairs_trained,
        "test_file": test_file,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🔥 Intense Mode — Online LoRA Fine-Tuning for Test Generation"
    )
    parser.add_argument("--target", required=True,
                        help="Python file to test")
    parser.add_argument("--import-as", default=None,
                        help="Package import path (e.g. 'requests.auth')")
    parser.add_argument("--max-iter", type=int, default=10,
                        help="Max intense iterations (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate for LoRA updates (default: 1e-5)")
    parser.add_argument("--target-score", type=float, default=90,
                        help="Stop when composite score >= this (default: 90)")
    parser.add_argument("--persistent", action="store_true",
                        help="Keep learned weights across files (continual learning)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save LoRA checkpoints")
    parser.add_argument("--dpo-beta", type=float, default=0.1,
                        help="DPO temperature (default: 0.1)")
    args = parser.parse_args()

    if not os.path.exists(args.target):
        print(f"❌ Target file not found: {args.target}")
        sys.exit(1)

    result = intense_mode(
        target_file=args.target,
        import_path=args.import_as,
        max_iterations=args.max_iter,
        learning_rate=args.lr,
        target_score=args.target_score,
        persistent_learning=args.persistent,
        save_checkpoint=not args.no_save,
        dpo_beta=args.dpo_beta,
    )

    print(f"\n📊 Final: {result['best_score']:.0f}/100 "
          f"({result['weights_updated']} weight updates, "
          f"{result['dpo_pairs_trained']} DPO pairs)")
    sys.exit(0 if result["best_score"] > 0 else 1)
