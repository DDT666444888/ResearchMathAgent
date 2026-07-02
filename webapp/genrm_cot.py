"""GenRM-CoT proof verifier — a separate, calibrated correctness score.

This complements the rubric judge in `proof_eval.py`. Where the rubric is a
single-shot LLM-as-judge returning fine-grained dimension scores, GenRM-CoT
(generative reward model with chain-of-thought, Zhang et al. 2024,
"Generative Verifiers: Reward Modeling as Next-Token Prediction") asks the model
to *verify* the proof: reason step by step looking for errors and gaps, then emit
a binary correct/incorrect verdict.

The original GenRM-CoT score is P("Yes" | verification) read from token logprobs.
We run on the Claude subscription CLI, which does not expose logprobs, so we use
the paper's logprob-free inference-time-scaling variant: sample K independent
chain-of-thought verifications and take the **majority vote** (Maj@K). The score
is the fraction of samples that judge the proof correct, in [0, 100]; the spread
across samples is a built-in confidence/agreement signal that a single-shot judge
cannot provide.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_MODEL = "claude-fable-5"
DEFAULT_SAMPLES = 5

_VERIFIER_SYSTEM = (
    "You are an exacting referee for a research-level mathematics journal. You are "
    "given a problem and a candidate proof. Your job is to VERIFY the proof: work "
    "through it step by step and decide whether it is a fully correct and complete "
    "proof of the stated claim. Err toward rejection — approving a flawed proof is "
    "worse than rejecting a sound one. A proof is INCORRECT if any step is invalid, "
    "any case is unhandled, any lemma is unproven, or the final claim does not follow."
)

_VERIFIER_PROMPT = """--- PROBLEM (LaTeX) ---
{problem}
--- CANDIDATE PROOF (LaTeX) ---
{proof}
--- END ---

Verify this proof. Think step by step:
1. Restate what must be proved.
2. Walk through the proof's argument, checking each step for valid inference,
   correct use of definitions/theorems, and unjustified leaps or missing cases.
3. Decide whether the proof is fully correct AND complete.

Write your reasoning, then end your response with EXACTLY these two lines and nothing after:
VERDICT: CORRECT
CONFIDENCE: <integer 0-100>

(use "VERDICT: INCORRECT" if the proof is not fully correct and complete; CONFIDENCE
is how sure you are of your verdict)."""

_VERDICT_RE = re.compile(r"VERDICT:\s*(CORRECT|INCORRECT)", re.IGNORECASE)
_CONF_RE = re.compile(r"CONFIDENCE:\s*(\d{1,3})", re.IGNORECASE)


def _parse_sample(text: str) -> dict | None:
    """Pull the verdict + confidence out of one verification sample."""
    if not text:
        return None
    vm = _VERDICT_RE.search(text)
    if not vm:
        return None
    correct = vm.group(1).upper() == "CORRECT"
    cm = _CONF_RE.search(text)
    confidence = max(0, min(100, int(cm.group(1)))) if cm else None
    # A short tail of the reasoning, for display.
    summary = text[: vm.start()].strip().splitlines()
    summary = " ".join(summary[-3:])[-400:] if summary else ""
    return {"correct": correct, "confidence": confidence, "summary": summary}


def genrm_cot_score(
    problem: str,
    proof: str,
    *,
    n_samples: int = DEFAULT_SAMPLES,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Run K chain-of-thought verifications and aggregate by majority vote.

    Returns {score, n_samples, n_valid, n_correct, mean_confidence, agreement,
    verdict, samples:[{correct,confidence,summary}], method}. ``score`` is the
    percentage of valid samples voting CORRECT (the Maj@K GenRM-CoT score).
    """
    from .llm import complete

    n_samples = max(1, min(int(n_samples), 11))
    prompt = _VERIFIER_PROMPT.format(problem=(problem or "")[:8000],
                                     proof=(proof or "")[:16000])
    samples: list[dict] = []
    for _ in range(n_samples):
        raw = complete(prompt, system=_VERIFIER_SYSTEM, model=model, max_tokens=4096)
        parsed = _parse_sample(raw or "")
        if parsed is not None:
            samples.append(parsed)

    n_valid = len(samples)
    if n_valid == 0:
        return {"error": "no parseable verification samples", "n_samples": n_samples}

    n_correct = sum(1 for s in samples if s["correct"])
    score = round(100.0 * n_correct / n_valid, 1)
    confs = [s["confidence"] for s in samples if s["confidence"] is not None]
    mean_conf = round(sum(confs) / len(confs), 1) if confs else None
    # Agreement = fraction of samples on the majority side (1.0 = unanimous).
    agreement = round(max(n_correct, n_valid - n_correct) / n_valid, 2)

    return {
        "method": f"GenRM-CoT (Maj@{n_valid} chain-of-thought verifications)",
        "score": score,                       # 0-100, separate from the rubric
        "verdict": "CORRECT" if n_correct * 2 > n_valid else "INCORRECT",
        "n_samples": n_samples,
        "n_valid": n_valid,
        "n_correct": n_correct,
        "mean_confidence": mean_conf,
        "agreement": agreement,
        "samples": samples,
    }


def _path(repo_root: Path, problem_id: str) -> Path:
    return repo_root / "documents" / "questions" / problem_id / "genrm_cot.json"


def load_genrm_cot(repo_root: Path, problem_id: str) -> dict | None:
    p = _path(repo_root, problem_id)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def evaluate_genrm_cot(
    repo_root: Path,
    problem_id: str,
    dataset: str = "first_proof_1",
    *,
    n_samples: int = DEFAULT_SAMPLES,
    force: bool = False,
) -> dict:
    """GenRM-CoT score for a problem's best proof. Cached alongside the rubric."""
    if not force:
        cached = load_genrm_cot(repo_root, problem_id)
        if cached:
            return cached

    prob_path = repo_root / "problems" / f"{problem_id}.tex"
    if prob_path.is_file():
        problem_text = prob_path.read_text(encoding="utf-8", errors="replace")
    else:
        try:
            from .dataset_store import get_problem
            p = get_problem(dataset, problem_id) or {}
            problem_text = p.get("tex") or p.get("statement") or ""
        except Exception:
            problem_text = ""
    if not problem_text:
        return {"error": f"Problem statement not found: {dataset}/{problem_id}"}

    try:
        from .proofs import get_best_proof
        best = get_best_proof(problem_id, dataset)
        if not best or not best.get("solution_tex"):
            return {"error": "No best proof available — run the agent and consolidate first"}
        proof_tex = best["solution_tex"]
    except Exception as e:
        return {"error": f"Could not load best proof: {e}"}

    result = genrm_cot_score(problem_text, proof_tex, n_samples=n_samples)
    if "error" not in result:
        p = _path(repo_root, problem_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
