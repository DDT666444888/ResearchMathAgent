"""LLM-based proof evaluation using the First Proof benchmark rubric (Appendix E)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_EVAL_SYSTEM = """You are an expert mathematician evaluating a research-level mathematical proof.
Evaluate the proof along the four fine-grained dimensions from the First Proof benchmark (Appendix E):

1. **Final Answer Accuracy** (0 or 1): Is the final answer/bound/construction claimed in the proof correct, independent of the derivation process?
   - 1 = the claimed conclusion is mathematically correct
   - 0 = the claimed conclusion is wrong or cannot be determined

2. **Logical Correctness** (0–10): Does each step follow valid logical inferences with correct application of definitions, theorems, and assumptions, with no invalid deductions?
   - 0–1 = severely flawed logic throughout
   - 2–3 = major logical errors that invalidate the proof
   - 4–5 = significant errors but partial correctness
   - 6–7 = mostly correct with minor gaps or errors
   - 8–9 = nearly flawless, only trivial issues
   - 10 = every step is logically sound and rigorously justified

3. **Proof Completeness** (0–10): Are all essential steps explicitly provided, with no missing arguments or unjustified leaps?
   - 0–1 = proof is largely incomplete or a sketch
   - 2–3 = major steps missing
   - 4–5 = several important cases or lemmas left unjustified
   - 6–7 = mostly complete but some steps hand-waved
   - 8–9 = nearly complete, only minor details omitted
   - 10 = fully explicit, every step justified

4. **Proof Clarity** (0–10): Is the proof coherent, well-structured, and easy for a mathematician in the field to follow?
   - 0–1 = incomprehensible or disorganized
   - 2–3 = very hard to follow
   - 4–5 = somewhat unclear
   - 6–7 = readable but room for improvement
   - 8–9 = clear and well-organized
   - 10 = exceptionally clear, well-structured, and readable

Return ONLY a JSON object with exactly these keys (no prose, no markdown fencing):
{
  "answer_accuracy": <0 or 1>,
  "logical_correctness": <integer 0..10>,
  "proof_completeness": <integer 0..10>,
  "proof_clarity": <integer 0..10>,
  "verdict": "<one sentence overall assessment>",
  "notes": "<2-4 sentences on main strengths and weaknesses>"
}"""

_EVAL_PROMPT = """Problem statement (LaTeX):
<problem>
{problem}
</problem>

Proof to evaluate (LaTeX source):
<proof>
{proof}
</proof>

Carefully evaluate this proof on all four dimensions and return the JSON scores."""


def _eval_path(repo_root: Path, problem_id: str) -> Path:
    return repo_root / "documents" / "questions" / problem_id / "proof_eval.json"


def load_proof_eval(repo_root: Path, problem_id: str) -> dict | None:
    p = _eval_path(repo_root, problem_id)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_proof_eval(repo_root: Path, problem_id: str, result: dict) -> None:
    p = _eval_path(repo_root, problem_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate_proof(repo_root: Path, problem_id: str, dataset: str = "first_proof_1",
                   force: bool = False) -> dict:
    """Run LLM evaluation of the best proof. Returns the result dict (and caches it)."""
    if not force:
        cached = load_proof_eval(repo_root, problem_id)
        if cached:
            return cached

    # Load the problem statement (legacy first_proof_1 .tex, else the dataset store).
    prob_path = repo_root / "problems" / f"{problem_id}.tex"
    if prob_path.is_file():
        problem_text = prob_path.read_text(encoding="utf-8", errors="replace")[:6000]
    else:
        try:
            from .dataset_store import get_problem
            p = get_problem(dataset, problem_id) or {}
            problem_text = (p.get("tex") or p.get("statement") or "")[:6000]
        except Exception:
            problem_text = ""
    if not problem_text:
        return {"error": f"Problem statement not found: {dataset}/{problem_id}"}

    # Load the best proof
    try:
        from .proofs import get_best_proof
        best = get_best_proof(problem_id, dataset)
        if not best or not best.get("solution_tex"):
            return {"error": "No best proof available — run the agent and consolidate first"}
        proof_tex = best["solution_tex"][:15000]
    except Exception as e:
        return {"error": f"Could not load best proof: {e}"}

    prompt = _EVAL_PROMPT.format(problem=problem_text, proof=proof_tex)

    from .llm import complete

    _SCALES = {"answer_accuracy": 1, "logical_correctness": 10,
               "proof_completeness": 10, "proof_clarity": 10}

    def _coerce(value, hi):
        """Coerce any model value (int/float/str/bool/None) to an int in [0,hi],
        or None if there is no number. Never raises."""
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)):
            return max(0, min(hi, int(value)))
        if isinstance(value, str):
            m = re.search(r"-?\d+", value)
            if m:
                return max(0, min(hi, int(m.group(0))))
        return None

    def _attempt():
        """One LLM call → (parsed_dict, {key: int|None}). Recovers each score from
        the parsed JSON, falling back to a direct regex over the raw text so a
        slightly malformed object never drops a score."""
        try:
            raw = complete(prompt, system=_EVAL_SYSTEM, max_tokens=2048) or ""
        except Exception as e:  # noqa: BLE001
            return {}, {}, f"LLM call failed: {e}"
        txt = raw.strip()
        if txt.startswith("```"):
            ls = txt.splitlines()
            txt = "\n".join(ls[1:-1] if ls and ls[-1].strip() == "```" else ls[1:])
        parsed: dict = {}
        try:
            parsed = json.loads(txt)
        except Exception:  # noqa: BLE001
            m = re.search(r"\{.*\}", txt, re.DOTALL)  # widest brace span
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except Exception:  # noqa: BLE001
                    parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        scores: dict = {}
        for key, hi in _SCALES.items():
            v = _coerce(parsed.get(key), hi)
            if v is None:  # fallback: scan the raw text for `"key": <number>`
                rm = re.search(rf'"{re.escape(key)}"\s*:\s*"?\s*(-?\d+)', txt)
                if rm:
                    v = max(0, min(hi, int(rm.group(1))))
            scores[key] = v
        return parsed, scores, (raw[:500] if not raw else None)

    parsed, scores, err = _attempt()
    # Retry once if the call failed or any score is missing — handles the
    # intermittent case where the model omits/garbles a field.
    if err is not None or any(v is None for v in scores.values()):
        p2, s2, err2 = _attempt()
        better = sum(v is not None for v in s2.values())
        if better > sum(v is not None for v in scores.values()):
            parsed, scores, err = p2, s2, err2

    # Require all four scores — return a clean error rather than show partial
    # empties in the UI/report (the bug being fixed).
    missing = [k for k, v in scores.items() if v is None]
    if missing:
        return {"error": f"Evaluation incomplete (missing {', '.join(missing)}); please re-run."}

    result = dict(parsed) if isinstance(parsed, dict) else {}
    result.update(scores)
    result["scale"] = 10  # dimension scale (0–10), so consumers can render correctly

    _save_proof_eval(repo_root, problem_id, result)
    logger.info("Proof eval for %s: %s", problem_id, {k: result.get(k) for k in _SCALES})
    return result
