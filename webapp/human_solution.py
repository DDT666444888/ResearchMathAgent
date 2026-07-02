"""Human (reference) solutions for the First Proof batch-2 problems, and a
cached, numbers-backed comparison of the RMA best proof against them.

The batch-2 human solutions live at
    data/batch-2/batch-2-human-solution/problem-NN/human-solution.tex
and map to the first_proof_2 problem ids ``prob-NN`` (NN = 01..10).

``generate_human_comparison`` asks an LLM to compare the AI's best proof to the
human solution and returns a structured dict with quantitative axes (0–10
similarity/completeness/correctness, matched vs. total key steps, same-answer
flag) plus prose. The result is cached in
    documents/questions/<pid>/human_comparison.json
keyed by a hash of (human solution + AI proof), so it is only recomputed when
either text actually changes — mirroring proof_eval's determinism pin.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# 0–10 quantitative axes the comparison must return.
CMP_SCALES = {
    "approach_similarity": 10,     # how close the AI's route is to the human's
    "completeness_vs_human": 10,   # share of the human proof's essential content covered
    "correctness_vs_human": 10,    # correctness judged against the human reference
    "rigor_vs_human": 10,          # rigor relative to the human write-up
}


def _dataset_uses_human_solutions(dataset: str) -> bool:
    return dataset == "first_proof_2"


def human_solution_dir(repo_root: Path, pid: str, dataset: str) -> Path | None:
    """Path to the human-solution folder for a first_proof_2 problem, or None."""
    if not _dataset_uses_human_solutions(dataset):
        return None
    m = re.match(r"prob-?0*(\d+)$", pid.strip().lower())
    if not m:
        return None
    n = int(m.group(1))
    d = repo_root / "data" / "batch-2" / "batch-2-human-solution" / f"problem-{n:02d}"
    return d if d.is_dir() else None


def human_solution_tex(repo_root: Path, pid: str, dataset: str) -> str | None:
    """The human solution's LaTeX body (between \\begin/\\end{document} if present)."""
    d = human_solution_dir(repo_root, pid, dataset)
    if d is None:
        return None
    f = d / "human-solution.tex"
    if not f.is_file():
        return None
    raw = f.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", raw, re.DOTALL)
    return (m.group(1) if m else raw).strip() or None


def human_solution_pdf_rel(repo_root: Path, pid: str, dataset: str) -> str | None:
    d = human_solution_dir(repo_root, pid, dataset)
    if d and (d / "human-solution.pdf").is_file():
        return f"data/batch-2/batch-2-human-solution/{d.name}/human-solution.pdf"
    return None


# ── comparison cache ──────────────────────────────────────────────────────────

def _cmp_path(repo_root: Path, pid: str) -> Path:
    return repo_root / "documents" / "questions" / pid / "human_comparison.json"


def load_human_comparison(repo_root: Path, pid: str) -> dict | None:
    p = _cmp_path(repo_root, pid)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save(repo_root: Path, pid: str, result: dict) -> None:
    p = _cmp_path(repo_root, pid)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def _hash(human_tex: str, proof_tex: str) -> str:
    h = hashlib.sha256()
    h.update(human_tex.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(proof_tex.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


_SYSTEM = (
    "You are an expert mathematician comparing an AI-generated proof to the "
    "official human (reference) solution of the same research problem. Judge the "
    "AI proof AGAINST the human solution as ground truth. Return ONLY a JSON "
    "object (no prose, no code fences)."
)

_PROMPT = """Problem statement:
{problem}

=== HUMAN (reference) solution ===
{human}

=== AI best proof ===
{proof}

Compare the AI proof to the human solution and return ONLY this JSON object:
{{
  "same_final_answer": true or false,   // does the AI reach the human's conclusion?
  "approach_similarity": 0-10,          // how close the AI's method is to the human's
  "completeness_vs_human": 0-10,        // share of the human proof's essential content covered
  "correctness_vs_human": 0-10,         // correctness vs the human reference
  "rigor_vs_human": 0-10,               // rigor vs the human write-up
  "human_key_steps": <int>,             // number of essential lemmas/steps in the human solution
  "ai_matched_steps": <int>,            // how many of those the AI proof also establishes
  "ai_missing_steps": ["short phrase", ...],   // human steps the AI omits or leaves incomplete
  "ai_divergences": ["short phrase", ...],      // where the AI diverges from / adds beyond the human
  "approach_summary": "2-4 sentences relating the two approaches",
  "verdict": "2-4 sentence overall comparison, referencing the numbers"
}}
Every numeric field is REQUIRED. ai_matched_steps must be <= human_key_steps."""


def generate_human_comparison(repo_root: Path, pid: str, dataset: str,
                              force: bool = False) -> dict:
    """Compare the AI best proof to the human solution; cache + return the dict.

    Returns {"available": False, ...} when there is no human solution or no AI
    proof yet (so callers can silently skip the section)."""
    human = human_solution_tex(repo_root, pid, dataset)
    if not human:
        return {"available": False, "reason": "no human solution for this problem/dataset"}

    from .dataset_store import get_problem
    p = get_problem(dataset, pid) or {}
    problem_text = (p.get("tex") or p.get("statement") or "")[:6000]

    try:
        from .proofs import get_best_proof
        best = get_best_proof(pid, dataset)
        proof_tex = (best or {}).get("solution_tex") or ""
    except Exception:
        proof_tex = ""
    if not proof_tex.strip():
        return {"available": False, "reason": "no AI proof yet"}

    cmp_hash = _hash(human, proof_tex)
    if not force:
        cached = load_human_comparison(repo_root, pid)
        if (cached and cached.get("available")
                and cached.get("cmp_hash") == cmp_hash
                and all(cached.get(k) is not None for k in CMP_SCALES)):
            logger.info("Human comparison %s: reusing cached (unchanged)", pid)
            return cached

    from .llm import complete
    prompt = _PROMPT.format(problem=problem_text, human=human[:16000], proof=proof_tex[:16000])

    def _coerce_int(v, hi=None):
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)):
            n = int(v)
        elif isinstance(v, str) and re.search(r"-?\d+", v):
            n = int(re.search(r"-?\d+", v).group(0))
        else:
            return None
        if hi is not None:
            n = max(0, min(hi, n))
        return max(0, n) if hi is None else n

    parsed: dict = {}
    for _ in range(2):  # one retry if a required number is missing/garbled
        try:
            raw = (complete(prompt, system=_SYSTEM, max_tokens=2048) or "").strip()
        except Exception as e:  # noqa: BLE001
            return {"available": False, "reason": f"LLM call failed: {e}"}
        txt = raw
        if txt.startswith("```"):
            ls = txt.splitlines()
            txt = "\n".join(ls[1:-1] if ls and ls[-1].strip() == "```" else ls[1:])
        try:
            parsed = json.loads(txt)
        except Exception:
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            parsed = json.loads(m.group(0)) if m else {}
        if isinstance(parsed, dict) and all(
                _coerce_int(parsed.get(k), CMP_SCALES[k]) is not None for k in CMP_SCALES):
            break

    if not isinstance(parsed, dict) or not parsed:
        return {"available": False, "reason": "comparison did not return usable JSON"}

    result: dict = {"available": True, "cmp_hash": cmp_hash, "dataset": dataset}
    for k, hi in CMP_SCALES.items():
        result[k] = _coerce_int(parsed.get(k), hi)
    result["human_key_steps"] = _coerce_int(parsed.get("human_key_steps")) or 0
    result["ai_matched_steps"] = min(
        _coerce_int(parsed.get("ai_matched_steps")) or 0,
        result["human_key_steps"] or 10_000,
    )
    result["same_final_answer"] = bool(parsed.get("same_final_answer"))
    result["ai_missing_steps"] = [str(x) for x in (parsed.get("ai_missing_steps") or [])][:12]
    result["ai_divergences"] = [str(x) for x in (parsed.get("ai_divergences") or [])][:12]
    result["approach_summary"] = str(parsed.get("approach_summary") or "").strip()
    result["verdict"] = str(parsed.get("verdict") or "").strip()

    _save(repo_root, pid, result)
    logger.info("Human comparison %s: sim=%s complete=%s correct=%s steps=%s/%s",
                pid, result.get("approach_similarity"), result.get("completeness_vs_human"),
                result.get("correctness_vs_human"), result.get("ai_matched_steps"),
                result.get("human_key_steps"))
    return result
