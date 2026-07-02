"""`rma push` — run the push-forward, update every tab's content, and build ONE
huge combined PDF (all problems, all tabs) for review.

Usage:
  rma push                       # first_proof_1: update + master PDF
  rma push --dataset first_proof_2
  rma push --problems prob-01 prob-02
  rma push --pdf-only            # skip the update; just (re)build the master PDF
  rma push --no-meetings         # update concepts/insights/docs/issues, skip meetings
"""
from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path

log = logging.getLogger("rma.push")


def _find_repo_root(args) -> Path:
    cand = getattr(args, "repo_root", None)
    if cand:
        return Path(cand).resolve()
    here = Path.cwd()
    for d in [here, *here.parents]:
        if (d / "webapp" / "server.py").is_file():
            return d
    # Fallback to the known location.
    return Path("/projects/bhov/zzhao18/code/ResearchMathAgent-web")


def _update_concepts_and_insights(repo: Path, dataset: str, problems, force: bool) -> None:
    from webapp.dataset_store import list_problems, get_problem
    from webapp.concepts import load_concepts, generate_concepts
    from webapp.insight_agents import (
        generate_question_insight, generate_dataset_insight, generate_system_insight,
    )
    from webapp.proof_eval import evaluate_proof
    pids = problems or [p["id"] for p in list_problems(dataset=dataset)]
    for pid in pids:
        # Proof evaluation FIRST — refreshed every push so the report PDF reflects it.
        try:
            ev = evaluate_proof(repo, pid, dataset, force=True)
            log.info("eval %s: %s", pid, ev.get("error") or
                     f"answer={ev.get('answer_accuracy')} logic={ev.get('logical_correctness')} "
                     f"complete={ev.get('proof_completeness')} clarity={ev.get('proof_clarity')}")
        except Exception as exc:  # noqa: BLE001
            log.warning("eval %s failed: %s", pid, exc)
        # Comparison to the human reference solution (first_proof_2 only).
        if dataset == "first_proof_2":
            try:
                from webapp.human_solution import generate_human_comparison
                hc = generate_human_comparison(repo, pid, dataset, force=True)
                if hc.get("available"):
                    log.info("human-cmp %s: sim=%s complete=%s correct=%s steps=%s/%s",
                             pid, hc.get("approach_similarity"), hc.get("completeness_vs_human"),
                             hc.get("correctness_vs_human"), hc.get("ai_matched_steps"),
                             hc.get("human_key_steps"))
                else:
                    log.info("human-cmp %s: skipped (%s)", pid, hc.get("reason"))
            except Exception as exc:  # noqa: BLE001
                log.warning("human-cmp %s failed: %s", pid, exc)
        # Concepts (only if missing, unless --force)
        try:
            if force or not load_concepts(repo, pid):
                full = get_problem(dataset, pid) or {}
                stmt = full.get("statement") or full.get("tex") or ""
                if stmt:
                    list(generate_concepts(repo, pid, full.get("title", pid), problem_tex=stmt))
                    log.info("concepts: %s done", pid)
        except Exception as exc:  # noqa: BLE001
            log.warning("concepts %s failed: %s", pid, exc)
        # Per-question insight
        try:
            generate_question_insight(repo, pid, dataset)
            log.info("insight: %s done", pid)
        except Exception as exc:  # noqa: BLE001
            log.warning("insight %s failed: %s", pid, exc)
    for fn, label in ((lambda: generate_dataset_insight(repo, dataset), "dataset insight"),
                      (lambda: generate_system_insight(repo), "system insight")):
        try:
            fn(); log.info("%s done", label)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s failed: %s", label, exc)


def _build_reports(repo: Path, dataset: str, problems, languages, force: bool) -> None:
    """Compile each problem's context report in every requested language, caching
    a single date-time-stamped PDF + .tex per build into documents/cache/."""
    from webapp.context_report import compile_report_pdf
    from webapp.dataset_store import list_problems
    pids = problems or [p["id"] for p in list_problems(dataset=dataset)]
    for pid in pids:
        for lang in languages:
            label = "Chinese" if lang in ("cn", "zh") else "English"
            try:
                res = compile_report_pdf(repo, pid, dataset, force=force,
                                         cache_document=True, language=lang)
                if res.get("ok"):
                    log.info("report %s [%s]: %s", pid, label,
                             res.get("cached_copy") or res.get("log"))
                else:
                    log.warning("report %s [%s] failed: %s", pid, label, res.get("log"))
            except Exception as exc:  # noqa: BLE001
                log.warning("report %s [%s] error: %s", pid, label, exc)


def run_push(args) -> int:
    provider = getattr(args, "provider", None) or "claude-code"
    os.environ["RMA_PROVIDER"] = provider
    if provider == "claude-code":
        # Subscription path: drive the `claude` CLI; strip any API key so it uses
        # the Pro/Max OAuth credential.
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

    repo = _find_repo_root(args)
    sys.path.insert(0, str(repo))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    dataset = args.dataset
    problems = args.problems or None

    # Section-ablation: publish the spec to the environment so every downstream
    # report build in this process (per-problem PDFs + master) picks it up via
    # report_sections.resolve_sections. Empty/None ⇒ full report (canonical names).
    sections_spec = getattr(args, "sections", None)
    if sections_spec:
        os.environ["RMA_REPORT_SECTIONS"] = sections_spec

    print(f"[rma push] dataset={dataset} problems={problems or 'all'} "
          f"pdf_only={args.pdf_only} provider={provider}"
          + (f" sections={sections_spec}" if sections_spec else ""), flush=True)

    if not args.pdf_only:
        if not args.no_meetings:
            from webapp.push_forward import run_push_forward
            job = uuid.uuid4().hex
            print(f"[rma push] running push-forward (issues + meetings + documents)…", flush=True)
            run_push_forward(repo, job, problems=problems, max_resolve=args.max_resolve,
                             n_meeting_rounds=args.rounds, dataset=dataset)
        else:
            # Still refresh the per-problem documents even if skipping meetings.
            from webapp.rich_documents import update_question_document
            from webapp.dataset_store import list_problems
            for p in (problems or [x["id"] for x in list_problems(dataset=dataset)]):
                try:
                    update_question_document(repo, p)
                except Exception as exc:  # noqa: BLE001
                    log.warning("documents %s failed: %s", p, exc)
        print(f"[rma push] updating concepts + insights…", flush=True)
        _update_concepts_and_insights(repo, dataset, problems, force=args.force)

        # Per-problem context reports, one timestamped PDF+TeX per language.
        # Default is BOTH English and Chinese ("each push forward generates both").
        lang_arg = getattr(args, "language", "both") or "both"
        languages = ("en", "cn") if lang_arg == "both" else (lang_arg,)
        _build_reports(repo, dataset, problems, languages, force=args.force)

    print(f"[rma push] building master PDF (all problems, all tabs)…", flush=True)
    from webapp.context_report import compile_master_pdf
    res = compile_master_pdf(repo, dataset, force=args.force,
                             cache_document_scope=getattr(args, "cache_document", None))
    if res.get("ok"):
        path = repo / "documents" / "pdf" / f"master_{dataset}.pdf"
        size = path.stat().st_size if path.is_file() else 0
        print(f"\n✅ MASTER PDF ({res.get('parts')} sections, {size//1024} KB)")
        print(f"   file: {path}")
        print(f"   view: http://localhost:8001{res['pdf_url']}")
        print(f"   log:  {res['log']}")
        return 0
    print(f"\n❌ MASTER PDF FAILED: {res.get('log')}")
    return 1
