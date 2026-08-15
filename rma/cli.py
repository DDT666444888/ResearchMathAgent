from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from .ablations import run_ablate_matrix, run_report_ablations
from .claims import run_claims
from .config import ABLATIONS, CONTEXT_MODES, DEFAULT_N_ROUNDS, run_config
from .doctor import run_doctor
from .orchestrator import run_inspect_context
from .push import run_push
from .solve import run_diff, run_parse, run_propose, run_refine, run_solve, run_verify
from .store import run_inspect_store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rma",
        description="Research Math Agent command-line tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor",
        help="Check whether the local repository is ready for RMA development.",
    )
    doctor.add_argument(
        "--repo-root",
        default=None,
        help="Repository root to inspect. Defaults to the current directory or its parents.",
    )
    doctor.set_defaults(func=run_doctor)

    parse = subparsers.add_parser(
        "parse",
        help="Parse one or all First Proof problem statements into structured run artifacts.",
    )
    _add_pipeline_arguments(parse, render=False, max_rounds=False)
    parse.set_defaults(func=run_parse)

    propose = subparsers.add_parser(
        "propose",
        help="Generate complete initial solution proposals from parsed problem artifacts.",
    )
    _add_pipeline_arguments(propose, render=False, max_rounds=False)
    propose.set_defaults(func=run_propose)

    verify = subparsers.add_parser(
        "verify",
        help="Run verifier checks on proposed/current solution artifacts.",
    )
    _add_pipeline_arguments(verify, render=True, max_rounds=False)
    verify.set_defaults(func=run_verify)

    refine = subparsers.add_parser(
        "refine",
        help="Apply verifier feedback to the current solution artifact.",
    )
    _add_pipeline_arguments(refine, render=False, max_rounds=False)
    refine.set_defaults(func=run_refine)

    solve = subparsers.add_parser(
        "solve",
        help="Run parser -> proposer -> verifier/refiner pipeline for one or all First Proof problems.",
    )
    _add_pipeline_arguments(solve, render=True, max_rounds=True)
    solve.add_argument(
        "--resume",
        action="store_true",
        help="Skip problems that are already marked verified in the output folder.",
    )
    solve.add_argument(
        "--fast",
        action="store_true",
        help="Skip verify and refine stages — only parse and propose. Useful for quick first-pass proof generation.",
    )
    solve.add_argument(
        "--strategies",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel proof strategies to attempt per problem (default: 1). "
            "When N > 1, a planner proposes N distinct approaches, sanity-checks each, "
            "runs them in parallel, and picks the best result by verifier score."
        ),
    )
    solve.add_argument(
        "--parent-run",
        default=None,
        metavar="RUN_ID",
        help="Parent experiment run ID for DAG lineage tracking (written to meta.json).",
    )
    solve.add_argument(
        "--dataset",
        default=None,
        metavar="SLUG",
        help="Dataset slug (e.g. aim_problem_lists, erdos_problems) when solving non-first_proof_1 problems.",
    )
    solve.add_argument(
        "--orchestrator",
        action="store_true",
        help="Run the Algorithm 1 round loop (critic -> solver -> literature -> "
             "meeting -> revise -> concepts -> evaluator) over the research store. "
             "This is the DEFAULT; the flag is kept for explicitness.",
    )
    solve.add_argument(
        "--legacy-pipeline",
        action="store_true",
        dest="legacy_pipeline",
        help="Opt into the older parse->propose->verify/refine pipeline instead of "
             "the default Algorithm 1 orchestrator.",
    )
    solve.add_argument(
        "--paper-faithful",
        action="store_true",
        dest="paper_faithful",
        help="Reproduce paper Algorithm 1 LITERALLY: each round uses "
             "pi = CurrentProof(S) = the last revision and the run outputs that "
             "final pi, with no best-of-rounds selection and no carry-forward. "
             "The default adds those two enhancements.",
    )
    solve.add_argument(
        "--backend",
        choices=("auto", "fake"),
        default="auto",
        help="Model backend for the orchestrator. 'fake' runs fully offline with "
             "deterministic canned artifacts (no tokens); implies --orchestrator.",
    )
    # `rma solve <q>` works out of the box on the user's Claude subscription
    # (claude-code = local `claude` CLI, billed to their Pro/Max plan — no API
    # key). Override with --model-name rma-skeleton for offline runs.
    solve.set_defaults(func=run_solve, model_name="claude-code")

    config = subparsers.add_parser(
        "config",
        help="Show the resolved run configuration (Algorithm 1 parameters).",
    )
    config.add_argument("--print", action="store_true", dest="print_config",
                        help="Print the resolved configuration (default action).")
    config.add_argument("--json", action="store_true", help="Emit JSON instead of a summary line.")
    _add_orchestration_arguments(config)
    config.set_defaults(func=run_config, model_name=None, max_rounds=None)

    inspect_store = subparsers.add_parser(
        "inspect-store",
        help="Show the research store S = (Pi, I, M, L, K, H, E) for one problem.",
    )
    inspect_store.add_argument("--problem", required=True, help="Problem id, e.g. q6.")
    inspect_store.add_argument("--dataset", default="first_proof_1", help="Dataset slug.")
    inspect_store.add_argument("--memory", choices=("full", "last-round-only", "stateless"),
                               default="full", help="Memory model to read with.")
    inspect_store.add_argument("--json", action="store_true", help="Emit JSON.")
    inspect_store.add_argument("--repo-root", default=None)
    inspect_store.set_defaults(func=run_inspect_store)

    inspect_context = subparsers.add_parser(
        "inspect-context",
        help="Show the compiled observation O an operation would receive (no model call).",
    )
    inspect_context.add_argument("--problem", required=True, help="Problem id, e.g. q6.")
    inspect_context.add_argument("--dataset", default="first_proof_1", help="Dataset slug.")
    inspect_context.add_argument("--unit", default="critic",
                                 help="Operation name (critic, solver, literature, ...).")
    inspect_context.add_argument("--budget", type=int, default=None,
                                 help="Context budget B in tokens (default: 60000).")
    inspect_context.add_argument("--show-text", action="store_true", dest="show_text",
                                 help="Include the compiled observation itself.")
    inspect_context.add_argument("--json", action="store_true", help="Emit JSON.")
    inspect_context.add_argument("--repo-root", default=None)
    _add_orchestration_arguments(inspect_context)
    inspect_context.set_defaults(func=run_inspect_context, model_name=None, max_rounds=None)

    ablate = subparsers.add_parser(
        "ablate-matrix",
        help="List the runnable ablation configurations (paper Figure 5).",
    )
    ablate.add_argument("--list", action="store_true", dest="list_configs",
                        help="List config names, one per line (default action).")
    ablate.add_argument("--json", action="store_true", help="Emit JSON.")
    ablate.set_defaults(func=run_ablate_matrix)

    report_abl = subparsers.add_parser(
        "report-ablations",
        help="Run every ablation config offline and emit figure-ready metrics.",
    )
    report_abl.add_argument("--backend", choices=("fake",), default="fake",
                            help="Only offline 'fake' is supported (deterministic).")
    report_abl.add_argument("--rounds", type=int, default=3, help="Rounds per config.")
    report_abl.add_argument("--out", default=None, help="Write JSON here.")
    report_abl.set_defaults(func=run_report_ablations)

    claims = subparsers.add_parser(
        "claims",
        help="Parse a proof into its claim-dependency graph, or evaluate extraction recall.",
    )
    claims.add_argument("tex", nargs="?", help="Path to a .tex proof.")
    claims.add_argument("--eval", dest="eval_dir", default=None,
                        help="Directory of *.tex + *.expected.json fixtures to score.")
    claims.add_argument("--min-recall", type=float, default=0.80, dest="min_recall",
                        help="Recall gate for --eval (default 0.80).")
    claims.add_argument("--json", action="store_true", help="Emit JSON.")
    claims.set_defaults(func=run_claims)

    diff = subparsers.add_parser(
        "diff",
        help="Compare verification results between two experiment output folders.",
    )
    diff_target = diff.add_mutually_exclusive_group(required=True)
    diff_target.add_argument("--exp-a", default=None, help="First experiment name (under outputs/first_proof_1/).")
    diff_target.add_argument("--output-a", default=None, help="Absolute path to first experiment folder.")
    diff_b = diff.add_mutually_exclusive_group(required=True)
    diff_b.add_argument("--exp-b", default=None, help="Second experiment name (under outputs/first_proof_1/).")
    diff_b.add_argument("--output-b", default=None, help="Absolute path to second experiment folder.")
    diff.add_argument("--repo-root", default=None)
    diff.set_defaults(func=run_diff)

    push = subparsers.add_parser(
        "push",
        help="Run the push-forward, update every tab, and build ONE huge combined PDF (all problems, all tabs).",
    )
    push.add_argument("--provider", default="claude-code", choices=["claude-code", "api"],
                      help="LLM backend (default: claude-code = Pro/Max subscription).")
    push.add_argument("--dataset", default="first_proof_1", help="Dataset slug (default: first_proof_1).")
    push.add_argument("--problems", nargs="*", default=None, help="Problem IDs to update (default: all in dataset).")
    push.add_argument("--rounds", type=int, default=1, help="Meeting discussion rounds (default: 1).")
    push.add_argument("--max-resolve", type=int, default=2, dest="max_resolve",
                      help="Max issues to resolve per problem (default: 2).")
    push.add_argument("--pdf-only", action="store_true", dest="pdf_only",
                      help="Skip the update; just (re)build the master PDF from current content.")
    push.add_argument("--no-meetings", action="store_true", dest="no_meetings",
                      help="Skip the meeting/issue cycle; still refresh docs + concepts + insights.")
    push.add_argument("--language", "--lang", default="both", dest="language",
                      choices=["both", "en", "cn", "zh"],
                      help="Report language(s) to generate per problem (default: both = "
                           "English + Chinese). 'cn'/'zh' = Chinese only, 'en' = English only.")
    push.add_argument("--cache-document", default=None, dest="cache_document",
                      help="Problem id whose report PDF to also copy into documents/cache/ for quick access (e.g. prob-09).")
    push.add_argument("--sections", default=None, dest="sections",
                      help="Ablate report sections for this build, e.g. "
                           "--sections 'concepts=0,meetings=off' or "
                           "--sections no_concepts,no_meetings. Ablated PDFs are written to "
                           "*__abl-<sig>.pdf so they never overwrite the full report. "
                           "Available: evaluation, research_status, push_forward_history, "
                           "problem_statement, best_proof, concepts, meetings, open_issues, "
                           "resolved_issues, insights, candidate_answer, strategy.")
    push.add_argument("--force", action="store_true", help="Force regenerate concepts/insights and recompile all reports.")
    push.add_argument("--repo-root", default=None)
    push.set_defaults(func=run_push)

    return parser


def _add_orchestration_arguments(parser: argparse.ArgumentParser) -> None:
    """Algorithm 1 parameters (paper defaults live in rma/config.py)."""
    group = parser.add_argument_group("Algorithm 1 orchestration")
    group.add_argument("--rounds", type=int, default=None, metavar="N_R",
                       help=f"Research rounds N_R (default: {DEFAULT_N_ROUNDS}).")
    group.add_argument("--issue-budget", type=int, default=None, metavar="B_ISSUES",
                       dest="issue_budget",
                       help="Issues repaired per round, b (default: 5).")
    group.add_argument("--context-budget", type=int, default=None, metavar="TOKENS",
                       dest="context_budget",
                       help="Per-call context budget B in tokens (default: 60000).")
    group.add_argument("--context-mode", choices=CONTEXT_MODES, default=None,
                       dest="context_mode",
                       help="How context is fitted to B (default: budget = PrefixToBudget).")
    group.add_argument("--effort", default=None,
                       help="Reasoning effort for the backbone (default: high).")
    group.add_argument("--ablate", default=None, metavar="SPEC",
                       help="Comma-separated ablations. Valid names: " + ", ".join(ABLATIONS))


def _add_pipeline_arguments(parser: argparse.ArgumentParser, *, render: bool, max_rounds: bool) -> None:
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "problem",
        nargs="?",
        help="Problem id, e.g. q6.",
    )
    target.add_argument(
        "--all",
        action="store_true",
        help="Run this stage for all q1 through q10 problems.",
    )
    parser.add_argument(
        "--tier",
        choices=("budget", "standard", "pro"),
        default="standard",
        help="Execution profile to record for the run.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output experiment directory. Defaults to outputs/first_proof_1/<exp-name>_<model-name>.",
    )
    parser.add_argument(
        "--exp-name",
        default=None,
        help="Experiment name for the output subfolder. Defaults to proofs_v1_<month><day>.",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("RMA_MODEL", "rma-skeleton"),
        help="Model name for generation and the output subfolder. Honors "
             "RMA_MODEL (e.g. claude-fable-5); falls back to rma-skeleton (offline).",
    )
    parser.add_argument(
        "--model-provider",
        choices=("auto", "offline", "anthropic", "claude-code"),
        default="auto",
        help="Generation backend. auto uses offline for rma-skeleton, Anthropic API for claude-* models, and Claude Code for claude-code.",
    )
    if render:
        parser.add_argument(
            "--no-render",
            action="store_true",
            help="Skip rendering qN_solution.tex to PDF during verification.",
        )
    else:
        parser.set_defaults(no_render=True)
    if max_rounds:
        parser.add_argument(
            "--max-rounds",
            type=int,
            default=None,
            help=f"Maximum verifier/refiner rounds per problem (default: N_R = {DEFAULT_N_ROUNDS}).",
        )
        _add_orchestration_arguments(parser)
    else:
        parser.set_defaults(max_rounds=None)
    parser.add_argument(
        "--skill-path",
        default="skills/math-research/SKILL.md",
        help="Math research skill instructions to load for this pipeline stage.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root to inspect. Defaults to the current directory or its parents.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
