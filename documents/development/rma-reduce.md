# `rma reduce`: compiler-guided Lean refactoring

Reduce an **explicitly supplied** valid submission, using the existing RMA research-context orchestrator and persistent research store. The original `rma solve` research benchmark remains separate. Reduction never loads previous RMA solution archives.

## Install and run

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[webapp]'
.venv/bin/rma reduce --help
```

The webapp extra is currently needed by the upstream persistence adapters. It does not route reduction requests to Anthropic. Reduction uses the explicitly injected Responses backend only.

```sh
.venv/bin/rma reduce \
  --benchmark /path/to/benchmark.jsonl \
  --baseline /path/to/submission.jsonl \
  --baseline-results /path/to/server-result.json \
  --manifest /path/to/manifest.json \
  --repo strata=/path/to/pinned/strata \
  --lake /path/to/elan/bin/lake --elan-home /path/to/elan \
  --credentials-file /private/path/to/credentials.txt \
  --model gpt-6-astra \
  --ledger /path/to/existing/usage.jsonl --budget-usd 100 --per-problem-usd 12 \
  --ids 03 09 --rounds 2 --strategies 2 --repairs 1 --beam 2 --jobs 2 \
  --out /path/to/new-reduction-run
```

Check the endpoint, key and deployment first — one metered call, no Lean toolchain needed:

```sh
.venv/bin/rma reduce --probe --model gpt-6-astra \
  --credentials-file /private/path/to/credentials.txt --out /path/to/preflight
```

A reduction run repeats that probe automatically before compiling any baseline (`--no-preflight` disables it), so a wrong deployment name fails in seconds instead of after the first Lean build.

Credentials are either a two-line file (endpoint, API key), matching the user's existing Azure project configuration, or JSON with `endpoint` and `api_key`. Azure Foundry project endpoints are converted to the same host's `/openai/v1/responses`; ordinary OpenAI-compatible HTTPS base URLs are also supported. The key travels through curl stdin, never command-line arguments or artifacts. The deployment name is not substituted.

Use `--dry-run` to validate inputs without paid calls or compilation. Add `--resume` for the same run; `--rounds` is a **total** target, not additional rounds. Source/baseline/model fingerprints must match. Each restored best proof is independently recompiled. Interrupted/unknown requests are not replayed. Use a new output directory when selecting a different problem set or changing pinned repositories.

## Architecture

| Existing RMA mechanism | Lean adaptation |
|---|---|
| `ResearchStore`: Pi/I/M/L/K/H/E | Verified proofs, cost issues, strategy plans, scoped facts, invariants, candidate/history notes, compiler evaluations |
| `run_unit`, `Query`, `PrefixToBudget` | Every solver/reviser receives the bounded observation and writes results with telemetry and links |
| `ranking.rank` | Optimization issue queue; explicit strategy portfolio |
| `termination.stalled` | Stop on lack of measured improvement, round limit, or shared budget exhaustion |
| Best-proof carry-forward | Pi changes only after improved local utility **and** an independent kernel/axiom gate |
| Store persistence and locking | Per-problem isolated research workspace, resumable states, cross-process budget locks |
| Verifier/refiner cycle | Real Lean errors are located in the submitted declaration and linked into the repair request; failed proofs never become current |
| `memory.record_attempt` (strategy memory) | `Playbook`: promoted transformations shared across problems as tactic vocabulary and measured deltas, never whole proofs |
| Ablation matrix protocol | `--ablate beam bandit playbook feedback` reduces one mechanism to its simplest functioning variant, so its contribution is measurable |

### Search over kernel-verified proofs

Greedy hill-climbing on a single best proof is the degenerate case of all three mechanisms, and it stalls on proofs whose shortest form is not reachable from the current one by one edit:

- **Beam** (`--beam k`, default 2): every kernel-verified candidate — including ones that did *not* beat the best — stays in a per-problem pool ranked by local utility. Round *r* seeds slot *i* from pool entry *i*, so one slot always refines the best proof while the others explore alternative verified structures. Promotion to Pi is unchanged: a candidate must beat the best **and** pass the independent uninstrumented recompilation and every configured version.
- **Strategy bandit**: the portfolio (structural compression / library reuse / elaboration efficiency) is selected per round by UCB1 over the measured utility gain each strategy has produced in this run's state, with untried strategies explored first. Selection is deterministic given the persisted state, so a `--resume` replays the same choices.
- **Playbook**: a promotion appends `dropped`/`introduced` tactic vocabulary plus length and heartbeat deltas to `playbook.jsonl`, shared across concurrent `--jobs` workers by the same file lock as the ledger. Later problems receive the top lessons as an `L` record inside the ordinary context budget.

`--ablate` is repeatable and recorded in `plan.json`; `results.json` reports per-problem `attempts`, `promoted`, `pool_size` and `strategy_stats`, which is the signal vector an ablation comparison needs. `--report` renders finished runs side by side into `OUT/report.md` without touching the API or the toolchain:

```sh
.venv/bin/rma reduce --report --out /path/to/full-run --compare /path/to/no-beam-run
```

The existing LaTeX current-proof heuristic rejects extreme shortening as degeneration. `ReductionStore` overrides only current-proof selection: any kernel-audited promoted revision is eligible, even if it is 90% shorter. Candidate replies live in H until promotion to Pi. The webapp's canonical proof history still uses `.tex` filenames internally; the actual Lean artifacts and exported submission use `.lean`/`.jsonl`.

Critic, strategy meeting, concepts, and evaluator stages are deterministic for this workflow. Paid calls are reserved for generating or repairing proofs. Unrelated paper search and LaTeX reviewers are off. `--hints facts.json` supplies reviewed per-id facts. Opt-in `--local-context` retrieves whole declarations referenced by the proof **only from before the target in its own source file**, within a character cap; saved excerpts then pass through RMA's context budget. It does not search historical solutions or import later results.

## Defaults

`--mode rma` is the default. The modules below are on by default because each has measured evidence on this
benchmark; `--mode flat` keeps the older full-rewrite portfolio (beam / bandit / playbook) for comparison.

| default-on module | why (measured) |
|---|---|
| kernel gate + public-rule scoring | local per-problem scores match the official ones within ~0.1 points (8/8 on the live 20260911T221250Z run) |
| strategy-driven rewrite each round | the largest single jumps came from full rewrites (11: +5.2/+6.1 problem points in flat) |
| Lean lint probe (`unusedTactic`, unused simp args) + deterministic deletion | free, kernel-backed; 03 gained +4.3 problem points with zero model calls |
| issue-driven localized patches | 9 of 10 promotions in the upgraded rma run |
| one compiler-feedback repair for high-value issues | produced 07's +1.56 |
| Chain-of-States probe | ImProver: symbolic proof state is what makes LLM rewriting reliable |
| grounded library lookup, meeting → revise, concepts (failed moves) | the meeting's plan produced 10's second promotion; failed moves stop repeats |
| API-free local search each round | free; 05 +0.05, 01 +0.11, and Codex's 03 result |

**Honest status:** at equal dollars on 04/05/08/11, flat still scored higher (+7.06 / +7.36 vs +4.57 for rma), and the
whole gap was problem 11. The every-round rewrite above is the change meant to close it and is **not yet validated**;
a paired equal-budget run is pending budget.

## `--mode rma`: Algorithm 1 for proof reduction

The default `--mode flat` spends every paid call on a full rewrite. `--mode rma` runs RMA's research loop per round, each
unit writing to its store component and reading the others back through `Run`'s bounded context:

| unit | store | what it does |
|---|---|---|
| Critic | I | Lean's own lints (unused simp arguments), repeated tactic steps, and an LM critic's located issues, each with an expected public-score gain; the queue is ranked by that gain |
| Solver | Pi | one exactly anchored patch per selected issue (`--issues-per-round`); lint fixes need no model call; a non-compiling patch for an issue worth ≥ 1 point gets one repair turn with the located compiler error |
| Literature | L | grep of proposed and compiler-unknown lemma names in the pinned sources and packages (field notation and local hypotheses stripped) |
| Concepts | K | failed moves with Lean's reason, linked into later rounds so they are not repeated |
| Meeting | M, H | coordinator, Lean golfer, elaboration engineer and library expert over I, E, K, L → action plan and insights |
| Revise | Pi | one coordinated rewrite executing the action plan |
| Evaluator | E | kernel compilation and the public score; an issue is resolved when its patch is promoted, `wontfix` after two failed or repeated moves |

Promotion uses the same gate as flat mode. `--rma-ablate critic.lm|critic.structural|literature|meeting|concepts|insights|evaluator-feedback`
switches one unit off for a controlled comparison. First live run (problem 10, one round, round-6 official start):
77.35 → 79.72 problem points for $5.06, with both promotions produced by RMA units (an issue-driven patch, then meeting → revise).

## Evaluation and invariants

- Preserve the exact theorem statement and original source declaration separator.
- Reject admissions, metaprogramming, environment mutation, or extra top-level declarations in candidate bodies.
- Compile a uniquely named temporary file in the pinned source repository; original source files stay unchanged.
- Kill the entire compiler process group on timeout, and always remove temporary source files.
- Check the target's final axiom audit: only `propext`, `Classical.choice`, and `Quot.sound` are accepted.
- Measure heartbeats in a separate instrumented run. Recompile promising candidates without instrumentation.
- Parse Lean's `file:line:col: error:` stream and re-anchor it onto the submitted declaration, so a repair turn sees the failing tactic with its surrounding lines instead of a truncated log tail. The raw tail is still used when nothing parseable was emitted (timeouts, fatal errors). `--ablate feedback` withholds the compiler text entirely.
- Record three lengths per measurement: `tokens_arena` (the fitted Arena lexer, used for scoring), the older regex `tokens_proxy`, and `tokens_model` from RMA's tokenizer.

## The objective: the public Arena score, computed locally

    problem score = mean( length reduction % , heartbeat reduction % , zero-shot compatibility % )
    run score     = mean of the problem scores over ALL benchmark problems

`rma/reduction/score.py` implements this rule and the pipeline ranks candidates by it; nothing waits for the server to decide whether a candidate is better. Length reduction is `1 − L/R` with `R` the benchmark's `proof_length` of the ORIGINAL proof; heartbeat reduction is `1 − H/RH` with `RH` the official reference count (or the original proof measured locally when unpublished) and is not clamped; zero-shot is the share of listed `version_info` toolchains on which the proof compiles unchanged. At fixed compatibility, one token is worth `100/(3R)` points and one heartbeat `100/(3·RH)`, so a shorter but slower proof can lose. Every solver prompt carries the rule, the problem's denominators, the current proof's axes and these exchange rates.

- **Length.** The Arena's tokenizer runs on its evaluation worker. `arena_tokens` is a Lean-aware lexer (identifiers with dotted/primed/subscript parts, numerals, bracket characters, symbol runs; comments ignored) fitted to 41 official (proof, length) pairs from this benchmark: 21 exact, mean absolute error 1.3 tokens, worst 11. GPT BPE tokenizers overcount by 80–180% and are not used.
- **Heartbeats.** The Arena harness measured locally reproduces the server's count for an unchanged proof on the same toolchain.
- **Zero-shot.** Pass `--repo CORPUS=PATH` once per prepared checkout of a listed commit. Only repositories whose toolchain is listed for a problem are checked for it. Versions without a local checkout are never assumed: `state.json` and `results.json` report the score as a lower bound (they fail) and an upper bound (they pass), with the unverified versions named.
- `results.json` → `estimated_score` holds `run_lower`, `run_upper` and every problem's bounds; problems without local state use the official row of their baseline proof, labelled as such.
- Repeat `--repo CORPUS=PATH` for additional prepared versions; promotion requires all configured versions to pass. Versions not configured locally remain **unverified locally** and are reported as score bounds.
- Rank with a **local surrogate**, not claimed official scores. With official baseline results, calibrate the regex length proxy to baseline official tokens and use original reference heartbeat denominators. Preserve both length and heartbeat tradeoffs. Arena may rank candidates differently; retain prior submissions for fallback.

## Budget and recovery

The ledger remains compatible with earlier Arena `usage.jsonl` reservation entries. Each dispatch atomically reserves an upper-bound token estimate using UTF-8 byte count and `max_output_tokens`. Default planning rates ($100/M input, $500/M output) are intentionally conservative and **are not Azure prices**. Actual invoices are separate. Do not delete/reset the ledger to bypass an existing user budget.

A **completed** response then settles that hold down to the server's metered `input_tokens`/`output_tokens` at `--settle-input-rate` / `--settle-output-rate` (defaulting to the planning rates, so settlement prices real tokens rather than the worst case). Settlement can only release budget, never add to it: an unknown transport state, a non-completed response, a missing usage report, or a usage report that would cost more than the pre-dispatch bound all keep the full upper bound. This matters in practice — at the defaults a 12k-token dispatch holds $6.00, which alone would cap a $100 budget at sixteen solver calls; a measured probe against the live deployment settles to $0.0036. `results.json` reports `reserved_usd`, `settled_usd`, `held_usd` and `unsettled_calls` so the retained holds stay visible.

`--per-problem-usd` caps one problem's share of the shared ledger, so a single pathological theorem cannot drain the run before the other workers start. The cap is enforced inside the same cross-process lock as the global limit.

Local context budgeting uses RMA's tokenizer/estimate, not Azure's billing tokenizer. Mandatory theorem/proof context exceeding the configured context budget causes rejection before a paid call.

## Artifacts

- `submission.jsonl` / `submission.zip`: all benchmark entries; untouched entries retain the supplied baseline.
- `results.json`: promoted proofs, measurements, per-problem attempt/strategy statistics, errors, ledger summary, and submission hash; no invented official score.
- `preflight.json`: the probe's deployment, URL, metered usage and ledger state.
- `playbook.jsonl`: promoted transformations shared across problems (vocabulary and deltas only).
- `report.md`: the `--report` comparison table, when rendered.

## What has been exercised against the live deployment

The `gpt-6-astra` deployment on the NAIRR Azure project endpoint answered a `--probe` (16 metered tokens, settled at $0.0036 under the default conservative rates, against a $1.00 pre-dispatch hold). A full `rma reduce` run — preflight, RMA store orchestration, two strategy slots, promotion gate, submission export and zip — completed end to end against that deployment with a stub kernel standing in for `lake`, since no Lean toolchain is installed on the development machine. That run also exercised the interruption path: a solver call killed in flight kept its $2.1616 reservation, `--resume` refused to replay it, ran the remaining slot instead, and reported the retained hold as `unsettled_calls: 1`.

Kernel-dependent behaviour — real heartbeat measurement, axiom audits, cross-version compatibility, diagnostics from an actual Lean error stream — is covered by unit tests only and still needs one run on the cluster with the pinned repositories before any submission.
- `problems/ID/`: state, candidate replies, `.lean` bodies, compiler logs and independent verification logs.
- `problems/ID/research/`: the reused seven-component RMA store and `orchestration_log.jsonl`.
- `api/REQUEST_ID/`: sanitized request and response payloads; credentials omitted.
- `inputs.json`, `plan.json`: fingerprints and run configuration.

No remote submission occurs automatically in `rma reduce`. The user-facing assistant can submit the exported file using the existing Arena authorization after reviewing results.

## Repository checkout note

Upstream commit `c1b6d1e79e334a62a6ae68322c7c985a3b0a270a` includes four pairs of historical webapp data paths differing only in letter case. macOS's default case-insensitive filesystem reports those as modified immediately after clone. This implementation does not modify or include those historical-data differences in its patch. All upstream objects remain available in Git; use a case-sensitive checkout to materialize both variants simultaneously.


## Public scoring objective (required for every reduction prompt and review)

The score formula is public, so compute tradeoffs locally instead of deferring the
judgment of benefit to the server:

- `Lred = 100 * (1 - candidate_length / original_reference_length)`.
- `Hred = 100 * (1 - candidate_heartbeats / original_reference_heartbeats)`.
- `problem_score = (Lred + Hred + zero_shot_compatibility_pct) / 3`.
- `your_score = sum(all 15 problem_scores) / 15`; untouched problems remain included.
- At unchanged compatibility, `delta_problem = 100/3 * ((old_length-new_length)/reference_length + (old_HB-new_HB)/reference_HB)`.
- Across changed problems, `delta_total = sum(delta_problem) / 15`.

A shorter proof with more heartbeats can improve the score. Compare normalized savings
against the ORIGINAL benchmark denominators, not percentage changes against the latest
proof. If compatibility changes, add `delta_compatibility_pct / 3` to the problem delta.
Do not silently assign 100% to untested versions.

Use exact measured inputs when available; explicitly label proxy-length predictions and
compatibility assumptions. Official evaluation verifies those inputs and compatibility;
it is not necessary to understand or calculate the scoring tradeoff. The existing
calibrated utility preserves this ordering at fixed compatibility; its delta converts
to problem-score points by multiplying by 2/3. Without original reference calibration,
the fallback utility is only a search heuristic and cannot establish official-score gain.
