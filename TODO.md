# TODO — Algorithm 1 faithfulness

Master plan (with the target + runnable check for every task):
`../ALGORITHM1_ACTION_PLAN.md` (English) · `../ALGORITHM1_ACTION_PLAN_zh.md` (中文)

Validate any phase with:

```bash
export PY=/sw/user/python/miniforge3-pytorch-2.11.0/bin/python3.12
bash scripts/check_alg1.sh 0      # one phase
bash scripts/check_alg1.sh all    # everything implemented so far
```

## Phase 0 — Unblock

- [x] **T0.1** Green test baseline (`rma doctor` 0 blocking, suite green)
- [x] **T0.2** `call_json()` — turn the LM gap critic back on
- [x] **T0.3** Persist P0–P3 and `issue_type` through the API
- [x] **T0.4** `RunConfig` with the paper's defaults (N_R=5, b=5, B=60k, opus-4-8, effort high, 64k out)
- [x] **T0.5** Close the five silent-data-loss holes
- [x] **T0.6** `scripts/check_alg1.sh` phase runner
- [x] **T0.7** Verifier gate reachable — dropped two skeleton-template string checks that made `passed` impossible
- [x] **T0.8** Narration-only replies salvage the document the agent wrote to disk

## Phase 1 — The research store S

- [x] **T1.1** `Record` + `ResearchStore` skeleton
- [x] **T1.2** Adapters onto existing persistence (issues / meets / literature / concepts / insights / proof_eval)
- [x] **T1.3** Versioned Π + `current_proof()`
- [x] **T1.4** Locking and atomic writes
- [x] **T1.5** `StatelessStore` and `LastRoundOnlyStore` (= the paper's memory ablations)
- [x] **T1.6** `rma inspect-store` CLI

## Phase 2 — `Run()` and `PrefixToBudget`

- [x] **T2.1** Token counting (tiktoken, offline)
- [x] **T2.2** `prefix_to_budget()` drops whole records
- [x] **T2.3** `Run(u,q,S,B)` with the paper's exact section order
- [x] **T2.4** Per-call telemetry (`orchestration_log.jsonl`)
- [x] **T2.5** Context modes `dump | truncate | budget`
- [x] **T2.6** `rma inspect-context` CLI

## Phase 3 — Claim DAG and ranking

- [x] **T3.1** Claim nodes and dependency EDGES (recall gate ≥ 0.80)
- [x] **T3.2** Terminal claims + proved-terminal fraction
- [x] **T3.3** Dependency impact + cycle detection
- [x] **T3.4** Per-claim finite/numeric flags
- [x] **T3.5** Severity P0–P3 + three ordering modes

## Phase 4 — The six operations

- [x] **T4.1** `Operation` base, registry, `FakeBackend`
- [x] **T4.2** Critic → ranked `Q`
- [x] **T4.3** Solver: localized patch, not full rewrite
- [x] **T4.4** Literature keyed by `Q`, with the paper's four fields
- [x] **T4.5** Meeting emits `(ρ, a, ΔH)`
- [x] **T4.6** Revise (new operation)
- [x] **T4.7** Evaluator scores the current π into E
- [x] **T4.8** `UpdateConcepts(S, π, ΔH)`

## Phase 5 — Round loop and termination

- [x] **T5.1** `solve_problem()` in Algorithm-1 order
- [x] **T5.2** `FinalizeRound(S, r)` — deliver the best round, not the last (−27% open issues on the B2 run)
- [x] **T5.3** `Solved(π,e)` and `Stalled(S)`
- [x] **T5.4** `run_solve` becomes the driver

## Phase 6 — Ablations, telemetry, CI

- [x] **T6.1** `--ablate` covering all 13 paper configurations
- [x] **T6.2** Figure-ready run summary
- [x] **T6.3** `FakeBackend` and offline CI
- [x] **T6.4** Full acceptance run

## Phase 7 — Converge the two half-systems

- [x] **T7.1** push-forward on the same store
- [x] **T7.2** [LIVE] subscription path validated — evaluator (26s) + critic (193s) ran live on q6, real results into the store, zero API tokens. Full 5-round run available on request.
