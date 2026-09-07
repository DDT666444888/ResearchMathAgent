#!/usr/bin/env bash
# Algorithm 1 faithfulness — phase check runner.
#
#   bash scripts/check_alg1.sh 0        # one phase
#   bash scripts/check_alg1.sh 0 1 2    # several
#   bash scripts/check_alg1.sh all      # everything implemented so far
#
# One PASS/FAIL line per task ID from ALGORITHM1_ACTION_PLAN.md. Every check is
# offline and costs no LLM tokens; the one [LIVE] task (T7.2) is never run here.
#
# Phases add their checks to this file as they land. A task whose phase is not
# implemented yet reports SKIP, so the runner is usable from Phase 0 onward.

set -uo pipefail

PY="${PY:-/sw/user/python/miniforge3-pytorch-2.11.0/bin/python3.12}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

if [ ! -x "$PY" ]; then
  echo "FATAL: python not found at $PY (override with PY=/path/to/python)" >&2
  exit 2
fi

PASS_N=0; FAIL_N=0; SKIP_N=0
FAILED_IDS=()

# run <task-id> <description> <command...>
run() {
  local id="$1" desc="$2"; shift 2
  local out
  if out=$("$@" 2>&1); then
    printf '  %-6s PASS  %s\n' "$id" "$desc"
    PASS_N=$((PASS_N + 1))
  else
    printf '  %-6s FAIL  %s\n' "$id" "$desc"
    printf '%s\n' "$out" | tail -15 | sed 's/^/           | /'
    FAIL_N=$((FAIL_N + 1))
    FAILED_IDS+=("$id")
  fi
}

skip() {
  printf '  %-6s SKIP  %s (not implemented yet)\n' "$1" "$2"
  SKIP_N=$((SKIP_N + 1))
}

# Assert a command's stdout matches a regex.
expect_match() {
  local id="$1" desc="$2" pattern="$3"; shift 3
  local out
  out=$("$@" 2>&1)
  if printf '%s' "$out" | grep -Eq "$pattern"; then
    printf '  %-6s PASS  %s\n' "$id" "$desc"
    PASS_N=$((PASS_N + 1))
  else
    printf '  %-6s FAIL  %s\n' "$id" "$desc"
    printf '           | expected match: %s\n' "$pattern"
    printf '%s\n' "$out" | tail -10 | sed 's/^/           | /'
    FAIL_N=$((FAIL_N + 1))
    FAILED_IDS+=("$id")
  fi
}

have() { [ -f "$1" ]; }

phase_0() {
  echo "Phase 0 — Unblock"
  expect_match T0.1 "rma doctor reports 0 blocking issues" \
    'Doctor passed' "$PY" -m rma doctor
  run   T0.1b "existing suite green (test_doctor, test_solve)" \
    "$PY" -m pytest tests/test_doctor.py tests/test_solve.py -q
  run   T0.2 "call_json bypasses the LaTeX gate; LM critic returns issues" \
    "$PY" -m pytest tests/test_models_json.py -q
  run   T0.3 "P0-P3 and issue_type survive a round trip" \
    "$PY" -m pytest tests/test_issue_api.py -q
  run   T0.4 "RunConfig carries the paper's defaults" \
    "$PY" -m pytest tests/test_config.py -q
  expect_match T0.4b "rma config --print shows the paper's parameters" \
    'n_rounds=5 issue_budget=5 context_budget=60000 model=claude-opus-4-8 effort=high max_output_tokens=64000' \
    "$PY" -m rma config --print
  run   T0.5 "five silent-data-loss holes closed" \
    "$PY" -m pytest tests/test_persistence_safety.py -q
  run   T0.7 "verifier gate is reachable by a real proof" \
    "$PY" -m pytest tests/test_verifier_gate.py -q
  run   T0.8 "narration-only reply salvages the document from disk" \
    "$PY" -m pytest tests/test_narration_salvage.py -q
}

phase_1() {
  echo "Phase 1 — The research store S"
  if have rma/store.py; then
    run T1.1 "Record + ResearchStore round-trip"    "$PY" -m pytest tests/test_store.py -q -k roundtrip
    run T1.2 "adapters onto existing persistence"   "$PY" -m pytest tests/test_store_adapters.py -q
    run T1.3 "versioned Pi + current_proof()"       "$PY" -m pytest tests/test_store.py -q -k revision
    run T1.4 "locking and atomic writes"            "$PY" -m pytest tests/test_store_concurrency.py -q
    run T1.5 "stateless / last-round-only stores"   "$PY" -m pytest tests/test_store_ablations.py -q
    expect_match T1.6 "rma inspect-store reports all 7 components" \
      '"current_proof_id"' \
      "$PY" -m rma inspect-store --problem q6 --dataset first_proof_1 --json
  else
    for t in T1.1 T1.2 T1.3 T1.4 T1.5 T1.6; do skip "$t" "store"; done
  fi
}

phase_2() {
  echo "Phase 2 — Run() and PrefixToBudget"
  if have rma/budget.py; then
    run T2.1 "token counting"                       "$PY" -m pytest tests/test_budget.py -q -k count
    run T2.2 "prefix_to_budget drops whole records" "$PY" -m pytest tests/test_budget.py -q -k prefix
    run T2.3 "Run() section order + write-back"     "$PY" -m pytest tests/test_orchestrator.py -q -k "order or writeback"
    run T2.4 "per-call telemetry"                   "$PY" -m pytest tests/test_orchestrator.py -q -k telemetry
    run T2.5 "context modes dump/truncate/budget"   "$PY" -m pytest tests/test_budget.py -q -k modes
    expect_match T2.6 "rma inspect-context builds C in the paper's order" \
      '"section_order"' \
      "$PY" -m rma inspect-context --problem q6 --unit critic --budget 60000 --json
  else
    for t in T2.1 T2.2 T2.3 T2.4 T2.5 T2.6; do skip "$t" "budget/Run"; done
  fi
}

phase_3() {
  echo "Phase 3 — Claim DAG and ranking"
  if have rma/claims.py; then
    run T3.1 "claim nodes and dependency edges"     "$PY" -m pytest tests/test_claims.py -q -k "nodes or edges"
    run T3.1b "edge-extraction recall gate"         "$PY" -m rma claims --eval tests/fixtures/claims/
    run T3.2 "terminal claims + proved fraction"    "$PY" -m pytest tests/test_claims.py -q -k terminal
    run T3.3 "dependency impact + cycles"           "$PY" -m pytest tests/test_claims.py -q -k "impact or cycle"
    run T3.4 "per-claim finite/numeric flags"       "$PY" -m pytest tests/test_claims.py -q -k finite
    run T3.5 "severity + three ordering modes"      "$PY" -m pytest tests/test_ranking.py -q
  else
    for t in T3.1 T3.2 T3.3 T3.4 T3.5; do skip "$t" "claims/ranking"; done
  fi
}

phase_4() {
  echo "Phase 4 — The six operations"
  # Guard each task on its own test file so an unimplemented one reports SKIP,
  # not FAIL: a directory existing does not mean every operation inside it does.
  if have tests/test_ops_base.py; then
    run T4.1 "Operation base + registry + FakeBackend"  "$PY" -m pytest tests/test_ops_base.py -q
    run T4.1b "no op bypasses Run()" bash -c \
      '! grep -rn --include=\*.py "call_anthropic\|call_claude_code\|call_json\|llm.complete" rma/ops/ | grep -v "/base.py:" | grep -q .'
  else
    skip T4.1 "operations"
  fi
  for spec in "T4.2:critic -> ranked Q:test_ops_critic.py" \
              "T4.3:solver patches locally:test_ops_solver.py" \
              "T4.4:literature keyed by Q, 4 fields:test_ops_literature.py" \
              "T4.5:meeting emits (rho, a, dH):test_ops_meeting.py" \
              "T4.6:revise consumes the action plan:test_ops_revise.py" \
              "T4.7:evaluator scores current pi into E:test_ops_evaluator.py" \
              "T4.8:UpdateConcepts:test_ops_concepts.py"; do
    id="${spec%%:*}"; rest="${spec#*:}"; desc="${rest%%:*}"; file="${rest##*:}"
    if have "tests/$file"; then
      run "$id" "$desc" "$PY" -m pytest "tests/$file" -q
    else
      skip "$id" "$desc"
    fi
  done
}

phase_5() {
  echo "Phase 5 — Round loop and termination"
  if have rma/finalize.py; then
    run T5.2 "FinalizeRound delivers the best round"  "$PY" -m pytest tests/test_finalize.py -q
  else
    skip T5.2 "finalize"
  fi
  if have rma/round_loop.py; then
    run T5.1 "round call sequence matches Alg. 1"   "$PY" -m pytest tests/test_round_loop.py -q -k "RoundOrder or Finalize"
    run T5.3 "Solved / Stalled"                     "$PY" -m pytest tests/test_termination.py -q
    run T5.4 "run_solve drives the orchestrator offline"  "$PY" -m pytest tests/test_round_loop.py -q -k "Driver or EndToEnd or Termination"
  else
    for t in T5.1 T5.3 T5.4; do skip "$t" "round loop"; done
  fi
}

phase_6() {
  echo "Phase 6 — Ablations, telemetry, CI"
  if have rma/ablations.py; then
    expect_match T6.1 "ablation matrix = full + every ablation" 'store.stateless' \
      bash -c "$PY -m rma ablate-matrix --list"
    run T6.1b "every config runs; no ablation is a no-op"  "$PY" -m pytest tests/test_ablations.py -q
    run T6.2 "figure-ready ablation summary"        "$PY" -m rma report-ablations --backend fake --out /tmp/abl_check.json
    run T6.3 "full suite offline (no key, no network)"  \
      env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN RMA_OFFLINE=1 RMA_DISABLE_TIKTOKEN= "$PY" -m pytest tests/ -q
  else
    for t in T6.1 T6.2 T6.3; do skip "$t" "ablations"; done
  fi
}

phase_7() {
  echo "Phase 7 — Converge the two half-systems"
  if have tests/test_single_store.py; then
    run T7.1 "push-forward and rma solve share one store" \
      "$PY" -m pytest tests/test_single_store.py -q
  else
    skip T7.1 "single store"
  fi
  printf '  %-6s SKIP  live end-to-end run (uses your Claude subscription; run by hand)\n' "T7.2"
  SKIP_N=$((SKIP_N + 1))
}

PHASES=("$@")
if [ ${#PHASES[@]} -eq 0 ]; then
  echo "usage: bash scripts/check_alg1.sh <phase|all> [phase...]" >&2
  exit 2
fi
if [ "${PHASES[0]}" = "all" ]; then
  PHASES=(0 1 2 3 4 5 6 7)
fi

echo "Algorithm 1 checks — python: $PY"
echo
for p in "${PHASES[@]}"; do
  case "$p" in
    0|1|2|3|4|5|6|7) "phase_$p" ;;
    *) echo "unknown phase: $p" >&2; exit 2 ;;
  esac
  echo
done

TOTAL=$((PASS_N + FAIL_N))
echo "-----------------------------------------------"
printf 'Result: %d/%d PASS' "$PASS_N" "$TOTAL"
[ "$SKIP_N" -gt 0 ] && printf ' (%d skipped)' "$SKIP_N"
echo
if [ "$FAIL_N" -gt 0 ]; then
  echo "FAILED: ${FAILED_IDS[*]}"
  exit 1
fi
echo "ALL CHECKS PASS"
