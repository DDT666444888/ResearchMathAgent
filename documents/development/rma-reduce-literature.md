# Why Lean proofs shrink so much, and what the literature says about doing it well

Scope: the 15 Lean Refactor Arena problems (strata, physlib, cslib, arklib, putnambench), our best proofs as of
round 7 (official 79.58%), and published work on automated proof optimization. Every number below is either an
official Arena count or a real Lean compilation on the pinned toolchains; the length lexer is a fitted proxy.

## 1. Where the reductions come from (original reference proof → current best)

| id | corpus | length (official) | heartbeats (official) | non-blank lines |
|---|---|---|---|---|
| 01 | arklib | 1906 → 455 (−76%) | 45,273 → 9,631 (−79%) | 150 → 41 |
| 02 | cslib | 407 → 189 (−54%) | 1,383 → 613 (−56%) | 64 → 24 |
| 03 | strata | 222 → 84 (−62%) | 2,696 → 962 (−64%) | 78 → 15 |
| 04 | physlib | 1217 → 459 (−62%) | 42,225 → 16,172 (−62%) | 111 → 50 |
| 05 | arklib | 1463 → 463 (−68%) | 15,645 → 6,311 (−60%) | 142 → 46 |
| 06 | arklib | 1353 → 341 (−75%) | 9,198 → 3,556 (−61%) | 126 → 35 |
| 07 | cslib | 408 → 319 (−22%) | 59,830 → 1,800 (−97%) | 83 → 37 |
| 08 | physlib | 1372 → 529 (−61%) | 26,970 → 9,059 (−66%) | 117 → 56 |
| 09 | strata | 313 → 133 (−58%) | 5,175 → 1,757 (−66%) | 92 → 16 |
| 10 | physlib | 851 → 331 (−61%) | 11,947 → 3,470 (−71%) | 81 → 30 |
| 11 | cslib | 251 → 138 (−45%) | 3,858 → 470 (−88%) | 61 → 19 |
| 12 | strata | 224 → 95 (−58%) | 4,251 → 236 (−94%) | 47 → 11 |
| 13 | putnambench | 1769 → 648 (−63%) | 111,476 → 7,091 (−94%) | 175 → 47 |
| 14 | putnambench | 4085 → 513 (−87%) | 14,130 → 2,742 (−81%) | 233 → 47 |
| 15 | putnambench | 1754 → 433 (−75%) | 134,499 → 2,497 (−98%) | 268 → 47 |

Tactic-family counts summed over all 15 proofs (original → best): structural tactics (`have show calc suffices obtain
rcases cases induction intro rintro refine constructor use`) 550 → 171; simp family (`simp simp_all simpa simp_rw dsimp
norm_num field_simp ring_nf push_cast norm_cast`) 259 → 67; heavy closers (`omega linarith nlinarith positivity decide
ring aesop grind tauto gcongr fun_prop …`) 195 → 35; rewriting 150 → 47; lemmas inside `simp only [...]` 235 → 93.

### The mechanisms, with the proofs that show them

1. **Shotgun automation → explicit reasoning (heartbeats).** Machine-generated originals (all three putnambench
   problems) chain `simp_all <;> aesop`, `tauto`, `nlinarith`, `norm_num [...] <;> linarith` over dozens of nested
   `have h₁ … h₁₁`. Problem 15 goes from 289 lines to 47 by isolating three real lemmas (`missing`, `either`, `excl`) and
   finishing with `Finset.card_bij'` + `omega`: −98% heartbeats. The same happens in research code: 07 replaces one
   `grind` per constructor case by explicit constructor terms, −97% heartbeats for only −22% length.
2. **Merging symmetric or parallel branches (length).** 11 spells out the `left`/`right` halves of a bisimulation
   separately; the best proof uses `refine ⟨?_, ?_⟩ <;>`, `first | have g := hb.follow_fst t | have g := hb.follow_snd
   t`, and multi-alternative `cases … with | a | b`. 07 merges six constructor cases into one `case a | b | c …` block
   selected by `first`. Tactic combinators (`<;>`, `first`, `all_goals`) *increase* in cslib while decreasing overall.
3. **Using the library instead of re-proving it (both axes).** 12 drops a whole manual induction because
   `updatedStatesComm`, `UpdateStatesUpdated` and `updatedStatesDefMonotone` already exist: 47 → 11 lines, −94% HB.
4. **Terms instead of tactic scaffolding (length).** Single-use `have`s become arguments; `⟨_, _⟩`, dot-constructors
   (`.choiceL`, `.bisim`), `fun _ h => …` and `absurd … (by simp [..])` replace `constructor`/`exists`/`apply` chains.
5. **Minimal simp sets and deleted no-op steps (both).** Unused simp arguments and tactics that do not change the goal
   are deleted (03: 94 → 84 tokens and 960 → 601 HB from simp-argument deletion alone, with zero model calls).
6. **Comments and layout** do not count toward the length metric (146 comment markers removed, no length effect).

Why the headroom is so large: putnambench originals are RL-prover outputs, rewarded only for correctness, so they carry
redundant steps and heavy automation — exactly the redundancy ProofOptimizer and Lean Refactor report. Research-library
originals are written for robustness and readability (explicit intermediate facts, symmetric cases spelled out), which
leaves length headroom but less than competition proofs; Lean Refactor reports 20–34% length reduction on research
repositories, while our multi-round, kernel-gated search reaches 45–76% on strata/physlib/cslib/arklib (07 is the
exception at 22%, where the gain is almost entirely heartbeats).

## 2. Literature

| paper | status | method | result relevant to us |
|---|---|---|---|
| Lu, Kong, Stehling, Yang, Wang, Sun, Chen. *Lean Refactor: Multi-Objective Controllable Proof Optimization via Agentic Strategy Search*, arXiv 2605.20244 (2026) — the Arena's own paper | ✅ verified (arXiv page) | frozen LLM + retrieval from 9,237 refactoring strategies distilled from 200K long→short pairs, each with when-to-apply, guide, example, profiled compile-time reduction and version-compatibility set; planner → refactorer → compiler-guided debugger | length and compile time weakly correlated (length-only retrieval *doubled* compile time on miniF2F); compile-aware rerank −30% to −60% compile time; version-filtered retrieval improves transfer; planner ablation −8.7 points |
| Gu, Piotrowski, Gloeckle, Yang, Markosyan. *ProofOptimizer*, arXiv 2510.15700 (2025) | ✅ verified (arXiv page) | symbolic pass with Lean's `linter.unusedTactic`; iterative shortening: sample k, keep shortest valid, repeat; expert iteration + RL | −87.9% miniF2F, −57.2% PutnamBench; repair-with-feedback underperforms resampling (repaired proofs often longer) |
| Ahuja, Avigad, Tetali, Welleck. *ImProver*, ICLR 2025, arXiv 2410.04753 | ✅ verified (arXiv page) | Chain-of-States (goal states between tactics shown to the model), error correction, retrieval, best-of-n, refinement | naive LLM rewriting falls short; symbolic proof-state context is what makes rewriting reliable |
| Ahuja, Rowney, Avigad, Welleck. *ImProver 2*, arXiv 2605.22885 (2026) | ✅ verified (arXiv page) | expert iteration + scaffold exposing formal structure with informal abstractions | 7B model competitive with mid-tier frontier models on structural metrics |
| Li, Tian, Wang. *Compile to Compress*, arXiv 2604.18587 (2026) | ✅ verified (arXiv page) | compiler maps attempts to a compact set of failure modes; localized tree search conditioned on verifier feedback | state of the art on PutnamBench at 8B/32B under equal test-time budget |
| Sledgehammer user guide (Isabelle, TUM) | ⚠️ UNVERIFIED (documentation, not a paper) | fact-set minimization by re-running the prover on subsets; preplay of candidate one-liners | minimal fact sets make reconstruction faster and more robust |
| *Automatic Test-Case Reduction in Proof Assistants: A Case Study in Coq*, arXiv 2202.13823 | ⚠️ UNVERIFIED (search result only) | delta debugging of proof scripts, in-order removal of syntactic units | systematic deletion is effective when forward references are impossible |
| *Towards Automatic Transformations of Coq Proof Scripts*, arXiv 2401.11897 | ⚠️ UNVERIFIED (search result only) | script → single-step atomic tactic scripts | structural transformations as a preprocessing step |
| Mathlib linters: `unusedTactic`, `unnecessarySimpa`, `flexible`, non-terminal `simp` → `simp?` | documentation | tactics that do not change the goal, `simpa` replaceable by `simp`, rigid tactics after flexible ones | verified locally: `unusedTactic` and core `unusedSimpArgs` both report exact positions (with a ready replacement for simp arguments) |

## 3. What this changes in `rma reduce`

1. **Lean-verified deletions first, for free** (ProofOptimizer's symbolic pass, Sledgehammer minimization): run the
   current best once per round with `linter.unusedTactic` (Mathlib corpora) and the core `unusedSimpArgs` lint; every
   reported no-op tactic or unused simp argument becomes a located issue with a deterministic fix, verified by the
   kernel and scored before any model call. Codex's `LocalSearch` extends this with bounded line/argument deletion.
2. **A strategy bank in L** (Lean Refactor): the six mechanisms above, with before/after examples from our own proofs,
   when-to-apply triggers (e.g. `grind`/`aesop`/`simp_all` present → explicit terms; mirrored `case` blocks → `<;>` /
   `first`; single-use `have` → inline; re-proved library facts → library lemma), and measured length/heartbeat effects.
   Retrieval is objective-aware: heartbeat-heavy proofs get the automation-removal strategies first, because length and
   compile cost are only weakly correlated.
3. **Proof states for the model** (ImProver's Chain-of-States): expose goals between tactics for the critic and solver.
4. **Resample before repairing** (ProofOptimizer): keep the one compiler-feedback repair only for high-value issues;
   otherwise spend the budget on fresh candidates from the current best.
5. **Version-aware acceptance** (Lean Refactor): already enforced — every prepared listed version is compiled before
   promotion; strategies that break older toolchains are recorded as failed moves.
