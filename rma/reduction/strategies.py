"""A seed bank of Lean refactoring strategies for the Literature component L.

Following Lean Refactor (arXiv 2605.20244), a strategy is not a citation but an
executable pattern: when it applies, how to apply it, a before/after example and
its measured effect. These six are distilled from our own original→best pairs on
the Arena benchmark (documents/development/rma-reduce-literature.md). Retrieval
is objective-aware because proof length and elaboration cost are only weakly
correlated: a proof whose remaining score is mostly in heartbeats is steered to
automation-removal first, a length-bound proof to structural compression.
"""
from __future__ import annotations
import re

BANK = (
    {'id': 'explicit-over-shotgun', 'objective': 'heartbeats',
     'title': 'Replace shotgun automation by explicit reasoning',
     'when': r'\b(aesop|grind|simp_all|tauto|nlinarith|polyrith|norm_num|decide|positivity|continuity|fun_prop)\b',
     'guide': 'For each heavy call, identify the goal it closes and replace it by the lemma application, constructor '
              'term, or a narrow closer (`omega`, `linarith`, `simp only [..]`) for exactly that goal. Keep automation '
              'only where it is already cheap.',
     'example': '07 Fsub.progress: one `grind` per constructor case became explicit constructor terms such as '
                '`⟨_, .appᵣ v h⟩`; heartbeats 59,830 → 1,800 for a 22% length reduction.'},
    {'id': 'merge-symmetric-branches', 'objective': 'length',
     'title': 'Merge symmetric or parallel branches',
     'when': r'(case left|case right|\|\s*inl\b|\|\s*inr\b|\bconstructor\b|\bcases\b|\binduction\b)',
     'guide': 'When two or more branches repeat the same moves up to a lemma name, split once and share the moves '
              'with `<;>`, `all_goals`, `first | a | b`, or multi-alternative `case a | b =>` patterns.',
     'example': '11 CCS.bisimilarity_congr_choice: separate left/right halves became `refine ⟨?_, ?_⟩ <;>` with '
                '`first | have g := hb.follow_fst t | have g := hb.follow_snd t`; 251 → 138 tokens, 3,858 → 470 HB.'},
    {'id': 'library-lemma', 'objective': 'both',
     'title': 'Use an existing library lemma instead of re-proving it',
     'when': r'(\binduction\b|\bhave\b[^\n]*:=\s*by\b|\bcalc\b)',
     'guide': 'Before re-deriving an intermediate fact or running an induction, look for a lemma in the pinned '
              'sources that states it (the literature records list what exists); apply it directly.',
     'example': '12 Core.InitsUpdatesComm: a manual induction became `updatedStatesComm` + `UpdateStatesUpdated` + '
                '`updatedStatesDefMonotone`; 47 → 11 lines, heartbeats 4,251 → 236.'},
    {'id': 'terms-over-scaffolding', 'objective': 'length',
     'title': 'Terms instead of tactic scaffolding',
     'when': r"(\bconstructor\b|\bexists\b|\bapply And\.intro\b|\brefine'|\bexact\s+⟨|\bhave\b)",
     'guide': 'Inline single-use `have`s as arguments; build conjunctions and existentials with `⟨_, _⟩`; use '
              'dot-constructors (`.inl`, `.choiceL`) and `fun _ h => …` instead of intro/constructor/apply chains.',
     'example': '11: `constructor; apply Tr.choiceL htr2; …` became `⟨.choiceL h.1, .bisim h.2⟩`.'},
    {'id': 'restructure-into-key-lemmas', 'objective': 'both',
     'title': 'Restructure around the few facts the argument needs',
     'when': r'(h₁[₀-₉]|h_[a-z]+[^\n]*\n[^\n]*h_[a-z]+|by_contra)',
     'guide': 'Machine-generated proofs nest dozens of auxiliary facts. State the two or three real lemmas of the '
              'argument once and derive the goal from them; delete everything else.',
     'example': '15 putnam_1964_b2: 289 lines of nested `have h₁ … h₁₁` became three lemmas plus `Finset.card_bij\'` '
                'and `omega`; heartbeats 134,499 → 2,497.'},
    {'id': 'minimal-simp', 'objective': 'both',
     'title': 'Minimal simp sets and no-op deletion',
     'when': r'\b(simp|simp_all|simpa|dsimp)\b',
     'guide': 'Delete simp arguments and tactics Lean reports as unused; where a broad `simp` is heavy, try the '
              '`simp only [...]` it actually needs; remove steps that do not change the goal.',
     'example': '03: deleting unused simp arguments and an empty simp stage took 94 → 84 tokens, 960 → 601 HB with no '
                'model call.'},
)


def retrieve(body: str, *, length_headroom: float, heartbeat_headroom: float, k: int = 3) -> list[dict]:
    """Strategies whose trigger occurs in the proof, the axis with more remaining points first."""
    prefer = 'heartbeats' if heartbeat_headroom > length_headroom else 'length'
    scored = []
    for strategy in BANK:
        hits = len(re.findall(strategy['when'], body))
        if hits:
            bonus = 2 if strategy['objective'] == prefer else 1 if strategy['objective'] == 'both' else 0
            scored.append((bonus, hits, strategy))
    scored.sort(key=lambda row: (-row[0], -row[1]))
    return [strategy for _, _, strategy in scored[:k]]


def render(strategy: dict) -> str:
    return (f"Strategy [{strategy['id']}] {strategy['title']} (targets {strategy['objective']}).\n"
            f"When: the proof uses patterns like /{strategy['when']}/.\nHow: {strategy['guide']}\n"
            f"Measured on this benchmark: {strategy['example']}")
