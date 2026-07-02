#!/usr/bin/env python3
"""Normalize malformed math in a LaTeX file.

Machine-generated report prose (esp. meeting action-plan bullets) often contains
math where Greek letters and big operators lost their leading backslash, e.g.
``$c_lambda$``, ``$V_lambda$``, ``$sum_b$``, ``$GL_infinity$``.  Under XeTeX these
render as literal letters ("c_lambda", "V lambda") instead of $c_\\lambda$, etc.

This tool rewrites bare macro-words back to real control sequences, operating
ONLY inside math regions ($...$, $$...$$, \\(..\\), \\[..\\], and the standard
math environments), so ordinary text (including Chinese, pinyin like "mu",
words like "sum") is never touched.  It is idempotent: an already-correct
``\\lambda`` is left alone because the word is preceded by a backslash.

Usage:
    python scripts/normalize_tex_math.py FILE.tex [--check] [--inplace]

    (default)   print a unified diff of proposed changes, do not write
    --inplace   apply changes in place (a .bak copy is written first)
    --check     exit non-zero if any change would be made (CI guard)
"""
import argparse
import difflib
import re
import sys
from pathlib import Path

# Words that, when they appear as a standalone identifier inside math, are
# almost certainly a control sequence that lost its backslash.  Map each to the
# canonical control sequence (usually "\\" + word, but a few are spelled out).
_GREEK = [
    "varepsilon", "varphi", "vartheta", "alpha", "beta", "gamma", "delta",
    "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu", "nu",
    "xi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Upsilon",
    "Phi", "Psi", "Omega",
]
_OPS = [
    "sum", "prod", "coprod", "bigcup", "bigcap", "bigoplus", "bigotimes",
    "bigsqcup", "nabla", "partial", "infty", "langle", "rangle", "otimes",
    "oplus", "cdot", "ldots", "cdots", "vdots", "ddots", "times", "sqrt",
    "ell",
]
# words spelled differently from their macro
_SPECIAL = {"infinity": r"\infty"}

# build (compiled-regex, replacement) list.  Order: specials first.
_RULES = []
for _w, _r in _SPECIAL.items():
    _RULES.append((re.compile(r"(?<![\\A-Za-z])" + _w + r"(?![A-Za-z])"), _r))
for _w in _GREEK + _OPS:
    # (?<![\\A-Za-z]) => not already a control sequence and not glued to a
    # longer identifier; (?![A-Za-z]) => the word is complete.
    _RULES.append((re.compile(r"(?<![\\A-Za-z])" + _w + r"(?![A-Za-z])"), "\\" + _w))

# regex that captures math regions.  $$...$$ must precede $...$ .
# `(?<!\\)` on the opening delimiter avoids matching an *escaped* \$ (which is a
# literal dollar sign in text, not a math switch).  The inline body consumes any
# escaped char via `\\.` so an internal \$ never ends the region early.
_MATH = re.compile(
    r"(?<!\\)\$\$.*?(?<!\\)\$\$"                     # display $$...$$
    r"|(?<!\\)\$(?:\\.|[^$\\])*\$"                   # inline $...$
    r"|\\\[.*?\\\]"                                  # \[ ... \]
    r"|\\\(.*?\\\)"                                  # \( ... \)
    r"|\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}"
    r".*?\\end\{\1\}",                               # math environments
    re.DOTALL,
)

# Control sequences whose brace/bracket argument is a *name* or literal text,
# never math to be normalized (labels, refs, citations, environment names,
# font/upright wrappers).  Their contents are masked out before substitution so
# e.g. \label{eq:phi-...} is never rewritten to \label{eq:\phi-...}.
_PROTECT = re.compile(
    r"\\(?:label|ref|eqref|cref|Cref|autoref|pageref|nameref"
    r"|cite[a-zA-Z]*|begin|end|text|mbox|hbox|operatorname"
    r"|mathrm|mathbb|mathcal|mathfrak|mathsf|mathbf|mathit"
    r"|textbf|textit|texttt|emph)\*?(?:\[[^\]]*\])?\{[^{}]*\}"
)


def _fix_math(region):
    # mask protected commands so their arguments are left untouched
    shields = []

    def _stash(m):
        shields.append(m.group(0))
        return "\x00%d\x00" % (len(shields) - 1)

    region = _PROTECT.sub(_stash, region)
    for pat, repl in _RULES:
        # function replacement inserts `repl` verbatim (no template escaping)
        region = pat.sub(lambda _m, r=repl: r, region)
    for i, original in enumerate(shields):
        region = region.replace("\x00%d\x00" % i, original)
    return region


def normalize(text: str) -> str:
    return _MATH.sub(lambda m: _fix_math(m.group(0)), text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("--inplace", action="store_true", help="apply changes in place")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if changes would be made (make no writes)")
    args = ap.parse_args()

    original = args.file.read_text(encoding="utf-8")
    fixed = normalize(original)

    if fixed == original:
        print(f"{args.file}: no malformed math found.")
        return 0

    diff = difflib.unified_diff(
        original.splitlines(keepends=True), fixed.splitlines(keepends=True),
        fromfile=str(args.file), tofile=str(args.file) + " (normalized)",
    )
    changed = [ln for ln in diff]
    n = sum(1 for ln in changed if ln.startswith("+") and not ln.startswith("+++"))

    if args.check:
        sys.stdout.writelines(changed)
        print(f"\n{args.file}: {n} line(s) would change.", file=sys.stderr)
        return 1

    if args.inplace:
        args.file.with_suffix(args.file.suffix + ".bak").write_text(original, encoding="utf-8")
        args.file.write_text(fixed, encoding="utf-8")
        print(f"{args.file}: normalized {n} line(s) (backup at {args.file}.bak).")
    else:
        sys.stdout.writelines(changed)
        print(f"\n{args.file}: {n} line(s) would change (dry run; use --inplace).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
