"""Chinese (简体中文) rendering of the per-problem context report.

The English report is assembled by ``context_report._build_problem_latex_body``
as a ``report``-class LaTeX book (chapters: statement, evaluation, best proof,
concepts, meetings, issues, insights). This module turns that same body into a
Chinese report by:

  1. keeping every LaTeX command, math environment, label/ref and verbatim proof
     block **unchanged**, and
  2. translating only the natural-language prose + section/chapter titles into
     Chinese, then
  3. wrapping the result in a ctex/fandol preamble (compiled with tectonic =
     XeTeX, never pdflatex — pdflatex renders CJK as ????).

Translation runs on the Claude Pro/Max subscription via the local ``claude`` CLI
(``complete_via_cli``); no API key, no Vertex. It is only invoked when the
English source changed (the caller hash-caches), so repeated push-forwards that
don't touch a problem are free.
"""
from __future__ import annotations

import re

# Static structural preamble (report class + ctex + Chinese chapter/theorem
# localization + the macros the proof bodies rely on). Problem-specific bits
# (\title/\date and any exotic \operatorname stubs) are added dynamically or by
# context_report's _safety_block, so they are intentionally absent here.
CN_PREAMBLE_STATIC = r"""\documentclass[12pt,oneside]{report}

\usepackage[fontset=fandol]{ctex}
\usepackage{microtype}
%% localize the chapter heading word to Chinese ("第 N 章")
\renewcommand{\chaptername}{第}
\makeatletter
\renewcommand{\@makechapterhead}[1]{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \huge\bfseries \@chapapp\thechapter 章\par\nobreak
      \vskip 20\p@
    \fi
    \interlinepenalty\@M
    \Huge \bfseries #1\par\nobreak
    \vskip 40\p@
  }}
\renewcommand{\@makeschapterhead}[1]{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright \normalfont
    \interlinepenalty\@M
    \Huge \bfseries #1\par\nobreak
    \vskip 40\p@
  }}
\renewcommand{\chaptermark}[1]{\markboth{第\thechapter 章\quad #1}{}}
\makeatother
\usepackage[margin=1.15in,top=1.3in,bottom=1.3in]{geometry}
\usepackage{amsmath,amsthm,amssymb,mathtools}
\usepackage{booktabs}
\usepackage[shortlabels]{enumitem}
\usepackage[dvipsnames]{xcolor}
\usepackage[hidelinks,pdfusetitle]{hyperref}
\usepackage{parskip}
\usepackage{fancyhdr}

%% score-bar glyphs for the evaluation chapter
\newcommand{\scorefull}{\textcolor{black}{\rule{7pt}{7pt}}}
\newcommand{\scoreempty}{\framebox[9pt]{\rule{0pt}{7pt}\hspace{2pt}}}

%% theorem environments — proof bodies use these (Chinese headings)
\newtheorem{theorem}{定理}[chapter]
\newtheorem{lemma}[theorem]{引理}
\newtheorem{proposition}[theorem]{命题}
\newtheorem{corollary}[theorem]{推论}
\newtheorem{claim}[theorem]{断言}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{定义}
\newtheorem{example}[theorem]{例}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{注}
\newtheorem*{theorem*}{定理}
\newtheorem*{lemma*}{引理}
\newtheorem*{corollary*}{推论}
\newtheorem*{remark*}{注}
\renewcommand{\proofname}{证明}

%% common math abbreviations
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\F}{\mathbb{F}}
\newcommand{\eps}{\varepsilon}
\DeclareMathOperator*{\argmin}{arg\,min}

%% running headers
\pagestyle{fancy}
\fancyhf{}
\fancyhead[R]{\small\nouppercase{\leftmark}}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}

%% Citation fallback: show [key] in gray when no .bib file is loaded.
\AtBeginDocument{%
  \renewcommand{\cite}[2][]{[\textcolor{gray}{\mbox{#2}}]}%
  \providecommand{\citep}[2][]{[\textcolor{gray}{\mbox{#2}}]}%
  \providecommand{\citet}[2][]{[\textcolor{gray}{\textit{\mbox{#2}}}]}%
  \providecommand{\citealt}[2][]{[\textcolor{gray}{\mbox{#2}}]}%
  \providecommand{\citealp}[2][]{[\textcolor{gray}{\mbox{#2}}]}%
  \providecommand{\citenum}[1]{\textcolor{gray}{\mbox{#1}}}%
}"""


_SYSTEM = (
    "You are a precise mathematical translator. You translate LaTeX documents "
    "from English to Simplified Chinese (简体中文) for a research math audience. "
    "ABSOLUTE RULES:\n"
    "1. Output ONLY the translated LaTeX. No preamble, no ```` ``` ```` fences, "
    "no commentary, no \\documentclass, no \\begin{document}.\n"
    "2. Preserve EVERY LaTeX command, environment, argument, label, \\ref, "
    "\\cite, and ALL mathematics (inline $...$, display \\[...\\], align, "
    "equation, etc.) EXACTLY, byte-for-byte. Never translate inside math mode.\n"
    "3. Translate ONLY human-readable prose and section/chapter/title TEXT "
    "(the argument of \\chapter, \\section, \\textbf, item labels, table cells, "
    "captions). Keep environment NAMES (theorem, lemma, proof, itemize) as-is — "
    "the preamble already renders their headings in Chinese.\n"
    "4. Keep verbatim proof/statement LaTeX blocks structurally identical; only "
    "translate embedded prose sentences, never symbols or identifiers.\n"
    "5. Use standard Chinese mathematical terminology. Keep the document "
    "compiling: balanced braces, no stray characters."
)


def _translate(text: str, model: str | None, timeout: int = 900) -> str | None:
    from .claude_code import complete_via_cli
    prompt = (
        "Translate the following LaTeX body content to Simplified Chinese, "
        "following every rule. Return only the translated LaTeX body:\n\n" + text
    )
    out = complete_via_cli(prompt, system=_SYSTEM, model=(model or "claude-opus-4-8"),
                           timeout=timeout)
    if not out:
        return None
    # Strip any accidental code fences the model may add.
    out = re.sub(r"^\s*```(?:latex|tex)?\s*", "", out)
    out = re.sub(r"\s*```\s*$", "", out)
    return out.strip() or None


def translate_body(body_en: str, model: str | None = None) -> str | None:
    """Translate the report body (between \\begin{document}/\\end{document})."""
    return _translate(body_en, model)


def translate_title(title_en: str, model: str | None = None) -> str:
    """Translate the short problem title; fall back to the English on failure."""
    t = (title_en or "").strip()
    if not t:
        return t
    out = _translate(t, model, timeout=120)
    return (out or t).splitlines()[0].strip() if out else t


def build_cn_preamble(title_cn: str, meta_line: str, ts_utc: str,
                      escape=lambda s: s) -> str:
    """Static ctex preamble + dynamic \\title/\\author/\\date (Chinese)."""
    lines = [CN_PREAMBLE_STATIC]
    lines.append(
        rf"\title{{\Large\bfseries {escape(title_cn)}\\"
        rf"\large RMA 上下文报告}}"
    )
    if meta_line:
        lines.append(rf"\author{{{meta_line}}}")
    lines.append(rf"\date{{生成时间 {escape(ts_utc)} UTC}}")
    return "\n".join(lines)
