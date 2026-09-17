#!/usr/bin/env bash
# Daily progress email for the TPAMI GTK-theory extension
# (drafts/tpami_gtk_extension/).
#
#   bash scripts/tpami_daily_report.sh              # gather, think, send
#   bash scripts/tpami_daily_report.sh --dry-run    # print the email, send nothing
#   TO=someone@example.com bash scripts/tpami_daily_report.sh
#
# Pattern mirrors agent_competitions/tools/daily_report.sh: a FACTS block
# that is purely measured (page count, theorem count, checklist state,
# running experiment jobs -- never invented), then a short reflection
# written by `claude -p` from those facts plus yesterday's snapshot.
# Pure text, no tables, per user preference.
#
# 2026-09-08: no longer emails on its own. Per user request ("consolidate
# the daily reports, don't send me many emails"), the finished report is
# dropped into the shared daily-digest inbox instead; exactly one combined
# email per day is sent by agent_competitions/tools/daily_digest.py (after
# ARC's 21:00 slot, at 21:45).
#
# drafts/ is gitignored for this project (see .gitignore), so day-to-day
# change detection can't use `git diff` -- state/history.json below is a
# hand-rolled snapshot diff instead.
set -uo pipefail

REPO=/projects/bhov/zzhao18/code/ResearchMathAgent-web
PAPER_DIR="${REPO}/drafts/tpami_gtk_extension"
CLAUDE="${CLAUDE_BIN:-/projects/bhov/zzhao18/software/npm-global/bin/claude}"
INBOX=/projects/bhov/zzhao18/code/agent_competitions/tools/daily_digest_inbox
# The bare `python3` on PATH resolves to the system's ancient 3.6 (no
# capture_output=True, no text=True, no PEP 585 generics) -- use a real
# interpreter directly rather than writing 3.6-compatible code.
PY=/usr/bin/python3.11

TO="${TO:-sjtuytc@gmail.com}"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/tpami_daily}"
SNAPSHOT="${STATE_DIR}/snapshot.json"
HISTORY="${STATE_DIR}/history.md"
mkdir -p "${STATE_DIR}"
touch "${HISTORY}"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

DATE="$(date +%Y-%m-%d)"
WORK="${STATE_DIR}/${DATE}"
mkdir -p "${WORK}"
FACTS="${WORK}/facts.md"
THOUGHTS="${WORK}/thoughts.md"
EMAIL="${WORK}/email.txt"

# =============================================================================
# 1. FACTS -- everything here is measured directly, nothing is written by a model
# =============================================================================
"${PY}" - "${PAPER_DIR}" "${SNAPSHOT}" "${FACTS}" "${DATE}" <<'PYFACTS'
import json, re, subprocess, sys
from datetime import datetime
from pathlib import Path

paper_dir, snapshot_path, out_path, today = (Path(sys.argv[1]), Path(sys.argv[2]),
                                              Path(sys.argv[3]), sys.argv[4])
main_tex = paper_dir / "main.tex"
checklist = paper_dir / "TPAMI_COMPLIANCE_CHECKLIST.md"
main_pdf = paper_dir / "main.pdf"
lines = []
w = lines.append

w(f"# 事实块 {today}（全部由 tpami_daily_report.sh 直接测量）\n")

# ---- page count -------------------------------------------------------
pages = None
try:
    out = subprocess.run(["pdfinfo", str(main_pdf)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30).stdout
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    if m:
        pages = int(m.group(1))
except Exception:
    pass

# ---- theorem-environment counts ---------------------------------------
tex = main_tex.read_text(encoding="utf-8") if main_tex.exists() else ""
def count_env(name):
    return len(re.findall(r"\\begin\{" + name + r"\}", tex))
counts = {
    "theorem": count_env("theorem"),
    "proposition": count_env("proposition"),
    "corollary": count_env("corollary"),
    "lemma": count_env("lemma"),
    "definition_defi": count_env("defi"),
    "assumption": count_env("assumption"),
}
abstract_m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
abstract_words = None
if abstract_m:
    # NOTE: newlines must become spaces, never be deleted -- deleting them
    # merges the last word of one line into the first word of the next
    # (e.g. "under\na" -> "undera") and silently undercounts. This exact bug
    # cost a day of believing the abstract was compliant when it was not
    # (see TPAMI_COMPLIANCE_CHECKLIST.md, 2026-08-26 entry) -- do not
    # reintroduce it here.
    raw = abstract_m.group(1).replace("\\GTK{}", "GTK").replace("\\GTK", "GTK")
    raw = re.sub(r"\\[A-Za-z]+", "", raw)
    raw = raw.replace("{", "").replace("}", "").replace("--", " ")
    raw = re.sub(r"\s+", " ", raw).strip()
    abstract_words = len(raw.split())

# ---- checklist "Remaining" section (last such heading) -----------------
remaining_items = []
if checklist.exists():
    ctext = checklist.read_text(encoding="utf-8")
    headings = [m.start() for m in re.finditer(r"^## .*[Rr]emaining.*$", ctext, re.M)]
    if headings:
        chunk = ctext[headings[-1]:]
        remaining_items = re.findall(r"^- .*$", chunk, re.M)[:12]

# ---- running experiment jobs (SLURM) ------------------------------------
running_jobs = []
try:
    q = subprocess.run(["squeue", "-u", "zzhao18", "-h", "-o", "%i %j %T %M"],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30).stdout.strip()
    if q:
        running_jobs = [l for l in q.splitlines() if "gtk" in l.lower()]
except Exception:
    pass

# ---- file mtimes ---------------------------------------------------------
def mtime_str(p: Path):
    if not p.exists():
        return "missing"
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

w("## 1. 论文当前状态\n")
w(f"- main.pdf 页数：{pages if pages is not None else '未知（pdfinfo 失败）'}（确认过的 TPAMI 硬上限 12 页）")
w(f"- 定理环境计数：theorem={counts['theorem']}, proposition={counts['proposition']}, "
  f"corollary={counts['corollary']}, lemma={counts['lemma']}, definition={counts['definition_defi']}, "
  f"assumption={counts['assumption']}")
w(f"- Abstract 字数：{abstract_words if abstract_words is not None else '未知'}（TPAMI 上限 200 词）")
w(f"- main.tex 最后修改：{mtime_str(main_tex)}")
w(f"- TPAMI_COMPLIANCE_CHECKLIST.md 最后修改：{mtime_str(checklist)}")
w("")

w("## 2. 与昨天的快照对比\n")
prev = {}
if snapshot_path.exists():
    try:
        prev = json.loads(snapshot_path.read_text())
    except Exception:
        prev = {}
cur = {"date": today, "pages": pages, **{f"count_{k}": v for k, v in counts.items()},
       "abstract_words": abstract_words}
if prev and prev.get("date") != today:
    def delta(key, label):
        pv, cv = prev.get(key), cur.get(key)
        if pv is None or cv is None:
            return None
        d = cv - pv
        if d == 0:
            return None
        return f"{label} {'+' if d > 0 else ''}{d}（{pv}→{cv}，昨天 {prev.get('date','?')}）"
    changes = list(filter(None, [
        delta("pages", "页数"),
        delta("count_theorem", "theorem 数"),
        delta("count_proposition", "proposition 数"),
        delta("count_corollary", "corollary 数"),
        delta("abstract_words", "abstract 字数"),
    ]))
    if changes:
        for c in changes:
            w(f"- {c}")
    else:
        w("- 论文的可测量指标（页数/定理计数/abstract 字数）自上次快照以来没有变化。")
elif not prev:
    w("- 这是第一次快照，没有可比较的历史。")
else:
    w("- 今天已经快照过一次，不重复比较。")
w("")
# persist today's snapshot for tomorrow (even on dry-run reads are fine, but only
# write on a real send -- handled by the caller after a successful mail)

w("## 3. 待办清单（TPAMI_COMPLIANCE_CHECKLIST.md 的 Remaining 小节，原文摘录）\n")
if remaining_items:
    for item in remaining_items:
        w(item)
else:
    w("- 未找到 Remaining 小节，或该小节为空。")
w("")

w("## 4. 实验作业状态（SLURM）\n")
if running_jobs:
    w("当前仍在队列中的 gtk 相关作业：")
    for j in running_jobs:
        w(f"  {j}")
else:
    w("当前没有 gtk 相关的 SLURM 作业在跑。")
w("")

out_path.write_text("\n".join(lines), encoding="utf-8")
snapshot_path.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"facts -> {out_path} ({len(lines)} lines); snapshot -> {snapshot_path}")
PYFACTS

# =============================================================================
# 2. THINKING -- a short, grounded reflection, never a source of new numbers
# =============================================================================
PROMPT_FILE="${WORK}/prompt.txt"
{
  echo "你是这篇 TPAMI 期刊延伸稿（GTK 理论，drafts/tpami_gtk_extension/）的研究搭档。"
  echo "下面是今天的事实块（全部由脚本直接测量，可信），以及最近几天的日报历史。"
  echo "请写一段简短的中文进展总结，纯文字，不要表格、不要 markdown 表格。"
  echo
  echo "【硬性要求】"
  echo "1. 不许出现事实块里没有的数字。"
  echo "2. 如果「与昨天的快照对比」一节显示没有变化，直接说清楚今天没有可测量的论文改动，"
  echo "   不要编造进展；如果这已经连续多天没有变化，指出这一点。"
  echo "3. 给出下一步最该做的 1-2 件具体事情，直接引用待办清单里的条目，不要泛泛而谈。"
  echo "4. 如果有 SLURM 作业还在跑，提一句预期会带来什么结果。"
  echo "5. 全文不超过 400 字，宁可简短。"
  echo "6. 如果你需要了解论文最新的定理/实验内容做更具体的判断，可以用 Read 工具查看"
  echo "   ${PAPER_DIR}/main.tex 或 ${PAPER_DIR}/TPAMI_COMPLIANCE_CHECKLIST.md，"
  echo "   但只用来理解上下文，不要在总结里引入事实块之外的具体数字。"
  echo
  echo "=== 事实块 ==="
  cat "${FACTS}"
  echo
  echo "=== 最近几天的日报历史 ==="
  tail -40 "${HISTORY}"
} > "${PROMPT_FILE}"

echo "--- calling claude (timeout 600s) ---"
if timeout 600 "${CLAUDE}" -p "$(cat "${PROMPT_FILE}")" --add-dir "${PAPER_DIR}" > "${THOUGHTS}" 2>"${WORK}/claude.err"; then
  echo "thoughts -> ${THOUGHTS} ($(wc -l < "${THOUGHTS}") lines)"
else
  {
    echo "【自动分析失败】claude -p 调用未成功（见 ${WORK}/claude.err）。"
    echo "本邮件只包含第 1 节的测量结果，反思/建议缺失，请手动补。"
    echo
    tail -5 "${WORK}/claude.err" 2>/dev/null
  } > "${THOUGHTS}"
fi

# =============================================================================
# 3. EMAIL
# =============================================================================
PAGES="$(grep -oP '页数：\K[0-9]+' "${FACTS}" | head -1)"
SUBJ="[TPAMI GTK] ${DATE} 日报"
[ -n "${PAGES:-}" ] && SUBJ="${SUBJ}（${PAGES} 页）"

{
  cat "${FACTS}"
  echo
  echo "---"
  echo
  cat "${THOUGHTS}"
  echo
  echo "---"
  echo "生成：$(hostname) $(date '+%F %T %Z')  |  脚本：scripts/tpami_daily_report.sh"
  echo "原始数据：${WORK}/"
} > "${EMAIL}"

if [ "${DRY_RUN}" = 1 ]; then
  echo "=================== DRY RUN: ${SUBJ} ==================="
  cat "${EMAIL}"
  echo
  echo "(dry run: history.md left untouched)"
  exit 0
fi

mkdir -p "${INBOX}"
{
  echo "SUBJECT: ${SUBJ}"
  echo
  cat "${EMAIL}"
} > "${INBOX}/tpami_${DATE}.txt"
echo "queued for digest: ${SUBJ} -> ${INBOX}/tpami_${DATE}.txt"

{
  echo "## ${DATE}"
  head -20 "${FACTS}"
  echo
} >> "${HISTORY}"
