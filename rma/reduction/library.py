"""Whole-declaration retrieval limited to the target's preceding source text.

Never scan RMA solution archives or later theorems. This is opt-in via --local-context;
the selected excerpts are persisted for review before entering bounded RMA context.
"""
import re
from pathlib import Path


def retrieve(entry: dict, proof: str, repo: Path, max_chars=10000) -> str:
    if entry['source']=='putnambench':return ''
    path=(repo/entry['file_path']).resolve()
    if not path.is_relative_to(repo.resolve()):raise ValueError('Source path escapes repo')
    text=path.read_text();pos=text.find(entry['src'])
    if pos<0:raise ValueError('Cannot establish preceding-source boundary')
    prefix=text[:pos]
    matches=list(re.finditer(r'(?m)^(?:private |protected |noncomputable |partial )?'
        r'(?:def|abbrev|inductive|theorem|lemma)\s+([\w.\u0080-\uffff]+)',prefix))
    tokens=set(re.findall(r'[\w\u0080-\uffff]+',proof))
    output=[];size=0
    for i,m in enumerate(matches):
        name=m[1].rsplit('.',1)[-1]
        if name not in tokens:continue
        end=matches[i+1].start() if i+1<len(matches) else len(prefix)
        block=prefix[m.start():end].strip()
        # Avoid dragging comments/namespaces or other declarations along with a fact.
        block=re.split(r'(?m)^/[-*]|^namespace |^end\b',block,1)[0].strip()
        if size+len(block)>max_chars:continue
        output.append(block);size+=len(block)
    return '\n\n'.join(output)
