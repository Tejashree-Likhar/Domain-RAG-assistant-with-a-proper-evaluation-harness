"""Load reStructuredText docs, split by heading, and strip Sphinx markup."""
from __future__ import annotations
import re, pathlib
from dataclasses import dataclass

ADORN = set("=-~^*+#\"'`:._")
DROP_DIRECTIVES = {"currentmodule", "include", "toctree", "image", "figure", "plot", "literalinclude",
                   "only", "module", "autosummary", "autoclass", "autofunction", "raw", "testsetup",
                   "doctest", "topic-end", "centered", "hlist", "code-block", "prompt", "sphinx_gallery"}
KEEP_WITH_LABEL = {"note", "warning", "versionadded", "versionchanged", "deprecated", "tip",
                   "important", "dropdown", "topic", "rubric", "seealso"}


@dataclass
class Section:
    doc_id: str
    doc_title: str
    headings: list  # path of headings, outermost first
    text: str


def _is_adorn(line: str) -> bool:
    s = line.rstrip()
    return len(s) >= 3 and len(set(s)) == 1 and s[0] in ADORN


def _clean_inline(s: str) -> str:
    def role(m):
        target = m.group(1).strip()
        mm = re.match(r"(.+?)\s*<([^>]+)>$", target, re.S)
        if mm:
            return mm.group(1).strip()
        if target[:1] in "~!":
            target = target[1:].split(".")[-1]
        return target
    s = re.sub(r":[\w:+-]+:`([^`]+)`", role, s)
    s = re.sub(r"``([^`]+)``", r"`\1`", s)
    s = re.sub(r"`([^`<]+?)\s*<[^>]+>`_{1,2}", r"\1", s)   # hyperlinks
    s = re.sub(r"\|([\w-]+)\|", "", s)                      # substitutions
    return s


def clean_body(lines: list[str]) -> str:
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)\.\. ([\w-]+)::\s*(.*)$", line)
        if m:
            indent, name, arg = m.groups()
            i += 1
            while i < len(lines) and re.match(r"^\s+:[\w-]+:", lines[i]):   # option lines
                i += 1
            if name in KEEP_WITH_LABEL:
                label = {"versionadded": "New in version", "versionchanged": "Changed in version"}.get(name, name.capitalize())
                out.append(f"{label}: {arg}".strip() if name != "rubric" else arg)
            elif name == "math":
                if arg:
                    out.append(arg)
            continue
        if re.match(r"^\s*\.\. (_[^:]+:|\|.*|[^\s:].*)?$", line) and line.lstrip().startswith(".."):
            i += 1                       # label / comment / substitution definition
            continue
        out.append(_clean_inline(line.rstrip()))
        i += 1
    text = "\n".join(out)
    text = re.sub(r"::\s*$", ":", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_sections(raw: str, doc_id: str) -> list[Section]:
    lines = raw.splitlines()
    styles: list = []          # order of first appearance -> heading level
    heads = []                 # (line_index_start, level, title, line_index_end)
    i = 0
    while i < len(lines):
        l = lines[i]
        if _is_adorn(l) and i + 2 < len(lines) and lines[i + 2] == l and lines[i + 1].strip() and not lines[i + 1].startswith(" "):
            style, title, end = (l[0], True), lines[i + 1].strip(), i + 3
        elif (l.strip() and not l.startswith((" ", "..")) and not _is_adorn(l)
              and i + 1 < len(lines) and _is_adorn(lines[i + 1]) and len(lines[i + 1].rstrip()) >= len(l.rstrip())):
            style, title, end = (lines[i + 1][0], False), l.strip(), i + 2
        else:
            i += 1
            continue
        if style not in styles:
            styles.append(style)
        heads.append((i, styles.index(style), _clean_inline(title), end))
        i = end
    sections, path = [], []
    doc_title = heads[0][2] if heads else doc_id
    bounds = [(h[3], heads[n + 1][0] if n + 1 < len(heads) else len(lines)) for n, h in enumerate(heads)]
    # preamble before first heading is ignored (labels, currentmodule)
    for (hstart, level, title, _), (s, e) in zip(heads, bounds):
        path = path[:level] + [title]
        body = clean_body(lines[s:e])
        if len(body.split()) >= 5:
            sections.append(Section(doc_id, doc_title, list(path), body))
    return sections


def load_corpus(raw_dir: str = "data/raw") -> list[Section]:
    out = []
    for p in sorted(pathlib.Path(raw_dir).glob("*.rst")):
        out.extend(parse_sections(p.read_text(encoding="utf-8"), p.stem))
    if not out:
        raise FileNotFoundError(f"no .rst files in {raw_dir}; run scripts/fetch_corpus.py")
    return out
