"""Option parser for prompt text.

Returns {"opts": {letter: text}, "style": "line" | "inline", "issues": [...]} or None.
Handles line-initial labels ("A. x", "(A) x", "A) x", "A: x"), inline lists
("A. x B. y ..."), empty options (image options) and duplicated or skipped labels.
Scoring a released item does not need it: the item already lists its options.
"""
import re

LET = "ABCDEFGHIJ"
_LINE = re.compile(r"^\s*(?:\(([A-J])\)|([A-J])\s*[\.\):：])\s*(.*)$")
_INLINE = re.compile(r"(?:(?<=\s)|^)\(?([A-J])(?:\)|[\.:：])\s+")
_TAIL = re.compile(r"\s*(?:Your answer is|Answer with|Answer the question|Please answer|Answer:)\b.*$", re.I)


def _clean(t):
    t = _TAIL.sub("", str(t)).strip()
    return re.sub(r"^[\s\.:;,]+|[\s]+$", "", t)


def _strip_own_labels(opts):
    """If every option text starts with its own label ("A: Same number ..."), drop that label."""
    pat = {l: re.compile(r"^\(?" + l + r"\s*[\)\.:：]\s+") for l in opts}
    if len(opts) >= 2 and all(v and pat[l].match(v) for l, v in opts.items()):
        return {l: pat[l].sub("", v).strip() for l, v in opts.items()}, True
    return opts, False


def _seq_ok(ls):
    return len(ls) >= 2 and ls[0] == "A" and all(LET.index(b) == LET.index(a) + 1 for a, b in zip(ls, ls[1:]))


def parse_options(prompt: str):
    p = str(prompt or "").replace("\r", "")
    lines = p.split("\n")
    # 1) Line-initial options: the last run that starts at A (worked examples come before the real options).
    marks = []
    for i, ln in enumerate(lines):
        m = _LINE.match(ln)
        if m:
            marks.append((i, m.group(1) or m.group(2), m.group(3)))
    best = None
    for s in range(len(marks)):
        if marks[s][1] != "A":
            continue
        blk = [marks[s]]
        for t in range(s + 1, len(marks)):
            if LET.index(marks[t][1]) == LET.index(blk[-1][1]) + 1:
                blk.append(marks[t])
            elif marks[t][1] == blk[-1][1]:
                blk.append(marks[t])  # duplicated label, reported as an issue
            elif len(blk) >= 2 and blk[-1][1] == blk[-2][1] and LET.index(marks[t][1]) == LET.index(blk[-1][1]) + 2:
                blk.append(marks[t])  # A, B, B, D
            else:
                break
        if len(blk) >= 2:
            best = blk
    issues = []
    if best:
        opts = {}
        ls = [b[1] for b in best]
        if len(set(ls)) != len(ls):
            issues.append("duplicate_label")
        for k, (i, l, t) in enumerate(best):
            txt = _clean(t)
            if l in opts:  # a duplicated label keeps its first text
                continue
            opts[l] = txt
        opts, st = _strip_own_labels(opts)
        if st:
            issues.append("own_label_prefix_stripped")
        if any(not v for v in opts.values()):
            issues.append("empty_option_text")
        return {"opts": opts, "style": "line", "issues": issues}
    # 2) Inline options on one line: "A. x B. y C. z" (last matching line).
    for ln in reversed(lines):
        ms = list(_INLINE.finditer(ln))
        ls = [m.group(1) for m in ms]
        if len(ms) >= 2 and _seq_ok(ls):
            opts = {}
            for k, m in enumerate(ms):
                end = ms[k + 1].start() if k + 1 < len(ms) else len(ln)
                opts[m.group(1)] = _clean(ln[m.end():end]).rstrip(".").strip()
            opts, st = _strip_own_labels(opts)
            if st:
                issues.append("own_label_prefix_stripped")
            if any(not v for v in opts.values()):
                issues.append("empty_option_text")
            return {"opts": opts, "style": "inline", "issues": issues}
    return None
