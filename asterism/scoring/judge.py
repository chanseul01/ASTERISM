"""Rule-based judge.

Two steps: extract(output, public contract) -> typed answer or INVALID, then
compare(typed answer, gold) -> 0 or 1. Extraction never looks at the gold answer.

The final answer is the last positive answer claim in the output: an answer declaration
("Answer:", "the correct/best ... answer is", "the correct option is"), a \\boxed{},
a closed <answer> or <box> tag or a GLM box. Without any of these, a direct-answer item
accepts the whole output only when it is a single line; an item whose prompt asks for
reasoning also accepts a bare answer on the last line, a claim sentence about an option,
or a bare answer on the first line.

Claims under negation, condition or hedging are skipped. If the last claim is empty or
sits inside an unclosed box, no earlier claim is used instead. Text after the claim
(cont) is checked for competing answers ("or A", another bare answer, corrections,
uncertainty), which make the answer invalid for every answer format.
An answer without an explicit marker is rejected rather than accepted when in doubt.
"""
import math
import re

from .extract import unwrap_layers
from .normalize import as_date, fold, light, parse_number, unit_ok as _unit_ok_c

INVALID = object()

_DECL = re.compile(
    r"(?<![a-z])(?:final\s+answer|answer)\s*(?:is\s*[:：]?|[:：])"
    r"|(?<![a-z])(?:most\s+(?:accurate|appropriate|likely|reasonable|plausible)|best|correct|right|final)\s+answer"
    r"(?:\s+(?:based\s+on|given|from|according\s+to|for)\b[^.\n:：]{0,60})?\s+is\s*[:：]?"
    r"|(?<![a-z])(?:correct|right|best|final)\s+(?:option|choice)\s+is\s*[:：]?", re.I)
_NEG_CTX = re.compile(r"\b\w*n't\b|(?<![a-z])(?:wrong|incorrect|false|not|none|neither|nor|never|nobody|no\s+one|nothing|impossible"
                      r"|unlikely|doubt\w*|cannot|unsure|whether|if|might|maybe|possibly|perhaps|"
                      r"eliminat\w*|rul(?:e|ed|ing)\s+out|reject\w*|exclud\w*|except|suppose|assum\w*)(?![a-z])", re.I)
_BOXED = re.compile(r"\\boxed\s*\{")
_TRUE = {"true", "t", "yes", "y", "correct", "right"}
_FALSE = {"false", "f", "no", "n", "incorrect", "wrong"}
_UNANS = {"not answerable", "unanswerable", "cannot be answered", "not answerable from the given pages", "no answer", "none of the pages"}
_TAGS = re.compile(r"<\s*(answer|final answer|final_answer|your answer|your_answer|ans)\s*>\s*(.*?)\s*<\s*/\s*\1\s*>", re.I | re.S)
# Corrections or competing markers at the start of a line ("instead" or "either" inside prose do not count).
_CORR = re.compile(r"(?im)^\s*[*_`]*(?:or|correction|actually|wait|sorry|on\s+second\s+thought|i\s+meant)\b"
                   r"|(?<![a-z])(?:or\s+rather|on\s+second\s+thought|i\s+meant)(?![a-z])", re.I)


def _sentence_prefix(t, pos):
    """Part of the sentence before pos (after the last . ! ? or newline).

    A newline after a line that ends with a comma is not a sentence break ("If X,\\nthe answer is B").
    Periods of abbreviations (e.g. i.e. etc. vs. Fig. No.) and decimal points are not breaks either.
    """
    tt = re.sub(r"(?i)(?<![a-z])(?:e\.g|i\.e|etc|vs|approx|fig|no|eq)\.|(?<=\d)\.(?=\d)", lambda m: m.group(0).replace(".", "\x00"), t[:pos])
    cut = max(tt.rfind(ch) for ch in ".!?\n")
    while cut >= 0 and tt[cut] == "\n" and tt[:cut].rstrip(" \t").endswith(","):
        cut = max(tt.rfind(ch, 0, cut) for ch in ".!?\n")
    return t[cut + 1:pos]


_COND = re.compile(r"(?<![a-z])(?:if|suppos\w*|assum\w*|unless|hypothetical\w*)(?![a-z])", re.I)
# Only a complete "no assumption needed" phrase lifts the condition ("Do not assume ..." keeps it).
_COND_OK = re.compile(r"(?<![a-z])(?:no\s+(?:further\s+|additional\s+|other\s+)?assumptions?\s+(?:are|is|were|was)\s+(?:needed|required|necessary|made)"
                      r"|without\s+(?:any\s+)?(?:further\s+|additional\s+)?assumptions?)(?![a-z])", re.I)
_SUBORD = re.compile(r"^\s*[*_`]*\s*(?:since|because|as|given(?:\s+that)?|although|though|while|whereas|considering(?:\s+that)?|based\s+on)\b", re.I)
_MATRIX_NEG = re.compile(r"(?<![a-z])(?:(?:it\s+is|it's|that\s+is|this\s+is)\s+not\s+(?:true|the\s+case|clear|certain)|(?:i|we)\s+"
                         r"(?:do\s+not|don't|cannot|can't|could\s+not|couldn't)\s+"
                         r"(?:think|say|believe|conclude|determine|tell|be\s+sure)|(?:i|we)\s+(?:doubt|wonder)|not\s+(?:sure|certain)\s+"
                         r"(?:that|whether|if)|unsure\s+(?:that|whether|if))(?![a-z])", re.I)
_EXAMPLE_LEAD = re.compile(r"(?<![a-z])(?:for\s+example|for\s+instance|e\.g\.?|as\s+an\s+example|such\s+as|an?\s+example)(?![a-z])", re.I)
_COORD = re.compile(r"^\s*(?:and|but|or|nor|yet|so)\b", re.I)
_COMP = re.compile(r"^\s*(?:that|whether|if)\b", re.I)
_SKIPPED = []  # claims dropped for negation or condition; cleared by judge() on every call


def _gov_seg(p):
    """The clause that governs a claim. By default the whole sentence (a comma is no boundary), except:
    (1) the part after the last ';'
    (2) after a leading cause or concession clause (since, because, as, given that, although, though,
        while, whereas, considering, based on): the main clause after the last comma, unless a matrix
        negation ("it is not true that", "I don't think", "I wonder") or a non-coordinate negation sits
        between the first and last comma, or the main clause is a complement clause
    (3) otherwise the part from the last ', (and) so/therefore/thus/hence/but'."""
    p = p.split(";")[-1]
    if _SUBORD.match(p) and "," in p:
        head, main = p.rsplit(",", 1)
        mids = head.split(",")[1:]
        if _COMP.match(main) or _MATRIX_NEG.search(",".join(mids)) or any(_NEG_CTX.search(x) and not _COORD.match(x) for x in mids):
            return p
        return main
    m = None
    for m in re.finditer(r",\s*(?:and\s+)?(?:so|therefore|thus|hence|but)(?![a-z])(?!\s*,)", p, re.I):  # ", therefore," is parenthetical
        pass
    if not m:
        return p
    g = p[m.start() + 1:]
    return p if _COMP.match(g.rsplit(",", 1)[-1]) else g


def _neg_text(pre):
    """Negation, condition or example in the text before a claim. A condition or assumption covers
    the whole sentence; negation, hedging and example leads cover the governing clause."""
    p = _COND_OK.sub(" ", pre)
    if _COND.search(p):
        return True
    g = _gov_seg(p)
    return bool(_NEG_CTX.search(g) or _EXAMPLE_LEAD.search(g))


def _negated(t, pos):
    return _neg_text(_sentence_prefix(t, pos))


def _cut_close(x):
    """Cut a payload at an unpaired closing channel marker, as in the declaration payload
    "E<|end_of_box|>" of "<|begin_of_box|>Answer: E<|end_of_box|>". Returns (payload, rest, cut);
    the caller keeps the rest after the answer block, so "or A" in "<answer>Answer: B</answer> or A." is still seen."""
    rest = ""
    cut = False
    for op, cl in (("<|begin_of_box|>", "<|end_of_box|>"), ("<answer>", "</answer>"), ("<box>", "</box>")):
        i = x.find(cl)
        if i >= 0 and op not in x[:i]:
            rest = x[i + len(cl):] + " " + rest
            x = x[:i]
            cut = True
    return x.strip(), rest.strip(), cut


_ANS_INTRO = re.compile(r"(?:\b(?:final\s+)?answer\b|\bcorrect\s+(?:option|choice|answer)\b)[^\n]{0,80}[:：]\s*[*_`]*\s*$", re.I)
_COP_INTRO = re.compile(r"(?<![a-z])(?:is|are|would\s+be|will\s+be|should\s+be|must\s+be)\s*[*_`]*\s*[:：]\s*[*_`]*\s*$", re.I)
# Example and list leads only (a content word such as "illustration" does not count).
_EXAMPLE = re.compile(r"(?<![a-z])(?:for\s+example|for\s+instance|e\.g\.?|such\s+as|as\s+an\s+example|an?\s+example|examples?\s*[:：]|samples?\s*[:：]"
                      r"|distractors?|decoys?|traps?"
                      r"|(?:the\s+)?(?:possible|candidate|other)\s+(?:answers?|options?|choices?|candidates?)|options?\s+(?:are|include|were))(?![a-z])", re.I)
# The lead names the correct answer.
_POS_DESIG = re.compile(r"(?<![a-z])(?:answer|correct|right|best|final|most\s+(?:accurate|appropriate|likely|suitable|reasonable|plausible))(?![a-z])"
                        r"|(?<![a-z])(?:that|which)\s+(?:best\s+|most\s+[a-z]+\s+|correctly\s+|accurately\s+)?"
                        r"(?:matches|represents|corresponds\s+to|satisfies|fits|depicts)(?![a-z])", re.I)


def _answer_intro(line):
    """Whether a line ending with a colon introduces the answer on the next line: an explicit final-answer
    lead, or a copula lead that names the correct answer and carries no negation, condition, example or
    list marker. "I ruled out:", "An example option is:" and "The options are:" do not count."""
    if _neg_text(line) or _EXAMPLE.search(_gov_seg(line)):  # "Since A is not possible, the correct answer is:" counts
        return False
    return bool(_ANS_INTRO.search(line) or (_COP_INTRO.search(line) and _POS_DESIG.search(line)))  # "The distractor is:" does not


_LABM = r"[*_`]*[\(\[\{]?\s*(?P<lab>(?-i:[A-J]))\s*[\)\]\}]?[*_`]*"
_EXPL = r"(?:\s*,?\s*(?:because|since|as|which)\s+(?P<expl>.*?))?"
_ADJ = r"(?:correct|right|best|final|most\s+(?:accurate|appropriate|likely|suitable|reasonable|plausible))"
# Claim grammar for multiple-choice items whose prompt asks for reasoning. The whole sentence must match.
_CLAIM_SENT = [
    re.compile(r"(?:the\s+)?" + _ADJ + r"\s+[a-z][\w\-'’]*(?:\s+[a-z][\w\-'’]*){0,7}\s+(?:is|would\s+be)\s*[:：]?\s*[*_`]*"
               r"(?:(?:option|choice|image|diagram|figure|graph|plot|picture|structure|visualization)\s+)?"
               + _LABM + r"(?![A-Za-z0-9])" + _EXPL, re.I),
    re.compile(r"[*_`]*(?:option|choice)\s+" + _LABM + r"\s+(?:best|most\s+closely|most\s+accurately|correctly|accurately)\s+"
               r"(?:represents|matches|corresponds\s+to|depicts|fits|satisfies)\s+(?P<obj>.*?)", re.I),
    re.compile(r"[*_`]*(?:option|choice)\s+" + _LABM + r"\s+is\s+(?:the\s+)?(?:correct|best|right)(?:\s+(?:answer|option|choice|one|match))?" + _EXPL, re.I),
    re.compile(r"[a-z][\w\s\-'’]{0,60}?\s+(?:is|are)\s+(?:best|most\s+closely|most\s+accurately)\s+(?:represented|depicted|shown|matched)\s+by\s+"
               r"[*_`]*(?:option|choice)\s+" + _LABM, re.I)]
_CLAIM_FULL = [re.compile(r"(?P<lead>.*?)(?:" + rx.pattern + r")", re.I | re.S) for rx in _CLAIM_SENT]
_LEAD_OK = re.compile(r"(?:(?:therefore|thus|hence|so|overall|finally|in\s+conclusion|in\s+summary)\s*,?\s*|[^,]{1,100},\s*)", re.I)


def _lab_mentions(s, labels):
    """Option label mentions such as "option B", "**B**", "(B)" or "B is/matches/...": upper-case labels of the item only."""
    out = []
    for m in re.finditer(r"(?<![A-Za-z])(?i:option|choice|answer|image|figure|diagram|graph|plot)\s*[*_`]*[\(\[\{]?\s*((?-i:[A-J]))\s*[\)\]\}]?[*_`]*(?![A-Za-z0-9])"
                         r"|[*_`]{1,2}\s*[\(\[\{]?([A-J])[\)\]\}]?\s*[*_`]{1,2}|[\(\[\{]([A-J])[\)\]\}]"
                         r"|(?<![A-Za-z'’])([A-J])(?=\s+(?i:is|are|was|were|seems|appears|best|matches|represents|shows|depicts|would|should|must|correctly"
                         r"|accurately|fits|also|could|might|instead|too)\b)"
                         r"|(?<![A-Za-z])(?i:is|was|be|are)\s*[:：]?\s*((?-i:[A-J]))(?=\s*(?:[.!?,;:)]|$)|\s+(?i:because|since|as|which)\b)"  # "... is B."
                         r"|(?<![A-Za-z0-9'’])([A-J])(?=\s*(?:[.!?,;:)]|$)|\s+(?i:or|and)\s)", s):  # "not B.", "choose A.", "B or A"
        x = next(g for g in m.groups() if g)
        if x in labels:
            out.append((m.start(), m.end(), x))
    return out


_NEG_MENTION = (r"(?:(?:option|choice)\s+)?[*_`(\[]*{x}[*_`)\]]*\s+(?:and\s+(?:(?:option|choice)\s+)?[*_`(\[]*[A-J][*_`)\]]*\s+)?"
                r"(?:is|are|was|were)\s+(?:not|incorrect|wrong|impossible|invalid|ruled\s+out|eliminated|excluded)")
_NEG_MENTION_PRE = (r"(?:not|unlike|rather\s+than|instead\s+of|except|ruling\s+out|eliminating|excluding|neither|nor)\s+"
                    r"(?:(?:option|choice)\s+)?[*_`(\[]*{x}(?![A-Za-z0-9])")


def _sent_spans(s):
    tt = re.sub(r"(?i)(?<![a-z])(?:e\.g|i\.e|etc|vs|approx|fig|no|eq)\.|(?<=\d)\.(?=\d)", lambda m: m.group(0).replace(".", "\x00"), s)
    spans, st = [], 0
    for m in re.finditer(r"[.!?]+(?=\s|$)|\n", tt):
        if tt[st:m.end()].strip():
            spans.append((st, m.end()))
        st = m.end()
    if tt[st:].strip():
        spans.append((st, len(s)))
    return spans


def _cot_claim(s, labels):
    """Undeclared conclusion of a multiple-choice item whose prompt asks for reasoning: the label of the
    last sentence that mentions a label, if that whole sentence matches the claim grammar.

    "The correct diagram is B. A is the best match." is invalid because its last label sentence is no
    claim. The lead may be one connective adverb or one clause ending with a comma (no negation, condition
    or example). Other labels in the lead must be negated ("Since A is not possible, ..."). The
    explanation may not hold a condition, and the object of an evaluative verb may not be negated."""
    spans = [(a, b) for a, b in _sent_spans(s) if _lab_mentions(s[a:b], labels)]
    if not spans:
        return None
    a, b = spans[-1]
    body = re.sub(r"^[#>\s]*[*_`]*\s*(?:conclusion|summary|final\s+decision|result)\s*[*_`]*\s*[:：]?\s*[*_`]*\s*", "", s[a:b].strip(), flags=re.I)
    body = re.sub(r"[\s*_`]*[.!]?[\s*_`]*$", "", body)
    for rx in _CLAIM_FULL:
        m = rx.fullmatch(body)
        if m:
            lead = m.group("lead")
            if lead and not _LEAD_OK.fullmatch(lead):
                continue
            if _neg_text(body[:m.start("lab")]):  # also the claim noun phrase ("The correct answer if ... is B")
                continue
            gd = m.groupdict()
            ex = gd.get("expl") or ""
            ob = gd.get("obj") or ""
            if _COND.search(ex + " " + ob) or re.search(r"(?<![a-z])(?:none|no|not|neither|nor|never|nothing)(?![a-z])|n't", ob, re.I):
                continue
            L = m.group("lab")
            others = {x[2] for x in _lab_mentions(body, labels)} - {L}
            if any(not (re.search(_NEG_MENTION.format(x=x), lead, re.I) or re.search(_NEG_MENTION_PRE.format(x=x), lead, re.I)) for x in others):
                return None, "cot_claim_conflict", ""
            if _lab_mentions(ex, labels):
                return None, "cot_claim_conflict", ""
            return L, "cot_claim", s[b:]  # the text after the claim sentence is checked for competing choices
    return None


# GLM box <|begin_of_box|>X<|end_of_box|>; <box> and <answer> only as closed pairs.
_BOXTOK = re.compile(r"<\|begin_of_box\|>(.*?)(<\|end_of_box\|>|$)|<(box|answer)>(.*?)(</\3>)", re.S)


def _boxed_events(t):
    ev = []
    for m in _BOXTOK.finditer(t):
        closed = bool(m.group(2) or m.group(5))
        inner = (m.group(1) if m.group(1) is not None else m.group(4)) or ""  # only tags of the same name pair up
        ev.append((m.start(), inner.strip() if closed else "", "boxed" if closed else "boxed_unclosed", m.end() if closed else len(t)))
    for m in _BOXED.finditer(t):
        i, d = m.end(), 1
        while i < len(t) and d:
            if t[i] == "\\" and i + 1 < len(t):  # \{ and \} do not count ("\left\{ ... \right.")
                i += 2
                continue
            d += {"{": 1, "}": -1}.get(t[i], 0)
            i += 1
        ev.append((m.start(), t[m.end():i - 1].strip() if d == 0 else "", "boxed" if d == 0 else "boxed_unclosed", i if d == 0 else len(t)))
    return ev


_NUMKINDS = ("int", "number", "anyof", "identifier")


def final_block(text, kind=None, pub=None):
    """(payload or None, how, cont). cont is the text after the payload, including what is left of the
    same line outside a box; it is used for the competing-answer check and for multi-line lists."""
    t = str(text or "").replace("\r", "")
    for op, cl in (("<think>", "</think>"), ("<reasoning>", "</reasoning>"), ("◁think▷", "◁/think▷")):
        if op in t:  # the response channel starts after the last closing tag; none, or a reopened one, means no response
            if cl not in t:
                return None, "truncated_reasoning", ""
            t = t.rsplit(cl, 1)[1]
            if op in t:
                return None, "truncated_reasoning", ""
    if "<answer>" in t and "</answer>" not in t:  # an unclosed answer tag is a boundary (a closed one is a box event)
        t = t.replace("<answer>", "\n")
    # A one-line code fence is a wrapper, but only when it fills whole lines.
    t = re.sub(r"(?m)^[ \t]*```[\w+\-]*[ \t]*\n[ \t]*([^\n`]+?)[ \t]*\n[ \t]*```[ \t]*$", r"\1", t)
    ev = []
    for m in _DECL.finditer(t):
        if _negated(t, m.start()):  # "the wrong answer is", "It isn't true that the answer is", "If the answer is"
            _SKIPPED.append(m.start())
            continue
        line_end = t.find("\n", m.end())
        line_end = len(t) if line_end < 0 else line_end
        pay, rest0, cut = _cut_close(t[m.end():line_end])
        end = line_end
        if not pay and not cut:  # "Answer:\nB": the next non-empty line; an empty declaration closed by a channel marker is not filled
            nxt = re.search(r"\S[^\n]*", t[line_end:])
            if nxt:
                p2, r2, _c = _cut_close(nxt.group(0))
                pay = p2
                rest0 = (r2 + " " + rest0).strip()
                end = line_end + nxt.end()
        ev.append((m.start(), pay, "decl", line_end, end, rest0))
    bx = _boxed_events(t)
    for b in bx:
        if _negated(t, b[0]):  # "Do not choose \boxed{B}"
            _SKIPPED.append(b[0])
            continue
        # A box on the same line as a declaration belongs to it ("Answer: \boxed{B}"); a box on a later line is its own event.
        if not any(e[0] <= b[0] < e[3] for e in ev if e[2] == "decl"):
            ev.append((b[0], b[1], b[2], b[3], b[3], ""))
    if ev:
        ev.sort(key=lambda e: e[0])
        pos, pay, how, _le, end, rest0 = ev[-1]
        if any(b[2] == "boxed_unclosed" and b[0] <= pos for b in bx):  # an answer inside or after an unclosed box is cut off
            return None, "boxed_unclosed", ""
        cont = (rest0 + "\n" + t[end:]) if rest0 else t[end:]
        if how in ("decl", "boxed"):  # a box inside a declaration or answer tag is the answer
            inner = _boxed_events(pay)
            if inner:
                if inner[-1][2] != "boxed":
                    return None, "boxed_unclosed", ""
                if _negated(pay, inner[-1][0]):  # "<answer>Do not choose \boxed{B}</answer>"
                    return None, "negated_box", ""
                b0, b1 = inner[-1][0], inner[-1][3]
                rest = (pay[:b0] + " " + pay[b1:]).strip()  # "Answer: \boxed{B} or A.": text outside the box stays in the block
                cont = (rest + "\n" + cont) if rest else cont
                pay = inner[-1][1]
        if how == "boxed_unclosed":
            return None, how, ""
        cont = re.sub(r"<\|(?:begin|end)_of_box\|>|</?answer>|</?box>", " ", cont)  # channel markers are not competing answers
        return (pay if pay else None), how, cont
    s = t.strip()
    if not s:
        return None, "whole", ""
    lines = [x.strip() for x in s.split("\n") if x.strip()]
    _ATOM = r"[*_`(<\[]*(?:[A-Ja-j]|[+\-]?\d+(?:\.\d+)?|yes|no|true|false)[*_`)>\].:]*"
    layout = (pub or {}).get("layout", "cot" if pub is None else "direct")
    if layout == "direct":  # direct prompts ask for "only the final answer": without a marker, only a one-line output is an answer
        return (s, "whole", "") if len(lines) == 1 else (None, "not_one_line", "")
    _intro = len(lines) >= 2 and re.search(r"[:：]\s*[*_`]*\s*$", lines[-2])  # a colon lead must introduce the final answer
    prev_ok = len(lines) >= 2 and (not _intro or _answer_intro(lines[-2]))
    if layout == "cot" and len(lines) >= 2 and prev_ok and re.fullmatch(_ATOM, lines[-1], re.I) and not re.fullmatch(_ATOM, lines[-2], re.I):
        return lines[-1], "cot_final_line", ""  # a bare answer on the last line after the reasoning; "A\nB" is a list, not an answer
    if layout == "cot" and kind == "mcq":
        c = _cot_claim(s, set((pub or {}).get("options") or {}))
        if c:
            return c
    if kind in (None, "mcq", "polar") + _NUMKINDS and len(lines) >= 2 and re.fullmatch(
            r"[*_`(<\[]*(?:[A-Ja-j]|[+\-]?\d+(?:\.\d+)?|yes|no|true|false)[*_`)>\].:]*", lines[0], re.I):
        return lines[0], "lead", "\n".join(lines[1:])  # answer first, explanation after; the explanation is checked as cont
    if len(lines) >= 2 and len(lines[-1]) <= 120 and _intro and _answer_intro(lines[-2]):
        return lines[-1], "tail_block", ""  # the last line, announced by an answer lead on the line before
    return s, "whole", ""


# ---------------------------------------------------------------- extraction per answer format (gold not used)
_DECOR = "<>*_`'\"()[]{}"
_PLACEHOLDER = re.compile(r"\s*\{(?:image|table)#\d+\}\s*")


def atomic_layers(pay, placeholders=False):
    """Candidates for short answers: the balanced unwrapping layers plus every step of removing one
    outer decoration character at a time ("<<A.>" -> "A", "<<3944276" -> "3944276"). Also tried: the
    payload without LaTeX text wrappers (\\text{A}), without a whole approved tag (<answer>...</answer>),
    and, for identifiers and numbers, without copied prompt placeholders such as {image#1}."""
    L = unwrap_layers(pay)
    raw = str(pay).strip()
    extra = [re.sub(r"\\(?:text|textbf|textit|mathrm|mathbf|mathit|mbox)\s*\{([^{}]*)\}", r"\1", raw)]
    if placeholders:
        extra.append(_PLACEHOLDER.sub(" ", raw).strip())  # "9152900{image#1}"
    m = _TAGS.fullmatch(raw)
    if m:
        extra.append(m.group(2))
    for c in extra:
        if c and c != raw and c not in L:
            L.extend(x for x in unwrap_layers(c) if x not in L)
    t = raw
    for _ in range(16):
        t2 = t
        if t2 and t2[0] in _DECOR:
            t2 = t2[1:]
        if t2 and t2[-1] in _DECOR:
            t2 = t2[:-1]
        t2 = t2.strip()
        t3 = t2.rstrip(".;,!").strip()
        for c in (t2, t3):
            if c and c not in L:
                L.append(c)
        if t3 == t:
            break
        t = t3
    return L


# Upper-case labels only: the "a" of "select a red object" is not a label.
_CHOOSE = re.compile(r"(?<![A-Za-z])(?i:choose|chose|choosing|select|selected|pick|picked|go\s+with|answer\s+is|answer\s*:|"
                     r"(?:correct|right|best|final)\s+(?:option|choice|answer)\s+is)\s*[:\-]?\s*(?i:option\s+|choice\s+)?[*_`]*\(?([A-J])\)?[*_`]*(?![A-Za-z])")
_OR = re.compile(r"(?<![A-Za-z])(?i:or)\s+(?i:rather\s+|maybe\s+|perhaps\s+|possibly\s+|probably\s+)?(?i:option\s+|choice\s+)?[*_`]*\(?([A-J])\)?[*_`]*(?![A-Za-z])")
_UNSURE = re.compile(r"(?im)^\s*[*_`]*(?:maybe|perhaps|probably|possibly)\b|(?<![a-z])(?:not\s+certain|could\s+also\s+be|(?:is|are)\s+also\s+possible"
                     r"|also\s+possible)(?![a-z])"
                     r"|(?<![a-z])(?:cannot|can't|unable\s+to)\s+(?:determine|distinguish|tell|decide)|(?<![a-z])(?:not\s+sure|unsure|"
                     r"correction|or\s+rather|on\s+second\s+thought|i'?m\s+(?:wrong|confused|stuck|not\s+sure)|wait,\s*no|no,\s*wait)(?![a-z])", re.I)


def _lone_label(ln):
    """The label if a whole line is one decorated label ("**A**", "(A)", "A.")."""
    for s in atomic_layers(ln):
        m = re.fullmatch(r"(?:or\s+)?(?:option|choice)?\s*\(?\s*([A-J])\s*\)?\s*[:：.]?", s)
        if m:
            return m.group(1)
    return None


def _mcq_conflict(text, chosen, labels):
    """True if the text after a label holds a competing choice ("I choose A", "or A", a lone label line),
    a negation of the chosen label ("reject B", "B is incorrect"), a correction or uncertainty.
    A negated choice ("I did not choose A") does not compete."""
    t = str(text or "")
    if not t.strip():
        return False
    for m in _CHOOSE.finditer(t):
        x = m.group(1).upper()
        if x in labels and x != chosen and not _negated(t, m.start()):  # "Since B is not listed, I choose A" competes
            return True
    for m in _OR.finditer(t):
        if m.group(1) in labels and m.group(1) != chosen:
            return True
    c = re.escape(chosen)
    if re.search(r"(?i:(?<![a-z])(?:not|reject(?:ed|ing)?|rul(?:e|ed|ing)\s+out|eliminat(?:e|ed|ing)|exclud(?:e|ed|ing)|isn't)\s+(?:option\s+|choice\s+)?)"
                 r"[*_`]*\(?" + c + r"\)?[*_`]*(?![A-Za-z])", t):  # only a negation attached to the chosen label
        return True
    if _CORR.search(t):  # "Answer: B\nActually, A."
        return True
    if re.search(r"(?<![A-Za-z])(?i:option\s+|choice\s+)?[*_`]*\(?" + c + r"\)?[*_`]*\s+(?i:is|was|seems|appears)\s+(?i:not\s+(?:correct|right)|incorrect|wrong)", t):
        return True
    if _UNSURE.search(t):
        return True
    for rx in _CLAIM_SENT:  # the claim grammar of extraction also finds competing claims ("Answer: B\nThe correct diagram is A.")
        for mm in rx.finditer(t):
            x = mm.group("lab")
            if x in labels and x != chosen and not _negated(t, mm.start()):
                return True
    for ln in [x.strip() for x in t.split("\n") if x.strip()]:  # lone labels on any line ("B\n**A**")
        if len(ln) <= 20:
            x = _lone_label(ln)
            if x and x in labels and x != chosen:
                return True
    return False


def _dstrip(x):
    """Comparison form of an option text: final sentence punctuation (. 。 ! ? ; ,) is removed only at the
    end of the string or right before a closing bracket or quote. Signs, inequality signs, decimal points,
    operators and spaces stay ("B. -1" != "+1", "x ≤ 1" != "x > 1")."""
    t = light(x)
    for _ in range(4):  # alternate until stable ("...right;。" = "...right;")
        t2 = light(re.sub(r"\s*[.。]+(?=[\]\)\}\"'”’」』]*$)", "", t))
        if t2 == t:
            break
        t = t2
    return t


def x_mcq(pay, pub, cont=""):
    """Rules in this order (the rule order wins over the layer order):
    (1) a lone label; (2) a label followed by text (contradicts and is invalid first if the text equals or
    starts with another option's text); (3) an exact unique option text; (4) several labels = hedge;
    (5) "A, because ..." or "**B** because ...". A label with text (2, 5) and the answer block (cont) are
    checked for competing choices, negation and uncertainty. If a layer that still carries decoration equals
    another option's text ("[B]" being the text of option A), it contradicts the label reading."""
    direct = pub.get("layout", "direct") == "direct"
    opts = pub.get("options") or {}
    L = {k: light(v) for k, v in opts.items()}
    labels = set(opts.keys())
    layers = atomic_layers(pay)
    bare = lambda s: re.fullmatch(r"(?:option|choice)?\s*\(?\s*[A-Za-z]\s*\)?\s*[:：.]?", s.strip(), re.I) is not None
    # Decorated layers equal to an option text (single-letter option texts such as "B" defer to the label reading).
    layer_hits = {k for s in layers if not bare(s) for k, v in L.items() if v and v == light(s) and not re.fullmatch(r"[a-z]", v)}
    bad_cont = lambda l: bool(cont) and _mcq_conflict(cont, l, labels)
    for s in layers:
        m = re.fullmatch(r"(?:option|choice)?\s*\(?\s*([A-Za-z])\s*\)?\s*[:：]?", s, re.I)
        if m and m.group(1).upper() in labels:
            l = m.group(1).upper()
            if m.group(1).islower() and any(v == s.lower() for v in L.values() if v):  # a lower-case letter that is an option text is ambiguous
                return INVALID
            if layer_hits and layer_hits != {l}:  # the unwrapped label and a decorated option text disagree
                return INVALID
            return INVALID if bad_cont(l) else l
    for s in layers:
        m = re.fullmatch(r"[*_`]*\(?([A-Za-z])\)?[*_`]*\s*[\)\.:：]\s*(.+)", s, re.S)
        if m and m.group(1).upper() in labels:
            l, raw_t = m.group(1).upper(), m.group(2)
            t = light(raw_t)
            if any(v == t for k, v in L.items() if v and k != l):  # "B. 1" with A = "1": label and text disagree
                return INVALID
            _ms = lambda x: re.sub(r"[\s$]", "", _dstrip(x))  # ignore only spacing and $ in formulas
            if _ms(raw_t) != _ms(opts.get(l) or "") and any(_ms(v) == _ms(raw_t) for k, v in opts.items() if v and k != l):
                return INVALID  # another option's text up to decoration
            if not (L.get(l) and t.startswith(L[l])) and any(re.match(re.escape(v) + r"(?:[\s,.;:]|$)", t) for k, v in L.items()
                                                              if v and k != l and (re.fullmatch(r"[\d.,\-+%/]+", v) or len(v) >= 3)):
                return INVALID  # "B. 1, because ..." with A = "1": starts with another option's text
            rest = re.sub(re.escape(opts[l]), " ", raw_t, count=1, flags=re.I) if opts.get(l) else raw_t  # leave out its own text ("C > A > B")
            if {x for x in re.findall(r"(?:^|\s)\(?([A-Z])\)?[\.\):](?=\s|$)", rest) if x in labels and x != l}:  # "B. C. E."
                return INVALID
            if L.get(l) and (t == L[l] or _ms(raw_t) == _ms(opts[l])):  # its own text, up to final decoration and formula spacing
                return INVALID if (bad_cont(l) or _COND.search(raw_t[len(opts[l]):] if raw_t.lower().startswith(opts[l].lower()) else "")) else l
            if m.group(1).islower():  # a lower-case label with text ("i.e. ...") counts only with its exact option text
                continue
            if direct:  # direct prompts: after the label only the full option text; try the next unwrapped layer
                continue
            if re.fullmatch(r"[\s\d+\-−*/^().,%=<>≤≥≠!\[\]{}$€£¥₩|:;]+|[\s\d+\-−*/^().,%=<>≤≥≠!\[\]{}$€£¥₩xXyYnN]+", raw_t.strip()):
                continue  # numbers and formulas after a label are an answer, not an explanation
            return INVALID if (_mcq_conflict(rest, l, labels) or bad_cont(l) or _COND.search(rest)) else l  # "B. dog, assuming ..." is invalid
    for s in layers:
        t = light(s)
        hits = [k for k, v in L.items() if v and v == t]
        if len(hits) == 1:
            return INVALID if bad_cont(hits[0]) else hits[0]
        if len(hits) > 1:
            return INVALID
    for s in layers:
        if re.fullmatch(r"\(?[A-Za-z]\)?(?:\s*(?:,|;|/|&|\band\b|\bor\b)\s*\(?[A-Za-z]\)?)+", s, re.I) and \
                len(set(re.findall(r"[A-Za-z]", re.sub(r"\b(?:and|or)\b", "", s, flags=re.I)))) > 1:
            return INVALID  # "{A, B}", "A and C", "A/B": several answers
    for s in layers:
        m = re.fullmatch(r"([A-Za-z])\s*[,;]\s+(\S.*)", s, re.S) or \
            re.fullmatch(r"[\W_]*(?:(?i:option|choice)\s+)?[\W_]*([A-Za-z])[\W_]*\s+((?:because|since|as|which)\b.*)", s, re.S)
        if m and m.group(1).upper() in labels and not direct:  # "B, because ..." only when the prompt asks for reasoning
            l = m.group(1).upper()
            return INVALID if (_mcq_conflict(m.group(2), l, labels) or bad_cont(l) or _COND.search(m.group(2))) else l
    return INVALID


def _close_int(v):
    return abs(v - round(v)) <= 1e-12 * max(1.0, abs(v))  # float noise only (166.00000000000002); 1.0000000005 is not an integer


def x_polar(pay, pub):
    ones = pub.get("polar_as") == "1/0"
    for s in atomic_layers(pay):
        t = light(s)
        if ones and re.fullmatch(r"\+?[01](?:\.\d+)?", t):  # binary tasks: "1", "0", "1.0"; "1%" and "0.9" do not count
            v = float(t)
            if _close_int(v):
                return round(v) == 1
        if t in _TRUE:
            return True
        if t in _FALSE:
            return False
        m = re.match(r"([a-z]+)\s*[,;:\.]\s+(\S.*)", t)  # "yes, because ..." ("not true" does not count)
        if m and m.group(1) in (_TRUE | _FALSE) - {"t", "y", "f", "n"}:
            pol = m.group(1) in _TRUE
            opp = (_FALSE if pol else _TRUE) - {"t", "y", "f", "n"}
            if re.search(r"(?<![a-z])or(?![a-z])", m.group(2)) or any(re.search(r"(?<![a-z])" + w + r"(?![a-z])", m.group(2)) for w in opp) \
                    or _UNSURE.search(m.group(2)) or _COND.search(m.group(2)) \
                    or re.search(r"(?<![a-z])(?:maybe|perhaps|possibly|probably)(?![a-z])", m.group(2)):  # "Yes, if ...", "Yes, maybe."
                return INVALID  # "Yes, or no.", "Yes, although no ...": competing polarity
            return pol
    return INVALID


def _latex_num(t):
    t = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", t)
    t = re.sub(r"\\text\s*\{([^{}]*)\}", r" \1", t)
    t = re.sub(r"\^\s*\{?\s*\\circ\s*\}?|°", "", t)
    t = t.replace("\\%", "%").replace("\\$", "$").replace("\\,", "").replace("\\!", "").replace("\\left", "").replace("\\right", "")
    return t.strip()


_SCALE = {"thousand": 1e3, "thousands": 1e3, "million": 1e6, "millions": 1e6, "billion": 1e9, "billions": 1e9, "bn": 1e9,
          "trillion": 1e12, "trillions": 1e12}
_BADWORDS = {"or", "more", "less", "fewer", "least", "most", "than", "not", "nearly", "almost", "over", "under", "above", "below",
             "approximately", "about", "around", "roughly", "plus", "minus", "at", "up", "and", "to"}
_NUMTOK = r"[+\-]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?(?:e[+\-]?\d+)?"
_NUMWORD = (r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
            r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)(?:[\s\-](?:one|two|three|four|five|six|seven|eight|nine))?")

_WHY = [None]  # why a number was rejected (unit or target mismatch); cleared by judge() on every call


def _unit_ok(ph, pub):
    """A unit phrase after a number must be in the item's unit contract (public.units) or share its canonical unit."""
    ok = _unit_ok_c(ph, (pub or {}).get("units"))
    if not ok:
        _WHY[0] = "unit_mismatch"
    return ok


def _lhs_ok(lhs, pub):
    """The left side of an equation answer must be the expression the question asks for (public.target_expr)."""
    te = (pub or {}).get("target_expr")
    l = re.sub(r"\s+", "", lhs)
    ok = bool(te) and l == te
    if not ok:
        _WHY[0] = "target_mismatch"
    return ok


def _num_parse(s, pub=None):
    """One layer -> (value, raw value, scale word, currency, percent flag) or None; the gold answer is not used.

    Allowed: one number (currency, %, one unit word, fraction, English number words, scientific notation),
    LaTeX clean-up, sign before currency ("-$1,443"), a removed "(± 3.1)" error term, an equation whose left
    side is the target expression of the question, a number followed by an allowed unit phrase, and scale
    words ("2 thousand" -> 2000, raw value kept). A one-sided inequality ("<4", ">4") is invalid."""
    if re.search(r"°|\\circ", str(s)) and not _unit_ok("degree", pub):  # a degree sign needs the unit contract too
        return None
    t = _latex_num(light(s)).replace("−", "-")
    if re.match(r"^[<>≤≥]\s*[\d$€£¥₩+\-]", t) or re.search(r"[\d%]\s*[<>≤≥]$", t):  # "<4" and ">4" are ranges
        return None
    t = re.sub(r"\s*\(\s*(?:±|\+/-|\+-)[^)]*\)", "", t).strip()
    if "=" in t or re.search(r"[≠≤≥]", t):
        if re.search(r"[!<>]=|=[=<>]|[;≠≤≥]|(?<!\d),|,(?!\d{3}(?!\d))", t):  # comparisons and several equations ("x = 3, y = 4")
            return None
        segs = [x.strip() for x in t.split("=")]
        if len(segs) != 2 or not _lhs_ok(segs[0], pub):  # only "target = value" ("g(v) = 1", "c = 2")
            return None
        t = segs[-1]
    t = re.sub(r"^([+\-])\s*([\$€£¥₩])", r"\2\1", t)
    cur = (re.match(r"^([\$€£¥₩])", t) or [None, None])[1]  # currency is read after the right side is fixed ("x = €1200")
    pp = re.search(r"\s*(?:percentage[\s\-]+points?|pp|p\.p\.|pct\.?\s*pts?)\.?$", t)  # percentage points only where the item allows them
    if pp:
        if not (pub or {}).get("pp_ok"):
            _WHY[0] = "unit_mismatch"
            return None
        t = t[:pp.start()].strip()
    pct = t.endswith("%") or t.endswith(" percent")
    cw = re.search(r"\s+(?:us\s+)?(dollars?|usd|euros?|eur|gbp|yen|jpy|yuan|rmb|cny|won|krw)\.?$", t)  # currency words act as symbols
    if cw:
        cur = {"dollar": "$", "dollars": "$", "usd": "$", "euro": "€", "euros": "€", "eur": "€", "gbp": "£", "yen": "¥", "jpy": "¥",
               "yuan": "¥", "rmb": "¥", "cny": "¥", "won": "₩", "krw": "₩"}[cw.group(1)]
        t = t[:cw.start()].strip()
    if cur and pub is not None and "money" in pub and not pub["money"]:  # a currency on an item without a money context
        _WHY[0] = "unit_mismatch"
        return None
    v = parse_number(t)
    if v is not None:
        mu = re.match(r"^[\$€£¥₩]?\s*[+\-]?[\d,]*\.?\d+(?:e[+\-]?\d+)?(?:\s*/\s*\d+)?\s*(.*)$", t) or \
            re.match(r"^[a-z]+(?:[\s\-](?:one|two|three|four|five|six|seven|eight|nine))?\s*(.*)$", t)  # the denominator of "-29/4" is no unit
        unit = (mu.group(1) if mu else "").strip()
        if unit and unit not in ("%", "percent", "per cent") and not _unit_ok(unit, pub):  # "N percent" falls under the % rule
            return None
        return (v, v, None, cur, pct)
    m = re.fullmatch(r"([\$€£¥₩]?\s*(?:" + _NUMTOK + r"|" + _NUMWORD + r"))\s*(thousand|thousands|million|millions|billion|billions|bn|trillion|trillions)\b"
                     r"\.?\s*([a-z][a-z\-']*)?", t)  # at most one unit word after a scale word ("2 thousand people are standing" fails)
    if m:
        r = parse_number(m.group(1))
        if r is not None and not (set((m.group(3) or "").split()) & _BADWORDS) and (not m.group(3) or _unit_ok(m.group(3), pub)):
            return (r * _SCALE[m.group(2)], r, m.group(2).rstrip("s"), cur, pct)
        return None
    # A number followed by a multi-word unit phrase that the question allows ("214 fertilizer warehouses", not "4 people are standing").
    m = re.fullmatch(r"([\$€£¥₩]?\s*" + _NUMTOK + r"\s*%?)\s+([a-z][a-z\s\-']*)", t)
    if m and not _unit_ok(m.group(2), pub):
        m = None
    if m and len(re.findall(r"\d+(?:[.,]\d+)*", t)) == 1 and not (set(m.group(2).split()) & _BADWORDS):
        r = parse_number(m.group(1))
        if r is not None:
            return (r, r, None, cur, pct or "%" in m.group(1))  # the % of "18.5% higher" falls under the % rule
    return None


def _num(pay, pub=None):
    raw = str(pay).strip().strip("*_`'\"")
    if (re.match(r"^[<>≤≥]\s*[\d$€£¥₩+\-]", raw) and not (raw.startswith("<") and raw.endswith(">"))) or \
            re.search(r"\d\s*[<>≤≥]$", raw) and not raw.startswith("<"):
        return INVALID  # one-sided inequality (">4", "4<"); "<4>" and "<<4>>" wrappers are fine
    for s in atomic_layers(pay, placeholders=True):
        try:
            p = _num_parse(s, pub)
        except (OverflowError, ValueError, ZeroDivisionError):
            p = None
        if p is not None and all(math.isfinite(x) for x in p[:2]):
            return p
    return INVALID


def x_int(pay, pub):
    p = _num(pay, pub)
    if p is INVALID or not _close_int(p[0]):  # whether % is allowed is decided in compare (gold.percent_ok)
        return INVALID
    return (int(round(p[0])), p[1], p[2], p[3], p[4])


_TIMEUNIT = re.compile(r"(?i)\b(?:hours?|hrs?|h|seconds?|secs?|s|days?|weeks?|months?|years?)\.?\s*$")


def x_number(pay, pub):
    if pub.get("duration_minutes"):  # answers in minutes: a bare number or minutes only ("120 hours" is not 120 minutes)
        for s in atomic_layers(pay):
            if re.search(r"\d\s*[a-z]", light(s)) and _TIMEUNIT.search(light(s)):
                return INVALID
    return _num(pay, pub)


def x_identifier(pay, pub):
    """Whole-payload grammar only: a bare number (".0" allowed) or "The (special magic X) number (for Y) is: N"."""
    for s in atomic_layers(pay, placeholders=True):
        m = re.fullmatch(r"(?:(?:the\s+)?(?:special\s+magic\s+[\w\-' ]{1,30}\s+)?number(?:\s+for\s+[\w\-' ]{1,30})?\s+is\s*[:：]?\s*)?(\d{4,})(?:\.0+)?",
                         light(s))
        if m:
            return m.group(1)
    return INVALID


def x_text(pay, pub):
    return atomic_layers(pay)  # any candidate may match the accepted answers after folding


def x_anyof_single(pay, pub):
    for s in atomic_layers(pay):
        t = re.sub(r"(?<=\d)\.0+$", "", light(s))  # "1.0" -> "1"
        m = re.fullmatch(r"(?:the\s+)?(?:circle\s*(?:numbered|number|no\.?)?\s*)?#?\s*(\d+)", t)  # "The circle numbered 10." (no sentences)
        if m:
            return m.group(1)
    return INVALID


def x_set(pay, pub):
    alpha = set(pub.get("alphabet") or [])
    for s in atomic_layers(pay):
        parts = [re.sub(r"^image\s*", "", p.strip(), flags=re.I).strip().upper() for p in re.split(r"\s*(?:,|\band\b|&|/)\s*", light(s)) if p.strip()]
        parts = [p for p in parts if p]
        if parts and all(p in alpha for p in parts):
            return sorted(parts)
    return INVALID


def x_ipair(pay, pub):
    """Ordered integer pair "(ROW_INDEX, COLUMN_INDEX)": "(0, 1)", "0, 1", "row 0, column 1"."""
    for s in atomic_layers(pay):
        t = light(s)
        m = re.fullmatch(r"\(?\s*(?:row\s*(?:index)?\s*[:=]?\s*)?(-?\d+)\s*,\s*(?:col(?:umn)?\s*(?:index)?\s*[:=]?\s*)?(-?\d+)\s*\)?", t)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    return INVALID


def _split_top(body):
    """Split a bracketed list body at every top-level comma (commas inside quotes stay). Tokens outside
    quotes are kept as elements, so extra wrong elements are caught."""
    items, cur, q = [], "", None
    for ch in body:
        if q:
            cur += ch
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
            cur += ch
        elif ch == ",":
            items.append(cur)
            cur = ""
        else:
            cur += ch
    items.append(cur)
    out = []
    for x in items:
        x = x.strip()
        if len(x) >= 2 and x[0] == x[-1] and x[0] in "'\"":
            x = x[1:-1]
        if x:
            out.append(x.strip())
    return out


def x_multiset(pay, pub):
    """A bracketed list ("['a', 'b']") uses the first list layer in full. Otherwise the fully unwrapped
    string is split by newlines, else by semicolons, else by commas or "and"."""
    L = unwrap_layers(pay)
    vals = None
    for s in L:
        if s.startswith("[") and s.endswith("]"):
            vals = _split_top(s[1:-1])
            break
    if vals is None:
        s = L[-1] if L else ""
        if "\n" in s:
            vals = [x.strip(" -*•\t") for x in s.split("\n") if x.strip(" -*•\t")]
        elif ";" in s:
            vals = [x.strip() for x in s.split(";") if x.strip()]
        else:
            vals = [x.strip() for x in re.split(r"\s*,\s*|\s+and\s+", s) if x.strip()]
    if not vals and any(x.replace(" ", "") == "[]" for x in L):  # an explicit empty list is a valid (usually wrong) answer
        return []
    return sorted(light(v) for v in vals) if vals else INVALID


def x_tuple(pay, pub):
    for s in atomic_layers(pay):
        m = re.fullmatch(r"\(?\s*(.+?)\s*,\s*(increased|decreased|increase|decrease)\s*\)?", light(s))
        if m:
            return (m.group(1), "increased" if m.group(2).startswith("incr") else "decreased")
    return INVALID


EXTRACT = {"mcq": x_mcq, "polar": x_polar, "int": x_int, "number": x_number, "identifier": x_identifier, "text": x_text,
           "anyof": x_anyof_single, "set": x_set, "multiset": x_multiset, "tuple": x_tuple, "ipair": x_ipair}
_SCALAR = ("polar", "int", "number", "identifier", "anyof", "ipair", "tuple", "set")


def _is_unans(pay):
    return any(light(s) in _UNANS for s in atomic_layers(pay))


# ---------------------------------------------------------------- comparison with the gold answer
def _num_ok(typed, gold, pub):
    v, raw, scale, cur = typed[0], typed[1], typed[2], typed[3]
    pct = typed[4] if len(typed) > 4 else False
    g = gold["value"]
    tol = max(gold.get("atol", 0.0), gold.get("rtol", 0.0) * abs(g)) + 1e-9
    if gold.get("currency") and cur and cur != gold["currency"]:  # "€5" != "$5"
        return False
    if pct and not gold.get("percent_ok", False):  # "4%" != 4 unless the item allows %
        return False
    if abs(v - g) <= tol:
        return True
    return bool(scale) and scale in (pub.get("scale_words") or []) and abs(raw - g) <= tol  # "in millions": the raw value of "714.3 million"


def compare(kind, typed, gold, pub):
    if typed is INVALID:
        return 0.0
    if kind == "mcq":
        return float(typed == gold["answer"])
    if kind == "polar":
        return float(typed == gold["value"])
    if kind == "int":
        return float(_num_ok((float(typed[0]),) + tuple(typed[1:]), gold, pub))
    if kind == "number":
        return float(_num_ok(typed, gold, pub))
    if kind == "identifier":
        return float(typed == str(gold["value"]).strip())
    if kind == "text" and not gold.get("prefix_strip") and not gold.get("ignore_space"):
        acc = [a for a in gold.get("accepted", [])]
        F = {fold(a) for a in acc}
        for s in typed:
            if fold(s) in F:
                return 1.0
            ds = as_date(s)  # dates match only when both sides are complete valid dates
            if ds is not None and any(as_date(a) == ds for a in acc):
                return 1.0
        return 0.0
    if kind == "text":
        acc = [a for a in gold.get("accepted", [])]

        def nz(x):
            y = light(x)
            for p in gold.get("prefix_strip", []):
                y = re.sub("^" + re.escape(p), "", y)
            if gold.get("ignore_space"):
                y = re.sub(r"\s+", "", y)
            if gold.get("strip_parens"):
                y = y.strip("()")
            return y
        A = {nz(a) for a in acc}
        return float(any(nz(s) in A for s in typed))
    if kind == "anyof":
        return float(typed in set(gold["items"]))
    if kind == "set":
        return float(typed == sorted(gold["items"]))
    if kind == "multiset":
        return float(typed == sorted(light(x) for x in gold["items"]))
    if kind == "tuple":
        return float(typed[0] == light(gold["fields"][0]) and typed[1] == gold["fields"][1])
    if kind == "ipair":
        return float(tuple(typed) == tuple(gold["pair"]))
    raise ValueError(f"unknown answer format {kind}")


def _same_val(kind, a, b):
    if kind in ("int", "number"):  # "2 thousand" and "2000" are the same value
        return abs(float(a[0]) - float(b[0])) <= 1e-9 * max(1.0, abs(float(a[0])))
    return a == b


def _cont_conflict(kind, typed, cont, pub):
    """Competing answers after the answer block: a line starting with "or", a correction or uncertainty
    marker, or a short line that holds another value (also as "label: value")."""
    if _CORR.search(cont) or _UNSURE.search(cont):
        return True
    for ln in [x.strip() for x in str(cont or "").split("\n") if x.strip()]:
        if re.match(r"(?i)or\b", ln):
            return True
        if len(ln) > 40:
            continue
        for cand in (ln, re.sub(r"^[A-Za-z ]{1,20}[:：]\s*", "", ln)):
            t2 = EXTRACT[kind](cand, pub)
            if t2 is not INVALID and not _same_val(kind, t2, typed):
                return True
    return False


def judge(kind, pub, gold, output):
    """Score one output.

    kind is the answer format, pub the public contract (options, layout, units, ...) and gold the gold
    contract. Returns {"score": 0.0 or 1.0, "valid": whether an answer was extracted, "how": where the
    answer was found, ...}; "reason" says why an output scored 0 without a valid answer."""
    if kind not in EXTRACT:
        raise ValueError(f"no extractor for answer format {kind!r}")
    _SKIPPED.clear()
    _WHY[0] = None
    pay, how, cont = final_block(output, kind, pub)
    if pay is None:
        rs = how if how in ("truncated_reasoning", "boxed_unclosed", "negated_box", "not_one_line", "cot_claim_conflict") else \
            ("empty_declaration" if how in ("decl", "boxed") else "no_answer")
        if _SKIPPED and rs in ("no_answer", "not_one_line"):
            rs = "negated_or_conditional_claim"
        return {"score": 0.0, "valid": False, "how": how, "reason": rs}
    if kind == "multiset" and cont:  # "Answer:\nCircle\nRectangle": consecutive short lines form the list
        blk, used = [], 0
        for ln in cont.split("\n"):
            used += 1
            if not ln.strip():
                if blk:
                    break
                continue
            if len(ln.strip()) > 80:
                break
            blk.append(ln.strip())
        if blk:
            pay = pay + "\n" + "\n".join(blk)
            cont = "\n".join(cont.split("\n")[used:])
    if kind in ("text", "multiset") or (pub.get("unanswerable_capable") and _is_unans(pay)):
        extra = [x.strip() for x in cont.split("\n") if x.strip()] if cont else []
        short_alone = [x for x in extra if len(x.split()) <= 4 and not re.search(r"[.!?:]$", x)] if kind != "multiset" else []
        if cont and (_CORR.search(cont) or _UNSURE.search(cont) or re.search(r"(?im)^\s*or\b", cont) or short_alone):
            return {"score": 0.0, "valid": False, "how": how, "reason": "competing_answer", "pay": str(pay)[:160]}  # "Answer: Paris\nLondon"
    if pub.get("unanswerable_capable") and _is_unans(pay):
        return {"score": float(bool(gold.get("unanswerable"))), "valid": True, "how": how, "parsed": "UNANSWERABLE"}
    if gold.get("unanswerable"):
        return {"score": 0.0, "valid": True, "how": how}
    typed = x_mcq(pay, pub, cont) if kind == "mcq" else EXTRACT[kind](pay, pub)
    reason = None
    if typed is INVALID:
        reason = "competing_answer" if (kind == "mcq" and cont and x_mcq(pay, pub, "") is not INVALID) else \
            (_WHY[0] or ("negated_or_conditional_claim" if _SKIPPED and how == "whole" else "extract_fail"))
    elif kind in _SCALAR and cont and _cont_conflict(kind, typed, cont, pub):
        typed = INVALID
        reason = "competing_answer"
    r = {"score": compare(kind, typed, gold, pub), "valid": typed is not INVALID, "how": how, "pay": str(pay)[:160]}
    if typed is not INVALID:
        r["typed"] = repr(typed)[:80]
    if reason:
        r["reason"] = reason
    return r
