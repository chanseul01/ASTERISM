"""Scoring entries of the identifier sources.

The released scoring entries were derived by rules from the source task, the question, the options
and the label; the rules below are those rules for the seven identifier sources. Entries that hold
a decision of ours (a corrected answer format, a reviewed tolerance) come with the ASTERISM record.
"""
import json
import re

from asterism.scoring.normalize import light, parse_number, target_expr, unit_contract
from asterism.scoring.options import parse_options

_QUANTITY = re.compile(
    r"\bhow\s+many\b|\bhow\s+much\b|\bwhat\s+(?:is|was|are|were)\s+the\s+(?:total\s+)?(?:number|amount|count|sum|percentage|proportion|ratio|share)\s+of\b"
    r"|\btotal\s+(?:number|amount|count)\b|\bhow\s+old\b|\bhow\s+long\b|\bhow\s+far\b|\bwhat(?:'s|\s+is)\s+the\s+value\s+of\b", re.I)
_SCALE = re.compile(r"\b(thousand|million|billion|trillion)s?\b", re.I)
_CURRENCY = re.compile(r"[\$€£¥₩]")
_PERCENT = re.compile(r"percent|percentage|%", re.I)
UNANSWERABLE = ["not answerable", "unanswerable", "cannot be answered", "not answerable from the given pages"]


class Unresolved(ValueError):
    pass


def question_of(prompt):
    p = str(prompt or "")
    return p.rsplit("Question:", 1)[-1] if "Question:" in p else p[-400:]


def quantity_public(prompt, base):
    public = dict(base)
    words = sorted({m.group(1).lower() for m in _SCALE.finditer(question_of(prompt))})
    if words:
        public["scale_words"] = words
    return public


def options_of(prompt):
    parsed = parse_options(prompt)
    return parsed["opts"] if parsed else None


def mcq_by_letter(options, label):
    if label not in options:
        raise Unresolved(f"label {label!r} is not an option")
    return "mcq", {"options": options}, {"answer": label}


def mcq_by_text(options, label):
    hits = [k for k, v in options.items() if v and light(v) == light(label)]
    if len(hits) != 1:
        raise Unresolved(f"label {label!r} matches {len(hits)} options")
    return "mcq", {"options": options}, {"answer": hits[0]}


def numeric_entry(label, public, prompt):
    """Number entry for a question that asks for a quantity, else None."""
    question = question_of(prompt)
    if not _QUANTITY.search(question):
        return None
    v = parse_number(label)
    if v is None:
        return None
    m = _CURRENCY.search(label)
    percent_ok = bool("%" in label or _PERCENT.search(question))
    if float(v).is_integer():
        gold = {"value": int(v), "percent_ok": percent_ok}
        if m:
            gold["currency"] = m.group(0)
        return "int", quantity_public(prompt, public), gold
    gold = {"value": v, "atol": 0.0, "rtol": 0.0}
    if m:
        gold["currency"] = m.group(0)
    gold["percent_ok"] = percent_ok
    return "number", quantity_public(prompt, public), gold


def label_entry(prompt, label, options_by_text=False, true_false_letters=False):
    """Entry from the options in the prompt and the label, as for most sources."""
    g = str(label).strip()
    options = options_of(prompt)
    if options and options_by_text:
        return mcq_by_text(options, g)
    if options and re.fullmatch(r"[A-J]", g):
        return mcq_by_letter(options, g)
    if options:
        return mcq_by_text(options, g)
    lg = light(g)
    v = parse_number(g)
    if true_false_letters and lg in ("t", "f", "true", "false"):
        return "polar", {"polar_as": "T/F"}, {"value": lg in ("t", "true")}
    if lg in ("true", "false", "yes", "no"):
        return "polar", {"polar_as": "True/False" if lg in ("true", "false") else "Yes/No"}, {"value": lg in ("true", "yes")}
    if _QUANTITY.search(prompt[-700:]) and v is not None and float(v).is_integer():
        return "int", quantity_public(prompt, {}), {"value": int(v)}
    if v is not None:
        if float(v).is_integer():
            return "int", {}, {"value": int(v)}
        return "number", {}, {"value": v, "atol": 0.0, "rtol": 0.0, "percent_ok": "%" in g}
    return "text", {}, {"accepted": [g]}


def document_entry(prompt, label, answer_format):
    """MMLongBench-Doc: the answer format of the source (Int, Float, Str, List, None) decides the entry."""
    g = str(label).strip()
    base = {"unanswerable_capable": True}
    question = question_of(prompt)
    if answer_format == "None" or light(g) in UNANSWERABLE:
        return "text", base, {"accepted": ["Not answerable"], "unanswerable": True}
    if answer_format == "Int":
        v = parse_number(g)
        if v is not None and float(v).is_integer():
            return "int", quantity_public(prompt, base), {"value": int(v), "percent_ok": bool("%" in g or _PERCENT.search(question))}
    if answer_format == "Float":
        v = parse_number(g)
        if v is not None:
            gold = {"value": v, "atol": 0.0, "rtol": 0.01, "percent_ok": bool("%" in g or _PERCENT.search(question))}
            if _CURRENCY.search(g):
                gold["currency"] = _CURRENCY.search(g).group(0)
            return "number", quantity_public(prompt, base), gold
    if answer_format == "List":
        try:
            items = json.loads(g.replace("'", '"')) if g.startswith("[") else None
        except ValueError:
            items = None
        if isinstance(items, list):
            return "multiset", base, {"items": [str(x) for x in items]}
    return numeric_entry(g, base, prompt) or ("text", base, {"accepted": [g]})


def finish(kind, public, gold, prompt, layout="direct", answer_line=True):
    """Add the prompt layout and, for numbers, the units and the money context the question allows."""
    public = dict(public)
    public["layout"] = layout
    if answer_line:
        public["answer_line"] = True
    if kind in ("int", "number"):
        question = question_of(prompt)
        units, _, money = unit_contract(question)
        public["units"] = units
        public["pp_ok"] = False   # percentage points are accepted only for a reviewed list, carried in the record
        public["money"] = bool(money or gold.get("currency"))
        te = target_expr(question)
        if te:
            public["target_expr"] = te
    return kind, public, gold
