"""Prompt text rules: the answer-format line, the final answer line and the option exchange."""
import re

FINAL_ANSWER_LINE = ("Respond with only the final answer on one line, written as 'Answer: <your answer>' "
                     "following the required format above.")

LETTER = "Required answer format: the option letter only (e.g., B)."
TRUE_FALSE = "Required answer format: True or False only."
YES_NO = "Required answer format: Yes or No only."
IMAGE_NUMBER = "Required answer format: name the image number (e.g., Image 2)."
SENTENCE = "Required answer format: one short factual sentence (as brief as possible)."
SHORT = "Required answer format: a short answer (word or phrase)."

_ANSWER_PREFIX = re.compile(r"^\s*answer\s*[:：]\s*", re.I)
_ANSWER_TAIL = re.compile(r"(?:final\s+)?answer\s*[:：]\s*([^\n]{1,120})\s*$", re.I)
_SAME = re.compile(r"\b(look|looks|are|appear|appears|re)\s+(exactly\s+)?(the\s+)?(same|identical)"
                   r"|\bno (visible )?difference|\bnearly identical|\bsame (kind|type|species)\b")
_ORDINALS = {"first": 1, "initial": 1, "1st": 1, "one": 1, "primary": 1,
             "second": 2, "secondary": 2, "2nd": 2, "two": 2,
             "third": 3, "tertiary": 3, "3rd": 3, "three": 3,
             "fourth": 4, "4th": 4, "four": 4, "quaternary": 4,
             "fifth": 5, "5th": 5, "five": 5, "sixth": 6, "6th": 6, "six": 6,
             "seventh": 7, "7th": 7, "seven": 7, "eighth": 8, "8th": 8, "eight": 8,
             "last": -1, "final": -1}


def direct_prompt(text, format_line):
    """Item text, the answer-format line and the final answer line."""
    return f"{text}\n{format_line}\n{FINAL_ANSWER_LINE}"


def clean_label(label):
    """Drop an 'Answer:' prefix, or take the trailing 'Answer: X' of a label that explains itself."""
    g = _ANSWER_PREFIX.sub("", str(label)).strip()
    m = _ANSWER_TAIL.search(g)
    if m and m.start() >= 20 and m.group(1).strip():
        return m.group(1).strip()
    return g


def _plain(t):
    t = re.sub(r"[^a-z0-9 ]", " ", str(t).lower())
    return re.sub(r"\s+", " ", t).strip()


def image_number(t):
    """Image number named by a text such as 'Image 3' or 'the third image', or None."""
    t = _plain(t)
    m = re.search(r"image\s*(\d+)", t)
    if m:
        return int(m.group(1))
    m = re.search(r"(\w+)\s+image", t)
    if m and m.group(1) in _ORDINALS:
        return _ORDINALS[m.group(1)]
    for w, n in _ORDINALS.items():
        if re.search(rf"\b{w}\b", t) and "image" in t:
            return n
    return None


def true_false_format(label):
    return YES_NO if clean_label(label).lower() in ("yes", "no") else TRUE_FALSE


def free_format(label):
    """Answer-format line of an item whose answer is not chosen from listed options."""
    g = clean_label(label)
    if re.fullmatch(r"\(?[A-J]\)?", g):
        return LETTER
    if image_number(g) is not None:
        return IMAGE_NUMBER
    if _SAME.search(g.lower()) or len(g.split()) >= 4:
        return SENTENCE
    return SHORT


def _label_at(prompt, letter, text):
    """The one place where `letter` is directly followed by its option text, or None."""
    if not text.strip():
        return None
    pattern = re.compile(r"(?<![A-Za-z0-9])\(?" + letter + r"\)?[.:)]?\s+(" + re.escape(text) + r")")
    hits = list(pattern.finditer(prompt))
    return hits[0] if len(hits) == 1 else None


def exchange_options(prompt, a, text_a, b, text_b):
    """Swap the texts of options `a` and `b` in the prompt (used to balance answer positions)."""
    ma, mb = _label_at(prompt, a, text_a), _label_at(prompt, b, text_b)
    if not ma or not mb:
        raise ValueError(f"options {a} and {b} not found once each")
    (a0, a1), (b0, b1) = ma.span(1), mb.span(1)
    if not (a1 <= b0 or b1 <= a0):
        raise ValueError(f"options {a} and {b} overlap")
    if a0 < b0:
        new = prompt[:a0] + text_b + prompt[a1:b0] + text_a + prompt[b1:]
    else:
        new = prompt[:b0] + text_a + prompt[b1:a0] + text_b + prompt[a1:]
    if not _label_at(new, a, text_b) or not _label_at(new, b, text_a):
        raise ValueError(f"options {a} and {b} could not be exchanged")
    return new
