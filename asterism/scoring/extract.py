"""Peeling of wrappers around an answer payload.

Nothing here looks at the gold answer.
"""
import re

_PAIRS = {"(": ")", "[": "]", "{": "}", "<": ">"}


def _balanced(s, angle=False):
    """True if the brackets in s are balanced.

    Angle brackets count only when an angle-bracket wrapper is being removed, so that
    "(f(1) > g(1) > h(1))" can still lose its outer parentheses.
    """
    pairs = dict(_PAIRS) if angle else {k: v for k, v in _PAIRS.items() if k != "<"}
    st = []
    for c in s:
        if c in pairs:
            st.append(pairs[c])
        elif c in pairs.values():
            if not st or st.pop() != c:
                return False
    return not st


def _unwrap_once(s):
    """Remove exactly one layer: trailing punctuation, then a symmetric wrapper
    (**, __, `, quotes, ...) that does not occur inside, then a balanced bracket pair.
    One layer at a time keeps middle layers such as "<image_2>" in "**<image_2>**"."""
    o = s
    t = re.sub(r"[\s\.;,!]+$", "", s).strip()
    if t != o:
        return t
    for a, b in (("**", "**"), ("__", "__"), ("$$", "$$"), ("\\(", "\\)"), ("\\[", "\\]"), ("$", "$"), ("`", "`"),
                 ('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")):
        if len(s) > len(a) + len(b) and s.startswith(a) and s.endswith(b):
            inner = s[len(a):len(s) - len(b)]
            if a not in inner and b not in inner:  # "'White', '10%'" is not one quoted string
                return inner.strip()
    if len(s) >= 2 and s[0] in _PAIRS and s[-1] == _PAIRS[s[0]] and _balanced(s[1:-1], angle=(s[0] == "<")):
        return s[1:-1].strip()
    return o


def unwrap_layers(p):
    """Candidates from the raw payload to the fully unwrapped one, one layer at a time.

    The order matters: matchers try the raw text first, so a list answer such as
    "[3, 2, 0, 1]" is seen before its brackets are removed.
    """
    out = []
    s = str(p).strip()
    for _ in range(10):
        if s and s not in out:
            out.append(s)
        n = _unwrap_once(s)
        if n == s:
            break
        s = n
    if s and s not in out:
        out.append(s)
    return out
