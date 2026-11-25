"""Light normalization and number parsing.

Digits, signs, decimal points, currency symbols and other symbols are kept: a strong
normalization that drops all punctuation would merge 1.07 / 10.7 / 107, -2 / 2 and $ / €.
"""
import datetime
import re
import unicodedata

_WS = re.compile(r"\s+")


def nfkc(s):
    return unicodedata.normalize("NFKC", str(s))


def light(s):
    """NFKC, lower case, collapsed spaces, outer quotes or backticks and trailing . ; : ! , removed."""
    t = _WS.sub(" ", nfkc(s).strip().lower())
    t = t.strip("\"'`“”‘’ ")
    t = re.sub(r"[\s\.;:!,]+$", "", t)
    return t.strip()


_NUMW = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
         "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
         "hundred": 100, "none": None}
_BADUNIT = {"or", "and", "more", "less", "fewer", "than", "to", "plus", "minus", "not", "nearly", "almost", "about", "around", "over", "under",
            "above", "below", "least", "most", "approximately", "roughly", "times"} - {"times"}
_SCALEW = {"thousand", "thousands", "million", "millions", "billion", "billions", "bn", "trillion", "trillions", "k", "m", "mn"}
_NUM = re.compile(r"^[\+\-−]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?$")


def parse_number(s):
    """Float value when the whole payload is one number, else None.

    Allowed: a currency sign, %, one unit word, a fraction, English number words up to 99
    and scientific notation. Two numbers, ranges and expressions give None.
    """
    t = light(s).replace("−", "-")
    t = re.sub(r"^[\$€£¥₩]\s*", "", t)
    t = re.sub(r"\s*percentage[\s\-]+points?\.?$", " pp", t)  # "12 percentage points" counts as one unit word
    m = re.fullmatch(r"([\+\-]?[\d,]*\.?\d+)((?:e[\+\-]?\d+)?)\s*(%|[a-z°²³/]{1,12}\.?)?", t)
    if m and _NUM.match(m.group(1)) and any(c.isdigit() for c in m.group(1)):
        u = (m.group(3) or "").rstrip(".")
        if u in _SCALEW:  # "2 thousand" is a scale word, handled by the caller
            return None
        if u in _BADUNIT:  # "4 more", "4 or"
            return None
        try:
            return float(m.group(1).replace(",", "") + m.group(2))
        except (ValueError, OverflowError):
            return None
    m = re.fullmatch(r"([\+\-]?\d{1,30})\s*/\s*(\d{1,30})", t)
    if m and int(m.group(2)) != 0:
        try:
            return int(m.group(1)) / int(m.group(2))
        except OverflowError:
            return None
    w = re.fullmatch(r"([a-z]+)(?:[\s\-]([a-z]+))?(?:\s+([a-z]{1,12}))?", t)  # "three", "twenty-one", "three objects"
    if w and w.group(1) in _NUMW and _NUMW[w.group(1)] is not None:
        v = _NUMW[w.group(1)]
        tail = [x for x in (w.group(2), w.group(3)) if x]
        if tail and tail[0] in _NUMW:
            if _NUMW[tail[0]] is not None and v >= 20 and _NUMW[tail[0]] < 10:
                v += _NUMW[tail[0]]
                tail = tail[1:]
            else:
                return None
        if len(tail) > 1 or any(x in _BADUNIT or x in _SCALEW for x in tail):  # "four or five", "two thousand"
            return None
        return float(v)
    return None


_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september",
                                        "october", "november", "december"], 1)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})


def fold(s):
    """Comparison form for short text answers.

    Strips diacritics, lower-cases, maps & to "and", turns punctuation into spaces (keeping
    decimal points, signs, % and currency or operator symbols as separate tokens) and drops
    a leading article.
    """
    t = unicodedata.normalize("NFKD", nfkc(s))
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = t.replace("&", " and ")
    t = re.sub(r"(?<=\d)[,](?=\d{3}\b)", "", t)  # 1,000 -> 1000
    t = re.sub(r"([$€£¥₩+#<>=])", r" \1 ", t)  # $712M != €712M, C++ != C# != C
    t = re.sub(r"(?<!\d)[.](?!\d)|[^\w\s.%\-$€£¥₩+#<>=]|(?<=\w)-(?=\w)|_", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    o = nfkc(s).strip().strip("\"'`“”‘’*_<>()[]{} ")
    if re.match(r"^(?:the|an)\s+\S", o, re.I) or re.match(r"^a\s+[a-z]", o):  # the "A" of "A & B" is a name
        t = re.sub(r"^(?:the|an|a)\s+", "", t)
    return t


def as_date(s):
    """(year, month, day) for common date spellings, else None. Ambiguous forms such as 03/04 give None."""
    t = fold(s)
    t = re.sub(r"\bday of\b", "", t)
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    m = re.fullmatch(r"(\d{1,2}) ([a-z]+) (\d{4})", t) or re.fullmatch(r"([a-z]+) (\d{1,2}) (\d{4})", t)
    if m:
        a, b, y = m.groups()
        mon, day = (b, a) if a.isdigit() else (a, b)
        if mon in _MONTHS:
            return _valid_date(int(y), _MONTHS[mon], int(day))
    m = re.fullmatch(r"(\d{4}) (\d{1,2}) (\d{1,2})", t)
    if m:
        return _valid_date(*(int(x) for x in m.groups()))
    return None


def _valid_date(y, mo, d):
    try:
        datetime.date(y, mo, d)
        return (y, mo, d)
    except ValueError:
        return None


# Unit contract: a number may carry only a unit that the question allows. The gold answer is not used.
_UNITS = {"second": "s sec secs second seconds", "minute": "min mins minute minutes", "hour": "h hr hrs hour hours", "day": "day days",
          "week": "week weeks", "month": "month months", "year": "yr yrs year years",
          "millimeter": "mm millimeter millimeters millimetre millimetres",
          "centimeter": "cm centimeter centimeters centimetre centimetres", "meter": "m meter meters metre metres",
          "kilometer": "km kilometer kilometers kilometre kilometres",
          "inch": "inch inches", "foot": "ft foot feet", "yard": "yd yard yards", "mile": "mi mile miles",
          "milligram": "mg milligram milligrams",
          "gram": "g gram grams", "kilogram": "kg kilogram kilograms kilo kilos", "pound_mass": "lb lbs", "ounce": "oz ounce ounces",
          "ton": "ton tons tonne tonnes",
          "milliliter": "ml milliliter milliliters millilitre millilitres", "liter": "l liter liters litre litres",
          "gallon": "gal gallon gallons",
          "degree": "deg degree degrees °", "km_per_h": "km/h kmh kph", "m_per_s": "m/s", "mph": "mph", "times": "x × times fold",
          "square_meter": "m2 m² sqm", "square_centimeter": "cm2 cm²", "square_kilometer": "km2 km²", "square_foot": "ft2 ft² sqft",
          "square_inch": "in2 in²",
          "cubic_meter": "m3 m³", "cubic_centimeter": "cm3 cm³ cc",
          "joule": "j joule joules", "kilojoule": "kj kilojoule kilojoules", "newton": "n newton newtons", "watt": "w watt watts",
          "kilowatt": "kw kilowatt kilowatts",
          "volt": "v volt volts", "ampere": "amp amps ampere amperes", "pascal": "pa pascal pascals", "hertz": "hz hertz",
          "calorie": "cal calorie calories", "kilocalorie": "kcal"}
_UCANON = {v: k for k, vs in _UNITS.items() for v in vs.split()}


def canon_unit(w):
    """Canonical unit name for a unit spelling, or None. "square meters" and "sq m" give square_meter."""
    t = re.sub(r"[.\s]+$", "", str(w).strip().lower())
    t = re.sub(r"\s+", " ", t)
    m = re.fullmatch(r"(square|sq|cubic|cu)\.?\s+(.+)", t)
    if m:
        base = _UCANON.get(m.group(2))
        if base in ("meter", "centimeter", "kilometer", "foot", "inch", "millimeter"):
            return ("square_" if m.group(1) in ("square", "sq") else "cubic_") + base
        return None
    return _UCANON.get(t)


def _plural_variants(ph):
    return {ph, re.sub(r"s$", "", ph), re.sub(r"es$", "", ph), ph + "s", ph + "es", re.sub(r"ies$", "y", ph), re.sub(r"y$", "ies", ph)}


_Q_STOP = (r"(?=\s+(?:are|is|were|was|do|does|did|can|could|will|would|should|have|has|had|in|on|at|of|from|for|to|with|by|there|"
           r"appear|appears|exist|exists|shown|visible|total|remain|left|that|which|who)\b|\s*[?,.;:!)]|\s*$)")


def unit_contract(question):
    """Question text -> (allowed unit phrases, percentage-point flag, money context).

    Allowed units: the counted noun phrase of "how many" / "number of" (and its head noun),
    explicit units ("in <unit>", "formatted as 'xx <unit>'"), degree for angle questions,
    and x / times for ratio questions.
    """
    q0 = re.sub(r"\s+", " ", str(question).lower()).replace("$", " ")
    q = re.sub(r"[\"“”‘’`]|(?<![a-z])'|'(?![a-z])", " ", q0)  # quotes around the counted noun; apostrophes stay
    q = re.sub(r"\s+", " ", q)
    units = set()
    hit = set()
    for rx in (r"\bhow[a-z]?\s+many\s+((?:[a-z][a-z\-']*\s+){0,3}?[a-z][a-z\-']*?)" + _Q_STOP,
               r"\b(?:number|count|amount|total)\s+of\s+((?:[a-z][a-z\-']*\s+){0,2}?[a-z][a-z\-']*?)" + _Q_STOP):
        for m in re.finditer(rx, q):
            w = [x for x in m.group(1).split() if x not in ("the", "a", "an")]
            for i in range(1, len(w) + 1):
                units.add(" ".join(w[:i]))  # leading phrases ("months take injection" -> "months")
            if w:
                units.add(w[-1])  # head noun ("fertilizer warehouses" -> "warehouses")
                hit.add(m.start(1))
    for m in re.finditer(r"\bhow[a-z]?\s+many\s+((?:[a-z][a-z\-']*)(?:\s+[a-z][a-z\-']*){0,3})", q):  # "how many lines go through" -> "lines"
        if m.start(1) in hit:
            continue
        w = [x for x in m.group(1).split() if x not in ("the", "a", "an")]
        for i in range(1, len(w) + 1):
            units.add(" ".join(w[:i]))
    for m in re.finditer(r"\bin\s+(?:terms\s+of\s+)?((?:square\s+|cubic\s+|sq\.?\s+)?[a-z²³]+(?:/[a-z]+)?)\b", q):
        if canon_unit(m.group(1)):
            units.add(m.group(1))
    for m in re.finditer(r"formatted\s+as\s+['\"]?\s*(?:x+|n+|\d+|#+)\s*([a-z]+(?:/[a-z]+)?)", q0):
        units.add(m.group(1))
    if re.search(r"(?<![a-z])(?:angles?|degrees?|rotat\w*)(?![a-z])|°|\\circ", q):  # word boundary: not "triangle"
        units.add("degree")
    for m in re.finditer(r"\bhow\s+much\s+(higher|lower|greater|larger|smaller|faster|slower|longer|shorter|bigger|older|younger|heavier"
                         r"|lighter|taller|cheaper)\b", q):
        units.add(m.group(1))  # "How much higher ..." -> "18.5% higher"
    if re.search(r"speed\s*up|speedup|how\s+many\s+times|\btimes\s+(?:faster|slower|more|less|larger|smaller|higher|lower|greater|as)\b"
                 r"|\d+\s*-?\s*fold|\bfold\b|faster\s+than|slower\s+than|multiple\s+of", q):
        units.add("times")
    diffq = re.search(r"(?<![a-z])(?:difference|gap|increase|decrease|higher|lower|greater|larger|smaller|more\s+than|less\s+than"
                      r"|compared\s+(?:to|with)|changed?|changes|grow|grew|drop|rise|rose|fell|decline)(?![a-z])", q)
    pp_ok = bool(re.search(r"percentage\s+points?|\bpp\b", q)
                 or (diffq and re.search(r"(?<![a-z])(?:percent|percentage|proportion|share|rate)(?![a-z])|%", q)))
    money = bool(re.search(r"(?<![a-z])(?:cost|costs|price|prices|money|pay|paid|payment|spend|spent|spending|revenues?|income|salary|salaries"
                           r"|wages?|budget|fees?|funds?|funding|grants?|dollars?|euros?|pounds?|yen|yuan|won|usd|worth"
                           r"|expenses?|expenditures?|sales|profits?|earnings|assets?|liabilit(?:y|ies)|debts?|cash|capital|dividends?"
                           r"|tax(?:es)?|amortization|depreciation|valuation|investments?|compensation|financ\w*|fiscal|net\s+loss"
                           r"|leases?|fy|ebitda|margins?|equity|operating\s+(?:income|expenses?|costs?|leases?)|gross|balance\s+sheet)"
                           r"(?![a-z])|answer\s+in\s+(?:millions?|billions?|thousands?)"
                           r"|[\$€£¥₩]\s*\d|\d\s*[€£¥₩]|value\s+of\s+the\s+(?:contract|deal|award)", str(question).lower()))
    return sorted(units), pp_ok, money


def unit_ok(ph, units):
    """A unit phrase after a number is allowed if it is in the allowed list (singular or
    plural form) or has the same canonical unit as an allowed one. Unknown words are not."""
    ph = re.sub(r"[.\s]+$", "", str(ph).strip().lower())
    if not ph:
        return True
    A = set(units or [])
    if _plural_variants(ph) & A:
        return True
    cu = canon_unit(ph)
    return bool(cu) and cu in {canon_unit(a) for a in A if canon_unit(a)}


def target_expr(question):
    """Left-hand side that an equation answer may use: the last thing the question asks
    for ("compute g(v)", "find x", "what is the value of c"), or None."""
    q = re.sub(r"\s+", " ", str(question).lower()).replace("$", " ")
    q = re.sub(r"\\[()\[\]]", " ", q)
    c = []
    _FN = r"[a-z]\s*\(\s*-?[a-z0-9]{1,3}\s*\)"  # f(-1); a compound target such as "f(1) - g(1)" is kept whole
    for m in re.finditer(r"(?:compute|calculate|find|evaluate|determine|what\s+is|what's|give)\s+(?:the\s+value\s+of\s+)?(" + _FN
                         + r"(?:\s*[+\-*/]\s*" + _FN + r")*)(?=\s*(?:[?.,;:]|$))", q):
        c.append((m.start(), re.sub(r"\s+", "", m.group(1))))
    for m in re.finditer(r"(?:find|solve\s+for|compute|calculate|determine|what\s+is|what's)\s+(?:the\s+value\s+of\s+)?([a-z])"
                         r"(?=\s*(?:[?.,;:]|$)|\s+(?:if|when|given|such|in|for|so)\b)", q):
        c.append((m.start(), m.group(1)))
    return max(c)[1] if c else None
