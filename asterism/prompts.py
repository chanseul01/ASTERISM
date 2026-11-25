"""Chat messages that a model receives for an ASTERISM item.

The `prompt` of an item is the complete text of the user turn: the question with its options,
the answer-format instruction and, where the source has an official prompt (MV-MATH with its worked
examples, EMMA, ReMI), that prompt. Every model gets the same messages:
- the images of the item in order, followed by the prompt;
- EMMA items whose prompt holds the placeholders <image_1> to <image_n> once each and in order
  take each image at the position of its placeholder instead, as in the official EMMA evaluation;
- items whose prompt starts with the worked examples of the official MV-MATH evaluation
  (MV_MATH_EXAMPLES), whatever their source, also take its system prompt; it names the letters A to E
  when the question has an option E.
Messages carry {"type": "image"} placeholders; the images themselves are passed to the model
separately, in the same order.
"""
import re

MAX_IMAGES = 128

MV_MATH_SYSTEM = ("You are an assistant for solving math problems. Your input consists of a math question and images. Your task is to "
                  "output the solution steps and the answer. The output format should be a step-by-step approach. Each question is "
                  "multiple choice with one correct answer. Your final answer must be one of A, B, C, or D, For example: {A},{B},{C} or {D}")
MV_MATH_EXAMPLES = (
    "\n"
    "Question: If a triangle has two sides of length 3 and 4, what is the length of the hypotenuse?  A.10  B.8  C.5  D.4\n"
    "\n"
    "Answer:  \n"
    "Step 1 (Mathematical theorem used: Pythagorean theorem): The Pythagorean theorem states that in a right triangle, the square of the hypotenuse is equal to the sum of the squares of the other two sides. The formula is:  \n"
    "\\[c^2 = a^2 + b^2\\], where \\( a \\) and \\( b \\) are the legs, and \\( c \\) is the hypotenuse.\n"
    "Step 2 (Substitute the known values): Given \\( a = 3 \\) and \\( b = 4 \\). Substituting these values into the formula:  \n"
    "\\[c^2 = 3^2 + 4^2 = 9 + 16 = 25\\]\n"
    "Step 3 (Calculate the hypotenuse): Taking the square root gives:  \n"
    "\\[c = \\sqrt{25} = 5\\]\n"
    "Answer: {C}\n"
    "\n"
    "\n"
    "Question: In the right triangle ABC, AB is perpendicular to BC. It is known that AC=5 and AB=4. Find the area of the right triangle.  A.20  B.10  C.5  D.6\n"
    "Answer:  \n"
    "Step 1 (Mathematical theorem used: Pythagorean theorem): We first use the Pythagorean theorem to find the length of \\( BC \\). The formula is:  \n"
    "\\[AC^2 = AB^2 + BC^2\\], where \\( AC \\) is the hypotenuse, and \\( AB \\) and \\( BC \\) are the legs.\n"
    "Step 2 (Substitute the known values): Given \\( AC = 5 \\) and \\( AB = 4 \\). Substituting these values:  \n"
    "\\[5^2 = 4^2 + BC^2 \\implies 25 = 16 + BC^2\\]\n"
    "Step 3 (Solve for \\( BC \\)):  \n"
    "\\[BC^2 = 25 - 16 = 9 \\implies BC = \\sqrt{9} = 3\\]\n"
    "Step 4 (Calculate the area): The area of the right triangle is given by \\( \\frac{1}{2} \\times AB \\times BC \\). Substituting the known values:  \n"
    "\\[ \\text{Area} = \\frac{1}{2} \\times 4 \\times 3 = 6\\]\n"
    "Answer: {D}\n"
    "\n"
)
EMMA = "EMMA"
_IMAGE_MARK = re.compile(r"<image_(\d+)>")


def prompt_style(record):
    """Prompt style of an item: "mv-math" for a prompt that starts with the MV-MATH worked examples,
    "emma" for EMMA items other than the rewrites, else "default"."""
    prompt = record.get("prompt")
    if isinstance(prompt, str) and prompt.startswith(MV_MATH_EXAMPLES):
        return "mv-math"
    if record.get("source") == EMMA and not record.get("is_rewrite"):
        return "emma"
    return "default"


def _mv_math_system(prompt):
    if re.search(r"(?m)^\s*E[\.\):]", prompt):  # five options, A to E
        return MV_MATH_SYSTEM.replace("one of A, B, C, or D", "one of A, B, C, D, or E").replace("{A},{B},{C} or {D}", "{A},{B},{C},{D} or {E}")
    return MV_MATH_SYSTEM


def user_text(record):
    """Text of the user turn: the item's prompt as released."""
    prompt = record.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError(f"{record.get('id')}: no prompt (items of identifier sources must be assembled from their public release first)")
    return prompt


def _interleaved(text, n_img):
    """User content with an image at each <image_k> marker, if the markers are exactly 1..n in order; else None."""
    ks = [int(k) for k in _IMAGE_MARK.findall(text)]
    if n_img <= 0 or ks != list(range(1, n_img + 1)):
        return None
    parts = []
    pos = 0
    for m in _IMAGE_MARK.finditer(text):
        if m.start() > pos:
            parts.append({"type": "text", "text": text[pos:m.start()]})
        parts.append({"type": "image"})
        pos = m.end()
    if pos < len(text):
        parts.append({"type": "text", "text": text[pos:]})
    return parts


def num_images(record, max_images=MAX_IMAGES):
    return min(len(record.get("images") or []), max_images)


def build_messages(record, n_img=None):
    """Chat messages for one item. n_img is the number of images given to the model
    (default: all images of the item, at most MAX_IMAGES, in item order)."""
    if n_img is None:
        n_img = num_images(record)
    style = prompt_style(record)
    text = user_text(record)
    content = _interleaved(text, n_img) if style == "emma" else None
    if content is None:
        content = [{"type": "image"} for _ in range(n_img)] + [{"type": "text", "text": text}]
    msgs = [{"role": "user", "content": content}]
    if style == "mv-math":
        msgs.insert(0, {"role": "system", "content": _mv_math_system(text)})
    return msgs
