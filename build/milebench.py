"""MileBench: the six archives MileBench_part0-5.tar.gz extracted into one folder (<task>/<task>.json, <task>/images/)."""
import json
import os

from . import images, scoring, text

NAME = "MileBench"
SLUG = "milebench"
NEEDLE_TASKS = {"ImageNeedleInAHaystack", "TextNeedleInAHaystack"}


class Release:
    official_prompt = False
    layout = {"layout": "direct", "answer_line": True}

    def __init__(self, root, records):
        self.root = root
        self._tasks = {}

    def _task(self, task):
        if task not in self._tasks:
            with open(os.path.join(self.root, task, f"{task}.json"), encoding="utf-8") as f:
                data = json.load(f)
            samples = {str(s["sample_id"]): s for s in data["data"]}
            self._tasks[task] = (data["meta_data"]["task_instruction"], samples)
        return self._tasks[task]

    def item(self, record):
        task = record["source_subset"]
        instructions, samples = self._task(task)
        sample = samples[record["source_id"]]
        inst = sample["task_instance"]
        context = inst["context"]
        if "choice_list" in inst:
            choices = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(inst["choice_list"]))
            context += f"\nChoice list: \n{choices}\nYour answer is: "
        body = instructions[sample.get("task_instruction_id", 0)] + "\n" + context
        return {"text": body, "label": str(sample["response"]), "task": task}

    def entry(self, item):
        if item["task"] in NEEDLE_TASKS:
            return "identifier", {}, {"value": item["label"].strip()}
        return scoring.label_entry(item["text"], item["label"], options_by_text=True)

    def format_line(self, item, label):
        return text.free_format(label) if item["task"] in NEEDLE_TASKS else text.LETTER

    def image(self, ref, out_root, link):
        rel = f"images/{SLUG}/{ref}"
        images.place_file(os.path.join(self.root, ref), os.path.join(out_root, rel), link)
        return rel
