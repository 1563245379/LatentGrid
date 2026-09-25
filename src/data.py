import json
import random
from pathlib import Path


RESPONSE_SEPARATOR = "--- response_template ---"


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_train_validation(path, validation_ratio, max_train_samples=None, seed=42):
    usable = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line.strip():
                record = json.loads(line)
                if record["cot"].strip():
                    usable.append(dict(record, example_id=line_number))
    indices = list(range(len(usable)))
    random.Random(seed).shuffle(indices)
    validation_count = int(len(usable) * validation_ratio)
    validation = [usable[index] for index in indices[:validation_count]]
    train = [usable[index] for index in indices[validation_count:]]
    if max_train_samples is not None:
        train = train[:max_train_samples]
    return train, validation


def load_prompt_template(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    return text.split(RESPONSE_SEPARATOR, maxsplit=1)[0].strip()


def load_prompt_response_templates(path: str | Path) -> tuple[str, str]:
    text = Path(path).read_text(encoding="utf-8")
    parts = text.split(RESPONSE_SEPARATOR, maxsplit=1)
    if len(parts) != 2:
        raise ValueError(
            f"template must contain {RESPONSE_SEPARATOR!r}: {path}"
        )
    return parts[0].strip(), parts[1].strip()


def render_prompt(prompt_template: str, question: str) -> str:
    return prompt_template.format(question=question, response="")


def render_no_cot(
    prompt_template: str,
    response_template: str,
    question: str,
    answer: str = "",
) -> str:
    response = response_template.format(cot="", answer=answer)
    return prompt_template.format(question=question, response=response)
