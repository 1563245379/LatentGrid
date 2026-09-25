import json
import re
from decimal import Decimal
from pathlib import Path

import torch


ANSWER_MARKER = "###Answer:"
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")


def _normalize_number(candidate: str) -> str:
    value = Decimal(candidate.replace(",", ""))
    if value == 0:
        return "0"
    normalized = format(value, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def extract_numeric_answer(text: str) -> str | None:
    answer_region = text.rsplit(ANSWER_MARKER, maxsplit=1)[-1]
    matches = NUMBER_PATTERN.findall(answer_region)
    return _normalize_number(matches[-1]) if matches else None


def score_generations(
    records: list[dict], generations: list[str]
) -> tuple[list[dict], dict]:
    if len(records) != len(generations):
        raise ValueError("records and generations must have equal length")

    predictions = []
    for index, (record, generation) in enumerate(zip(records, generations)):
        predicted = extract_numeric_answer(generation)
        gold = extract_numeric_answer(record["answer"])
        predictions.append(
            {
                "index": index,
                "question": record["question"],
                "gold_answer": gold,
                "predicted_answer": predicted,
                "correct": predicted is not None and predicted == gold,
                "generation": generation,
            }
        )

    correct = sum(row["correct"] for row in predictions)
    total = len(predictions)
    metrics = {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
    }
    return predictions, metrics


def score_latent_generations(
    records: list[dict],
    generations: list[str],
    latent_lengths,
    latent_truncated=None,
) -> tuple[list[dict], dict]:
    predictions, metrics = score_generations(records, generations)
    lengths = torch.tensor(latent_lengths, dtype=torch.float32)
    for prediction, length in zip(predictions, latent_lengths):
        prediction["latent_length"] = int(length)
    metrics.update(
        {
            "latent_length_mean": lengths.mean().item(),
            "latent_length_p50": torch.quantile(lengths, 0.50).item(),
            "latent_length_p95": torch.quantile(lengths, 0.95).item(),
        }
    )
    if latent_truncated is not None:
        flags = [bool(value) for value in latent_truncated]
        for prediction, flag in zip(predictions, flags):
            prediction["latent_truncated"] = flag
        metrics["latent_truncation_rate"] = sum(flags) / len(flags)
    return predictions, metrics


def write_results(
    output_dir: str | Path, predictions: list[dict], metrics: dict
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
