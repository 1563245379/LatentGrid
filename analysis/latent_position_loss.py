"""Train paired latent reasoners and analyze their live position gradients."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import copy
from dataclasses import dataclass
import gc
import importlib.metadata
import json
import math
from numbers import Integral
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml

from src.data import load_prompt_template, load_train_validation
from src.training import OnlineTraceRuntime, train_indirect_from_records


PAIR_PATHS = (
    ("base_model",),
    ("template_path",),
    ("latent_representation_type",),
    ("latent",),
    ("data", "train_path"),
    ("data", "validation_ratio"),
    ("data", "max_train_samples"),
    ("checkpoint", "cot_init_checkpoint"),
    ("training",),
)

OBSERVATION_KEY = (
    "boundary_epoch",
    "optimizer_update",
    "micro_batch",
    "process_index",
    "batch_row",
    "example_id",
    "position",
)

TRACE_FIELDS = (
    "method",
    "boundary_epoch",
    "training_epoch",
    "optimizer_update",
    "micro_batch",
    "process_index",
    "batch_row",
    "example_id",
    "latent_count",
    "position",
    "grad_l2",
)

OPTIMIZER_FIELDS = (
    "method",
    "boundary_epoch",
    "training_epoch",
    "optimizer_update",
    "micro_batches",
    "num_examples",
    "answer_tokens",
    "cot_tokens",
    "loss",
    "answer_loss",
    "cot_loss",
)

PER_EXAMPLE_FIELDS = tuple(
    field for field in TRACE_FIELDS if field not in {"process_index", "batch_row"}
)

_UNIT_KEY = OBSERVATION_KEY[:-1]


@dataclass(frozen=True)
class StagePoint:
    boundary_epoch: int
    training_epoch: int
    curriculum_latent_count: int


class TraceWriter:
    """Write detached live-gradient rows and per-update scalar summaries."""

    def __init__(self, method: str, partial_path: str | Path):
        self.method = method
        self.partial_path = Path(partial_path)
        self.optimizer_rows: list[dict[str, Any]] = []
        self._handle = None
        self._update_key: tuple[int, int, int] | None = None
        self._update: dict[str, Any] | None = None

    def _open_for_main(self, accelerator) -> None:
        if accelerator.is_main_process and self._handle is None:
            self.partial_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.partial_path.open("w", encoding="utf-8")

    @staticmethod
    def _detached_scalar(value: Any, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.detach().float().reshape(1).to(device)
        return torch.tensor([float(value)], dtype=torch.float32, device=device)

    def _begin_update(
        self, boundary_epoch: int, training_epoch: int, optimizer_update: int
    ) -> None:
        key = (int(boundary_epoch), int(training_epoch), int(optimizer_update))
        if self._update_key != key:
            self._update_key = key
            self._update = {
                "method": self.method,
                "boundary_epoch": key[0],
                "training_epoch": key[1],
                "optimizer_update": key[2],
                "micro_batches": 0,
                "num_examples": 0,
                "answer_tokens": 0,
                "cot_tokens": 0,
                "losses": [],
                "answer_losses": [],
                "cot_losses": [],
            }

    def on_backward(
        self,
        *,
        accelerator,
        method: str,
        boundary_epoch: int,
        training_epoch: int,
        optimizer_update: int,
        micro_batch: int,
        update_finished: bool,
        batch: Mapping[str, Any],
        outputs: Mapping[str, Any],
    ) -> None:
        del method
        latents = outputs.get("latents")
        if not latents:
            raise ValueError("latent gradients are missing")
        example_ids = batch.get("example_ids")
        if not torch.is_tensor(example_ids) or example_ids.ndim != 1:
            raise ValueError("inconsistent local batch dimensions")
        batch_size = int(example_ids.shape[0])
        columns = []
        for latent in latents:
            gradient = getattr(latent, "grad", None)
            if gradient is None:
                raise ValueError("latent gradient is missing")
            if gradient.ndim < 1 or int(gradient.shape[0]) != batch_size:
                raise ValueError("inconsistent local batch dimensions")
            norms = gradient.detach().float().norm(dim=-1)
            if not torch.isfinite(norms).all():
                raise ValueError("latent gradient is nonfinite")
            columns.append(norms)
        local_norms = torch.stack(columns, dim=1).detach()
        device = local_norms.device
        local_ids = example_ids.to(device=device, dtype=torch.int64)
        local_rows = torch.arange(batch_size, device=device, dtype=torch.int64)
        local_process = torch.full(
            (batch_size,),
            int(accelerator.process_index),
            device=device,
            dtype=torch.int64,
        )
        gathered_norms = accelerator.gather(local_norms).detach().cpu()
        gathered_ids = accelerator.gather(local_ids).detach().cpu()
        gathered_rows = accelerator.gather(local_rows).detach().cpu()
        gathered_process = accelerator.gather(local_process).detach().cpu()

        self._open_for_main(accelerator)
        if accelerator.is_main_process:
            for row_index in range(int(gathered_norms.shape[0])):
                for position in range(int(gathered_norms.shape[1])):
                    row = {
                        "method": self.method,
                        "boundary_epoch": int(boundary_epoch),
                        "training_epoch": int(training_epoch),
                        "optimizer_update": int(optimizer_update),
                        "micro_batch": int(micro_batch),
                        "process_index": int(gathered_process[row_index].item()),
                        "batch_row": int(gathered_rows[row_index].item()),
                        "example_id": int(gathered_ids[row_index].item()),
                        "latent_count": int(gathered_norms.shape[1]),
                        "position": position + 1,
                        "grad_l2": float(gathered_norms[row_index, position].item()),
                    }
                    self._handle.write(json.dumps(row, separators=(",", ":")) + "\n")

        self._begin_update(boundary_epoch, training_epoch, optimizer_update)
        losses = {}
        for name in ("loss", "answer_loss", "cot_loss"):
            value = self._detached_scalar(outputs[name], device)
            losses[name] = accelerator.gather(value).detach().cpu().reshape(-1).tolist()
        answer_token_value = outputs.get("answer_token_count")
        if answer_token_value is None:
            answer_token_value = outputs["answer_token_counts"].detach().sum()
        cot_token_value = outputs.get("cot_token_count")
        if cot_token_value is None:
            cot_token_value = outputs["cot_token_counts"].detach().sum()
        answer_tokens = accelerator.gather(
            self._detached_scalar(answer_token_value, device)
        ).detach().cpu().reshape(-1).tolist()
        cot_tokens = accelerator.gather(
            self._detached_scalar(cot_token_value, device)
        ).detach().cpu().reshape(-1).tolist()
        example_counts = accelerator.gather(
            torch.tensor([batch_size], dtype=torch.float32, device=device)
        ).detach().cpu().reshape(-1).tolist()
        update = self._update
        update["micro_batches"] += 1
        update["num_examples"] += int(sum(example_counts))
        update["answer_tokens"] += int(sum(answer_tokens))
        update["cot_tokens"] += int(sum(cot_tokens))
        update["losses"].extend(float(value) for value in losses["loss"])
        update["answer_losses"].extend(float(value) for value in losses["answer_loss"])
        update["cot_losses"].extend(float(value) for value in losses["cot_loss"])
        if update_finished:
            if accelerator.is_main_process:
                self.optimizer_rows.append(
                    {
                        "method": update["method"],
                        "boundary_epoch": update["boundary_epoch"],
                        "training_epoch": update["training_epoch"],
                        "optimizer_update": update["optimizer_update"],
                        "micro_batches": update["micro_batches"],
                        "num_examples": update["num_examples"],
                        "answer_tokens": update["answer_tokens"],
                        "cot_tokens": update["cot_tokens"],
                        "loss": float(np.mean(update["losses"])),
                        "answer_loss": float(np.mean(update["answer_losses"])),
                        "cot_loss": float(np.mean(update["cot_losses"])),
                    }
                )
            self._update_key = None
            self._update = None
        if accelerator.is_main_process:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None


def _positive_integer(mapping: Mapping[str, Any], key: str) -> int:
    try:
        value = mapping[key]
    except (KeyError, TypeError):
        raise ValueError(f"{key} must be a positive integer") from None
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return int(value)


def build_stage_schedule(
    curriculum_config: Mapping[str, Any], latent_config: Mapping[str, Any]
) -> tuple[StagePoint, ...]:
    """Return the completed-epoch and first-trace-epoch point for each stage."""

    stage_epochs = _positive_integer(curriculum_config, "stage_epochs")
    latent_tokens_per_stage = _positive_integer(
        curriculum_config, "latent_tokens_per_stage"
    )
    num_latent_tokens = _positive_integer(latent_config, "num_latent_tokens")
    if num_latent_tokens % latent_tokens_per_stage:
        raise ValueError(
            "num_latent_tokens must be exactly divisible by "
            "latent_tokens_per_stage"
        )

    stage_count = num_latent_tokens // latent_tokens_per_stage
    return tuple(
        StagePoint(
            boundary_epoch=stage * stage_epochs,
            training_epoch=stage * stage_epochs + 1,
            curriculum_latent_count=(stage + 1) * latent_tokens_per_stage,
        )
        for stage in range(stage_count)
    )


def _load_yaml(path: str | Path) -> dict[str, Any]:
    parsed = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return parsed


def _lookup(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            dotted = ".".join(path)
            raise ValueError(f"missing paired configuration field: {dotted}")
        value = value[key]
    return value


def _resume_value(config: Mapping[str, Any]) -> Any:
    checkpoint = config.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        return None
    return checkpoint.get("resume_from_checkpoint")


def load_pair_configs(
    answer_path: str | Path, curriculum_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and minimally validate the Answer/Curriculum experiment pair."""

    answer = _load_yaml(answer_path)
    curriculum = _load_yaml(curriculum_path)
    if answer.get("method") != "answer":
        raise ValueError("answer config method must be answer")
    if curriculum.get("method") != "curriculum":
        raise ValueError("curriculum config method must be curriculum")

    for name, config in (("answer", answer), ("curriculum", curriculum)):
        resume = _resume_value(config)
        if resume not in (None, ""):
            raise ValueError(f"{name} checkpoint.resume_from_checkpoint must be empty")

    for path in PAIR_PATHS:
        answer_value = _lookup(answer, path)
        curriculum_value = _lookup(curriculum, path)
        if answer_value != curriculum_value:
            raise ValueError(
                "paired config mismatch for " + ".".join(path)
            )

    curriculum_stages = curriculum.get("curriculum")
    schedule = build_stage_schedule(curriculum_stages, curriculum["latent"])
    final_trace_epoch = schedule[-1].training_epoch
    configured_epochs = _lookup(answer, ("training", "num_train_epochs"))
    if (
        isinstance(configured_epochs, bool)
        or not isinstance(configured_epochs, Integral)
        or configured_epochs < final_trace_epoch
    ):
        raise ValueError(
            "training.num_train_epochs must reach final trace epoch "
            f"{final_trace_epoch}"
        )

    return answer, curriculum


def make_runtime_config(
    config: Mapping[str, Any], run_dir: str | Path
) -> dict[str, Any]:
    """Copy a config and redirect only analysis-owned runtime destinations."""

    runtime = copy.deepcopy(config)
    runtime["output_dir"] = str(run_dir)
    runtime.setdefault("training", {})["report_to"] = None
    checkpoint = runtime.setdefault("checkpoint", {})
    checkpoint["resume_from_checkpoint"] = None
    checkpoint["eval_checkpoint"] = None
    return runtime


def _trace_groups(
    rows: Iterable[Mapping[str, Any]],
) -> dict[int, dict[tuple[Any, ...], dict[int, float]]]:
    groups: dict[int, dict[tuple[Any, ...], dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        boundary = int(row["boundary_epoch"])
        unit = tuple(row[field] for field in _UNIT_KEY)
        position = int(row["position"])
        if position <= 0:
            raise ValueError("trace positions must be positive")
        unit_positions = groups[boundary][unit]
        if position in unit_positions:
            raise ValueError(
                "duplicate observation key in latent-position trace"
            )
        unit_positions[position] = float(row["grad_l2"])
    return {boundary: dict(units) for boundary, units in groups.items()}


def _trace_width(rows: Iterable[Mapping[str, Any]]) -> int:
    widths = [int(row["latent_count"]) for row in rows]
    if not widths or any(width <= 0 for width in widths):
        raise ValueError("trace latent_count must be positive")
    return max(widths)


def _matrix(
    units: Mapping[tuple[Any, ...], Mapping[int, float]],
    shared_units: Sequence[tuple[Any, ...]],
    width: int,
    boundary: int,
    series: str,
) -> np.ndarray:
    values = []
    for unit in shared_units:
        positions = units[unit]
        missing = [position for position in range(1, width + 1) if position not in positions]
        if missing:
            raise ValueError(
                f"paired {series} trace at boundary {boundary} is missing positions"
            )
        values.append([positions[position] for position in range(1, width + 1)])
    return np.asarray(values, dtype=np.float64)


def _summary_for_matrix(
    values: np.ndarray,
    bootstrap_indices: np.ndarray,
    boundary: int,
    series: str,
    latent_count: int,
) -> list[dict[str, Any]]:
    n = int(values.shape[0])
    means = values.mean(axis=0)
    if n == 1:
        standard_deviations = np.zeros(values.shape[1], dtype=np.float64)
        lows = means
        highs = means
    else:
        standard_deviations = values.std(axis=0, ddof=1)
        bootstrap_means = values[bootstrap_indices].mean(axis=1)
        lows, highs = np.quantile(bootstrap_means, [0.025, 0.975], axis=0)

    return [
        {
            "boundary_epoch": int(boundary),
            "series": series,
            "latent_count": int(latent_count),
            "position": position,
            "n": n,
            "mean": float(means[position - 1]),
            "std": float(standard_deviations[position - 1]),
            "ci95_low": float(lows[position - 1]),
            "ci95_high": float(highs[position - 1]),
        }
        for position in range(1, latent_count + 1)
    ]


def summarize_trace_rows(
    answer_rows: Iterable[Mapping[str, Any]],
    curriculum_rows: Iterable[Mapping[str, Any]],
    seed: int = 42,
    replicates: int = 1000,
) -> list[dict[str, Any]]:
    """Summarize paired local observations with a shared bootstrap sample."""

    if isinstance(replicates, bool) or not isinstance(replicates, Integral) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    answer_rows = list(answer_rows)
    curriculum_rows = list(curriculum_rows)
    if not answer_rows or not curriculum_rows:
        raise ValueError("paired trace inputs cannot be empty")
    answer_groups = _trace_groups(answer_rows)
    curriculum_groups = _trace_groups(curriculum_rows)
    answer_widths = {
        boundary: _trace_width(
            row for row in answer_rows if int(row["boundary_epoch"]) == boundary
        )
        for boundary in answer_groups
    }
    curriculum_widths = {
        boundary: _trace_width(
            row for row in curriculum_rows if int(row["boundary_epoch"]) == boundary
        )
        for boundary in curriculum_groups
    }

    rng = np.random.default_rng(seed)
    summaries: list[dict[str, Any]] = []
    for boundary in sorted(set(answer_groups) | set(curriculum_groups)):
        answer_units = answer_groups.get(boundary, {})
        curriculum_units = curriculum_groups.get(boundary, {})
        answer_unit_keys = set(answer_units)
        curriculum_unit_keys = set(curriculum_units)
        if answer_unit_keys != curriculum_unit_keys:
            raise ValueError(
                f"paired unit mismatch at boundary {boundary}"
            )
        shared_units = sorted(answer_unit_keys)
        answer_width = answer_widths[boundary]
        curriculum_width = curriculum_widths[boundary]
        overlap_width = min(answer_width, curriculum_width)
        if overlap_width <= 0:
            raise ValueError(
                f"paired delta cannot be formed at boundary {boundary}: no overlap"
            )

        answer_matrix = _matrix(
            answer_units, shared_units, answer_width, boundary, "answer"
        )
        curriculum_matrix = _matrix(
            curriculum_units,
            shared_units,
            curriculum_width,
            boundary,
            "curriculum",
        )
        bootstrap_indices = rng.integers(
            0,
            len(shared_units),
            size=(int(replicates), len(shared_units)),
        )
        summaries.extend(
            _summary_for_matrix(
                answer_matrix,
                bootstrap_indices,
                boundary,
                "answer",
                answer_width,
            )
        )
        summaries.extend(
            _summary_for_matrix(
                curriculum_matrix,
                bootstrap_indices,
                boundary,
                "curriculum",
                curriculum_width,
            )
        )
        delta_matrix = curriculum_matrix[:, :overlap_width] - answer_matrix[:, :overlap_width]
        summaries.extend(
            _summary_for_matrix(
                delta_matrix,
                bootstrap_indices,
                boundary,
                "curriculum_minus_answer",
                overlap_width,
            )
        )
    return summaries


def compute_log2_mean_ratios(
    summary_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute per-boundary Curriculum/Answer ratios over shared positions."""

    by_boundary: dict[int, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in summary_rows:
        series = row.get("series")
        if series not in {"answer", "curriculum"}:
            continue
        boundary = int(row["boundary_epoch"])
        position = int(row["position"])
        by_boundary[boundary][series][position] = float(row["mean"])

    ratios: list[dict[str, Any]] = []
    for boundary in sorted(by_boundary):
        answer = by_boundary[boundary].get("answer", {})
        curriculum = by_boundary[boundary].get("curriculum", {})
        positions = tuple(sorted(set(answer) & set(curriculum)))
        if not positions:
            continue
        answer_mean = sum(answer[position] for position in positions) / len(positions)
        curriculum_mean = sum(curriculum[position] for position in positions) / len(positions)
        if answer_mean <= 0 or curriculum_mean <= 0:
            raise ValueError(
                f"positive means are required to compute a ratio at boundary {boundary}"
            )
        ratio = curriculum_mean / answer_mean
        ratios.append(
            {
                "boundary_epoch": boundary,
                "positions": positions,
                "answer_mean": answer_mean,
                "curriculum_mean": curriculum_mean,
                "ratio": ratio,
                "log2_ratio": math.log2(ratio),
            }
        )
    return ratios


def _layout(stage_count: int) -> tuple[int, int]:
    if stage_count == 6:
        return 2, 3
    columns = max(1, math.ceil(math.sqrt(stage_count)))
    rows = math.ceil(stage_count / columns)
    return rows, columns


def write_position_figure(
    summary_rows: Iterable[Mapping[str, Any]],
    schedule: Sequence[StagePoint],
    pdf_path: str | Path,
    png_path: str | Path,
) -> None:
    """Write a shared-scale stage grid and ratio summary as PDF and PNG."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schedule = tuple(schedule)
    if not schedule:
        raise ValueError("schedule must contain at least one stage")
    summary_rows = list(summary_rows)
    styles = {
        "answer": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
        "curriculum": {"color": "#D55E00", "linestyle": "--", "marker": "s"},
    }
    by_boundary: dict[int, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in summary_rows:
        series = row.get("series")
        if series in styles:
            by_boundary[int(row["boundary_epoch"])][series].append(row)

    nrows, ncols = _layout(len(schedule))
    width = max(3.4, 3.0 * ncols)
    height = max(4.0, 2.5 * nrows + 1.35)
    pdf_path = Path(pdf_path)
    png_path = Path(png_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
    }
    with matplotlib.rc_context(rc):
        figure = plt.figure(figsize=(width, height))
        grid = figure.add_gridspec(
            nrows + 1,
            ncols,
            height_ratios=[1.0] * nrows + [0.55],
            left=0.10,
            right=0.98,
            bottom=0.09,
            top=0.86,
            hspace=0.70,
            wspace=0.25,
        )
        flat_axes = []
        for index, point in enumerate(schedule):
            row_index, column_index = divmod(index, ncols)
            axis = figure.add_subplot(
                grid[row_index, column_index],
                sharey=flat_axes[0] if flat_axes else None,
            )
            flat_axes.append(axis)
            for series in ("answer", "curriculum"):
                rows = sorted(
                    by_boundary.get(point.boundary_epoch, {}).get(series, []),
                    key=lambda row: int(row["position"]),
                )
                if not rows:
                    continue
                x = np.asarray([int(row["position"]) for row in rows])
                mean = np.asarray([float(row["mean"]) for row in rows])
                low = np.asarray([float(row["ci95_low"]) for row in rows])
                high = np.asarray([float(row["ci95_high"]) for row in rows])
                axis.plot(
                    x,
                    mean,
                    label=series.title(),
                    linewidth=1.2,
                    markersize=4,
                    **styles[series],
                )
                axis.fill_between(
                    x,
                    low,
                    high,
                    color=styles[series]["color"],
                    alpha=0.14,
                    linewidth=0,
                )
            axis.set_yscale("log")
            axis.set_title(
                f"Boundary {point.boundary_epoch} ({point.curriculum_latent_count} latent tokens)",
                pad=4,
            )
            axis.set_xlabel("Latent position")
            axis.grid(True, linestyle="-", linewidth=0.45, alpha=0.2)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_axisbelow(True)

        figure.supylabel(
            "Backward-scaled local latent gradient norm",
            fontsize=8,
            x=0.02,
        )
        ratio_rows = compute_log2_mean_ratios(summary_rows)
        summary_axis = figure.add_subplot(grid[nrows, :])
        summary_positions = np.arange(len(ratio_rows))
        summary_values = np.asarray(
            [float(row["log2_ratio"]) for row in ratio_rows], dtype=np.float64
        )
        summary_colors = [
            styles["curriculum"]["color"] if value >= 0 else styles["answer"]["color"]
            for value in summary_values
        ]
        bars = summary_axis.bar(
            summary_positions,
            summary_values,
            color=summary_colors,
            width=0.72,
            linewidth=0,
        )
        summary_axis.axhline(0, color="#333333", linewidth=0.7)
        summary_axis.set_ylabel(r"$\log_2$(Curriculum / Answer)", fontsize=7)
        summary_axis.set_xlabel("Stage boundary (completed epoch)", fontsize=7)
        summary_axis.set_xticks(
            summary_positions,
            [str(row["boundary_epoch"]) for row in ratio_rows],
        )
        summary_axis.tick_params(axis="both", labelsize=7)
        summary_axis.grid(True, axis="y", linestyle="-", linewidth=0.45, alpha=0.2)
        summary_axis.spines["top"].set_visible(False)
        summary_axis.spines["right"].set_visible(False)
        summary_axis.set_axisbelow(True)
        for bar, ratio_row in zip(bars, ratio_rows):
            value = float(bar.get_height())
            summary_axis.annotate(
                f"{ratio_row['ratio']:.2f}×",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 3 if value >= 0 else -3),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7,
            )
        figure.text(
            0.5,
            0.91,
            "Curves summarize the first K optimizer updates at each stage start.",
            ha="center",
            va="bottom",
            fontsize=7,
        )
        handles, labels = flat_axes[0].get_legend_handles_labels()
        if handles:
            figure.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=2,
                frameon=False,
            )
        figure.savefig(pdf_path, format="pdf", bbox_inches="tight")
        figure.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
        plt.close(figure)


def _prepare_output_directory(output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise FileExistsError(f"output directory must be absent or empty: {destination}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"output directory must be absent or empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _write_csv_partial(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _read_trace_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _checkpoint_metadata(stage_root: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    checkpoints = sorted(
        (
            path
            for path in stage_root.glob("epoch-*")
            if path.is_dir() and (path / ".complete").is_file()
        ),
        key=lambda path: int(path.name.rsplit("-", 1)[1]),
    )
    states = [
        json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        for path in checkpoints
    ]
    return checkpoints, states


def _manifest(
    *,
    destination: Path,
    answer_config_path: str | Path,
    curriculum_config_path: str | Path,
    answer_config: Mapping[str, Any],
    curriculum_config: Mapping[str, Any],
    run_records: Sequence[Mapping[str, Any]],
    schedule: Sequence[StagePoint],
    seed: int,
    trace_optimizer_steps: int,
    accelerator,
    prepared_loader_length: int | None,
    updates_per_epoch: int | None,
    artifacts: Mapping[str, Path],
) -> dict[str, Any]:
    training = answer_config["training"]
    distributed_type = getattr(accelerator, "distributed_type", "NO")
    schedule_rows = [
        {
            "boundary_epoch": point.boundary_epoch,
            "training_epoch": point.training_epoch,
            "curriculum_latent_count": point.curriculum_latent_count,
            "answer_latent_count": answer_config["latent"]["num_latent_tokens"],
        }
        for point in schedule
    ]
    return {
        "schema_version": 1,
        "analysis": "latent_position_loss",
        "command": list(sys.argv),
        "git_commit": _git_revision(),
        "input_configs": {
            "answer": {
                "path": str(Path(answer_config_path).resolve()),
                "config": answer_config,
            },
            "curriculum": {
                "path": str(Path(curriculum_config_path).resolve()),
                "config": curriculum_config,
            },
        },
        "seed": int(seed),
        "split_seed": 42,
        "trace_optimizer_steps": int(trace_optimizer_steps),
        "world_size": int(getattr(accelerator, "num_processes", 1)),
        "distributed_type": str(getattr(distributed_type, "value", distributed_type)),
        "training_complete": False,
        "batch": {
            "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
            "per_device_eval_batch_size": int(training["per_device_eval_batch_size"]),
            "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
            "bf16": bool(training["bf16"]),
            "even_batches": getattr(
                getattr(accelerator, "dataloader_config", None), "even_batches", None
            ),
            "prepared_train_loader_length": prepared_loader_length,
            "updates_per_epoch": updates_per_epoch,
        },
        "schedule": schedule_rows,
        "stage_count": len(schedule),
        "stage_epochs": int(curriculum_config["curriculum"]["stage_epochs"]),
        "latent_tokens_per_stage": int(
            curriculum_config["curriculum"]["latent_tokens_per_stage"]
        ),
        "runs": list(run_records),
        "loss_semantics": {
            "objective": "accelerator.backward(outputs['loss']) exactly once per micro-batch",
            "answer": "outputs['loss'] == answer_loss",
            "curriculum": "outputs['loss'] == answer_loss + cot_loss with real remaining-CoT supervision",
            "gradient": "backward-scaled local latent gradient norm; no second backward, replay, or post-hoc rescaling",
            "trace_window": "the first K optimizer updates at each stage boundary; the final trace epoch is partial",
        },
        "bootstrap": {
            "seed": 42,
            "replicates": 1000,
            "confidence_level": 0.95,
            "unit_key": list(_UNIT_KEY),
        },
        "partial_final_epoch": {
            "training_epoch": schedule[-1].training_epoch,
            "optimizer_updates_completed": int(trace_optimizer_steps),
            "completed_epoch": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "accelerate": _package_version("accelerate"),
            "numpy": np.__version__,
            "matplotlib": _package_version("matplotlib"),
        },
        "artifacts": {name: str(path.resolve()) for name, path in artifacts.items()},
        "output_dir": str(destination.resolve()),
    }


def run_position_analysis(
    answer_config: str | Path,
    curriculum_config: str | Path,
    trace_optimizer_steps: int,
    seed: int = 42,
    output_dir: str | Path = "outputs/analysis/latent_position_loss",
) -> dict[str, Path]:
    """Freshly train Answer and Curriculum, then publish paired trace outputs."""

    if (
        isinstance(trace_optimizer_steps, bool)
        or not isinstance(trace_optimizer_steps, Integral)
        or trace_optimizer_steps <= 0
    ):
        raise ValueError("trace_optimizer_steps must be a positive integer")
    destination = _prepare_output_directory(output_dir)
    answer, curriculum = load_pair_configs(answer_config, curriculum_config)
    schedule = build_stage_schedule(curriculum["curriculum"], curriculum["latent"])
    train_records, validation_records = load_train_validation(
        answer["data"]["train_path"],
        answer["data"]["validation_ratio"],
        max_train_samples=answer["data"].get("max_train_samples"),
        seed=42,
    )
    if not train_records or not validation_records:
        raise ValueError("train and validation splits must both be nonempty")
    prompt_template = load_prompt_template(answer["template_path"])

    from accelerate import Accelerator

    training = answer["training"]
    accelerator = Accelerator(
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        mixed_precision="bf16" if training["bf16"] else "no",
        log_with=None,
        project_dir=None,
        step_scheduler_with_optimizer=False,
    )
    traces_dir = destination / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    run_records: list[dict[str, Any]] = []
    optimizer_rows: list[dict[str, Any]] = []
    prepared_loader_length = None
    updates_per_epoch = None
    training_failed = False
    try:
        for method, config in (("answer", answer), ("curriculum", curriculum)):
            run_dir = destination / "runs" / method
            runtime_config = make_runtime_config(config, run_dir)
            partial_trace = traces_dir / f"{method}.jsonl.partial"
            writer = TraceWriter(method, partial_trace)
            result = None
            try:
                runtime = OnlineTraceRuntime(
                    boundary_by_epoch={
                        point.training_epoch: point.boundary_epoch
                        for point in schedule
                    },
                    optimizer_steps=int(trace_optimizer_steps),
                    seed=int(seed),
                    stage_checkpoint_dir=run_dir / "stage_ckpt",
                    on_backward=writer.on_backward,
                )
                result = train_indirect_from_records(
                    runtime_config,
                    accelerator,
                    train_records,
                    validation_records,
                    prompt_template,
                    runtime=runtime,
                )
                writer.close()
                latest_checkpoint, model, optimizer, scheduler, train_loader, validation_loader = result
                prepared_loader_length = len(train_loader)
                updates_per_epoch = math.ceil(
                    prepared_loader_length / training["gradient_accumulation_steps"]
                )
                checkpoints, trainer_states = _checkpoint_metadata(
                    run_dir / "stage_ckpt"
                )
                run_records.append(
                    {
                        "method": method,
                        "training_complete": False,
                        "trace_final_epoch": schedule[-1].training_epoch,
                        "effective_output_dir": str(run_dir.resolve()),
                        "runtime_config": runtime_config,
                        "trace_path": str(partial_trace.resolve()),
                        "latest_checkpoint": str(Path(latest_checkpoint).resolve()),
                        "checkpoint_paths": [str(path.resolve()) for path in checkpoints],
                        "trainer_states": trainer_states,
                        "stage_checkpoints": [
                            {
                                "path": str(path.resolve()),
                                "trainer_state": state,
                            }
                            for path, state in zip(checkpoints, trainer_states)
                        ],
                        "partial_final_epoch": {
                            "training_epoch": schedule[-1].training_epoch,
                            "optimizer_updates_completed": int(trace_optimizer_steps),
                            "completed_epoch": False,
                        },
                    }
                )
                optimizer_rows.extend(writer.optimizer_rows)
                resources = accelerator.free_memory(
                    model, optimizer, scheduler, train_loader, validation_loader
                )
                del resources, model, optimizer, scheduler, train_loader, validation_loader, result
                gc.collect()
            finally:
                writer.close()
    except BaseException:
        training_failed = True
        raise
    finally:
        try:
            accelerator.wait_for_everyone()
        finally:
            if training_failed:
                accelerator.end_training()

    artifacts = {
        "runs_path": destination / "runs",
        "traces_path": traces_dir,
        "optimizer_updates_path": destination / "optimizer_updates.csv",
        "per_example_position_path": destination / "per_example_position.csv",
        "summary_path": destination / "summary.csv",
        "pdf_path": destination / "latent_position_loss.pdf",
        "png_path": destination / "latent_position_loss.png",
        "manifest_path": destination / "manifest.json",
    }
    try:
        if accelerator.is_main_process:
            answer_trace = traces_dir / "answer.jsonl.partial"
            curriculum_trace = traces_dir / "curriculum.jsonl.partial"
            answer_rows = _read_trace_rows(answer_trace)
            curriculum_rows = _read_trace_rows(curriculum_trace)
            optimizer_partial = destination / "optimizer_updates.csv.partial"
            per_example_partial = destination / "per_example_position.csv.partial"
            summary_partial = destination / "summary.csv.partial"
            pdf_partial = destination / "latent_position_loss.pdf.partial"
            png_partial = destination / "latent_position_loss.png.partial"
            _write_csv_partial(optimizer_partial, optimizer_rows, OPTIMIZER_FIELDS)
            _write_csv_partial(
                per_example_partial,
                answer_rows + curriculum_rows,
                PER_EXAMPLE_FIELDS,
            )
            summary_rows = summarize_trace_rows(
                answer_rows, curriculum_rows, seed=42, replicates=1000
            )
            _write_csv_partial(summary_partial, summary_rows, (
                "boundary_epoch",
                "series",
                "latent_count",
                "position",
                "n",
                "mean",
                "std",
                "ci95_low",
                "ci95_high",
            ))
            write_position_figure(summary_rows, schedule, pdf_partial, png_partial)
            for source, target in (
                (optimizer_partial, artifacts["optimizer_updates_path"]),
                (per_example_partial, artifacts["per_example_position_path"]),
                (summary_partial, artifacts["summary_path"]),
                (pdf_partial, artifacts["pdf_path"]),
                (png_partial, artifacts["png_path"]),
                (answer_trace, traces_dir / "answer.jsonl"),
                (curriculum_trace, traces_dir / "curriculum.jsonl"),
            ):
                source.replace(target)
            for record in run_records:
                record["trace_path"] = str(
                    (traces_dir / f"{record['method']}.jsonl").resolve()
                )
            manifest = _manifest(
                destination=destination,
                answer_config_path=answer_config,
                curriculum_config_path=curriculum_config,
                answer_config=answer,
                curriculum_config=curriculum,
                run_records=run_records,
                schedule=schedule,
                seed=int(seed),
                trace_optimizer_steps=int(trace_optimizer_steps),
                accelerator=accelerator,
                prepared_loader_length=prepared_loader_length,
                updates_per_epoch=updates_per_epoch,
                artifacts=artifacts,
            )
            manifest_partial = destination / "manifest.json.partial"
            manifest_partial.write_text(
                json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
            )
            manifest_partial.replace(artifacts["manifest_path"])
        accelerator.wait_for_everyone()
    finally:
        accelerator.end_training()
    return artifacts


def _positive_cli_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Freshly train paired Answer/Curriculum models and trace latent gradients."
    )
    parser.add_argument("--answer-config", required=True)
    parser.add_argument("--curriculum-config", required=True)
    parser.add_argument(
        "--trace-optimizer-steps", required=True, type=_positive_cli_integer
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", default="outputs/analysis/latent_position_loss"
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    paths = run_position_analysis(
        args.answer_config,
        args.curriculum_config,
        args.trace_optimizer_steps,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank == 0:
        print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


__all__ = [
    "OBSERVATION_KEY",
    "OPTIMIZER_FIELDS",
    "PAIR_PATHS",
    "PER_EXAMPLE_FIELDS",
    "StagePoint",
    "TRACE_FIELDS",
    "TraceWriter",
    "build_stage_schedule",
    "compute_log2_mean_ratios",
    "load_pair_configs",
    "make_runtime_config",
    "main",
    "parse_args",
    "run_position_analysis",
    "summarize_trace_rows",
    "write_position_figure",
]


if __name__ == "__main__":
    main()
