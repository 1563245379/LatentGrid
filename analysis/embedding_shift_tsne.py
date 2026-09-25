import argparse
import json
import csv
import gc
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src.compression import load_compression_reasoner
from src.data import load_prompt_template, load_train_validation, render_prompt
from src.evaluation import ANSWER_MARKER
from src.latent_model import (
    START_LATENT_TOKEN,
    load_latent_reasoner,
    position_ids_from_mask,
)
from src.model import generate_continuations, load_cot_model


SNAPSHOT_PATHS = (
    ("method",),
    ("latent_representation_type",),
    ("base_model",),
    ("template_path",),
    ("latent",),
    ("data", "train_path"),
    ("data", "validation_ratio"),
    ("checkpoint", "cot_init_checkpoint"),
)

PAIR_PATHS = (
    ("method",),
    ("base_model",),
    ("template_path",),
    ("data", "train_path"),
    ("data", "validation_ratio"),
    ("checkpoint", "cot_init_checkpoint"),
    ("latent", "num_latent_tokens"),
)

FIXED_METHODS = frozenset(
    {"answer", "curriculum", "reconstruction", "curriculum_reconstruction"}
)
ADAPTIVE_METHODS = frozenset(
    {
        "compression",
        "curriculum_compression",
        "token_reconstruction",
        "curriculum_token_reconstruction",
    }
)
WRAPPED_METHODS = frozenset(
    {"compression", "curriculum_compression"}
)
SUPPORTED_METHODS = FIXED_METHODS | ADAPTIVE_METHODS
MMD_BANDWIDTH_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)


def _mmd_vectors(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or 0 in vectors.shape:
        raise ValueError("MMD vectors must be a nonempty rank-two array")
    if not np.isfinite(vectors).all():
        raise ValueError("MMD vectors must be finite")
    return vectors


def _positive_bandwidths(values):
    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != 1 or not values.size
        or not np.isfinite(values).all() or np.any(values <= 0)
    ):
        raise ValueError("MMD bandwidths and multipliers must be finite and positive")
    return values


def _squared_distances(first, second):
    distances = (
        np.sum(first * first, axis=1)[:, None]
        + np.sum(second * second, axis=1)[None, :]
        - 2 * (first @ second.T)
    )
    return np.maximum(distances, 0.0)


def median_rbf_bandwidths(
    vectors, multipliers=MMD_BANDWIDTH_MULTIPLIERS, max_points=2000, seed=42
):
    """Scale the median positive off-diagonal Euclidean distance in a pooled sample.

    Sampling limits bandwidth-selection cost only. Duplicate points are excluded
    from the distance median; if all points coincide, use base sigma = 1.
    """
    vectors = _mmd_vectors(vectors)
    multipliers = _positive_bandwidths(multipliers)
    if max_points < 2:
        raise ValueError("MMD bandwidth max_points must be at least 2")
    if len(vectors) > max_points:
        indices = np.random.default_rng(seed).choice(
            len(vectors), max_points, replace=False
        )
        vectors = vectors[indices]
    distances = _squared_distances(vectors, vectors)
    # Identical rows must have distance zero, including after L2 normalization.
    _, inverse = np.unique(vectors, axis=0, return_inverse=True)
    distances[inverse[:, None] == inverse[None, :]] = 0.0
    distances = distances[np.triu_indices(len(vectors), k=1)]
    positive = np.sqrt(distances[distances > 0])
    median = float(np.median(positive)) if positive.size else 1.0
    return median * multipliers


def biased_mmd_squared(latents, reference, bandwidths):
    """Paper's biased MMD^2 with an equally weighted multi-bandwidth RBF kernel.

    Inputs are used as supplied; compute_question_mmd applies L2 normalization.
    Self-kernel means include diagonal terms and use n^2 and m^2 denominators.
    """
    latents, reference = _mmd_vectors(latents), _mmd_vectors(reference)
    bandwidths = _positive_bandwidths(bandwidths)
    if latents.shape[1] != reference.shape[1]:
        raise ValueError("MMD latent and reference hidden widths must match")

    def kernel_mean(first, second):
        distances = _squared_distances(first, second)
        if first is second:
            np.fill_diagonal(distances, 0.0)
        return np.mean([
            np.exp(-distances / (2 * sigma**2)).mean() for sigma in bandwidths
        ])

    value = (
        kernel_mean(latents, latents) + kernel_mean(reference, reference)
        - 2 * kernel_mean(latents, reference)
    )
    # The biased estimator is nonnegative; cancellation can leave tiny negatives.
    return float(max(value, 0.0))


def _validate_mmd_options(bandwidth_multipliers, bandwidth_max_points, bootstrap_resamples):
    _positive_bandwidths(bandwidth_multipliers)
    if bandwidth_max_points < 2:
        raise ValueError("MMD bandwidth max_points must be at least 2")
    if bootstrap_resamples < 1:
        raise ValueError("MMD bootstrap_resamples must be positive")


def compute_question_mmd(
    unconstrained, constrained, reference_bundle, example_ids,
    bandwidth_multipliers=MMD_BANDWIDTH_MULTIPLIERS,
    bandwidth_max_points=2000, bootstrap_resamples=10000, seed=42,
):
    """Compare complete matched trajectories before t-SNE sampling or projection.

    A single kernel is selected from pooled, L2-normalized CoT/U/C vectors for
    this run pair and held fixed during bootstrap. Questions receive equal
    weight regardless of trajectory length. U/C share bootstrap question draws.
    """
    _validate_mmd_options(bandwidth_multipliers, bandwidth_max_points, bootstrap_resamples)
    unconstrained = np.asarray(unconstrained, dtype=np.float64)
    constrained = np.asarray(constrained, dtype=np.float64)
    if (
        unconstrained.ndim != 3 or 0 in unconstrained.shape
        or unconstrained.shape != constrained.shape
    ):
        raise ValueError("paired MMD latent arrays must have the same nonempty rank-three shape")
    example_ids = np.asarray(example_ids)
    if (
        example_ids.shape != (len(unconstrained),)
        or len(np.unique(example_ids)) != len(example_ids)
    ):
        raise ValueError("unique example IDs must align with MMD latent rows")
    reference = _mmd_vectors(reference_bundle["vectors"])
    reference_ids = np.asarray(reference_bundle["example_id"])
    if (
        reference_ids.shape != (len(reference),)
        or set(reference_ids.tolist()) != set(example_ids.tolist())
    ):
        raise ValueError("MMD reference example IDs must match the latent example IDs")
    hidden_size = unconstrained.shape[-1]
    if reference.shape[1] != hidden_size:
        raise ValueError("MMD latent and reference hidden widths must match")

    def normalize(vectors):
        vectors = _mmd_vectors(vectors.reshape(-1, hidden_size))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Keep zero vectors at zero, as in standard L2 normalization.
        return vectors / np.where(norms > 0, norms, 1.0)

    latent_shape = unconstrained.shape
    unconstrained = normalize(unconstrained).reshape(latent_shape)
    constrained = normalize(constrained).reshape(latent_shape)
    reference = normalize(reference)
    pooled = np.concatenate([
        reference, unconstrained.reshape(-1, hidden_size),
        constrained.reshape(-1, hidden_size),
    ])
    pooled_count = len(pooled)
    bandwidths = median_rbf_bandwidths(
        pooled, bandwidth_multipliers, bandwidth_max_points, seed
    )
    del pooled
    per_question = []
    for index, example_id in enumerate(example_ids):
        cot = reference[reference_ids == example_id]
        per_question.append({
            "example_id": int(example_id),
            "num_latents": int(latent_shape[1]),
            "num_cot_tokens": len(cot),
            "unconstrained_mmd2": biased_mmd_squared(unconstrained[index], cot, bandwidths),
            "constrained_mmd2": biased_mmd_squared(constrained[index], cot, bandwidths),
        })
    values = np.array([
        [row["unconstrained_mmd2"], row["constrained_mmd2"]] for row in per_question
    ])
    indices = np.random.default_rng(seed).integers(
        len(values), size=(bootstrap_resamples, len(values))
    )
    intervals = np.quantile(values[indices].mean(axis=1), [0.025, 0.975], axis=0)
    report = {
        "estimator": "biased_mmd_squared",
        "normalization": "l2_per_vector; zero vectors remain zero",
        "aggregation": "equal_weight_question_mean",
        "num_questions": len(per_question),
        "kernel": {
            "name": "equally_weighted_multi_bandwidth_rbf",
            "bandwidths": bandwidths.tolist(),
            "bandwidth_multipliers": list(bandwidth_multipliers),
            "selection": "median_positive_off_diagonal_euclidean_distance",
            "zero_distance_fallback_sigma": 1.0,
            "pool": "matched_cot_unconstrained_constrained",
            "pooled_points": pooled_count,
            "sampled_points": min(pooled_count, bandwidth_max_points),
            "max_points": bandwidth_max_points,
            "seed": seed,
        },
        "bootstrap": {
            "unit": "question", "method": "percentile", "confidence_level": 0.95,
            "resamples": bootstrap_resamples, "seed": seed,
            "paired_resampling": True, "fixed_bandwidths": True,
        },
        "per_question": per_question,
    }
    for column, name in enumerate(("unconstrained", "constrained")):
        report[name] = {
            "mean_mmd2": float(values[:, column].mean()),
            "ci95": intervals[:, column].tolist(),
        }
    return report


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def nested(config, path):
    value = config
    for key in path:
        value = value[key]
    return value


def extract_generated_cot(generation):
    prefix, marker, _ = generation.partition(ANSWER_MARKER)
    if not marker:
        return None
    cot = prefix.strip()
    return cot or None


def load_analysis_run(config_path, checkpoint_override, expected_representation):
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    if config["method"] not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported embedding t-SNE method: {config['method']}")
    if config["latent_representation_type"] != expected_representation:
        raise ValueError(f"expected representation {expected_representation}")
    selected = checkpoint_override or config["checkpoint"]["eval_checkpoint"]
    if selected is None:
        raise ValueError("checkpoint override or checkpoint.eval_checkpoint is required")
    checkpoint_path = Path(selected).resolve()
    if not (checkpoint_path / ".complete").is_file():
        raise ValueError(f"checkpoint is not complete: {checkpoint_path}")
    checkpoint_config = load_yaml(checkpoint_path / "config.yaml")
    for path in SNAPSHOT_PATHS:
        if nested(config, path) != nested(checkpoint_config, path):
            raise ValueError(f"checkpoint config mismatch for {'.'.join(path)}")
    trainer_state = json.loads(
        (checkpoint_path / "trainer_state.json").read_text(encoding="utf-8")
    )
    return {
        "config_path": config_path,
        "config": config,
        "checkpoint_path": checkpoint_path,
        "checkpoint_config": checkpoint_config,
        "trainer_state": trainer_state,
        "training_complete": (
            trainer_state["completed_epoch"]
            == config["training"]["num_train_epochs"]
        ),
    }


def validate_tsne_runs(unconstrained_run, constrained_run):
    unconstrained = unconstrained_run["config"]
    constrained = constrained_run["config"]
    if unconstrained["latent_representation_type"] != "unconstrained":
        raise ValueError("first run must be unconstrained")
    if constrained["latent_representation_type"] != "constrained":
        raise ValueError("second run must be constrained")
    for path in PAIR_PATHS:
        if nested(unconstrained, path) != nested(constrained, path):
            raise ValueError(f"t-SNE run mismatch for {'.'.join(path)}")


def load_analysis_reasoner(config, checkpoint_path):
    loaders = {
        **{method: load_latent_reasoner for method in FIXED_METHODS},
        "compression": load_compression_reasoner,
        "curriculum_compression": load_compression_reasoner,
        "token_reconstruction": load_latent_reasoner,
        "curriculum_token_reconstruction": load_latent_reasoner,
    }
    return loaders[config["method"]](
        config, checkpoint_path=checkpoint_path, is_trainable=False
    )


def select_validation_records(config, num_samples):
    _, validation = load_train_validation(
        config["data"]["train_path"],
        config["data"]["validation_ratio"],
        max_train_samples=None,
        seed=42,
    )
    if not 1 <= num_samples <= len(validation):
        raise ValueError("num_samples must fit inside the validation split")
    return validation[:num_samples]


def build_prompt_batch(run, tokenizer, records):
    config = run["config"]
    prompt_template = load_prompt_template(config["template_path"])
    start_id = tokenizer.convert_tokens_to_ids(START_LATENT_TOKEN)
    prompt_ids = [
        tokenizer.encode(
            render_prompt(prompt_template, record["question"]),
            add_special_tokens=False,
        )
        + [start_id]
        for record in records
    ]
    width = max(len(ids) for ids in prompt_ids)
    padded_ids = []
    attention_masks = []
    for ids in prompt_ids:
        padding = width - len(ids)
        padded_ids.append([tokenizer.pad_token_id] * padding + ids)
        attention_masks.append([0] * padding + [1] * len(ids))
    return {
        "prompt_input_ids": torch.tensor(padded_ids),
        "prompt_attention_mask": torch.tensor(attention_masks),
    }


def move_batch(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.inference_mode()
def rollout_fixed_latents(
    model,
    method,
    prompt_input_ids,
    prompt_attention_mask,
    latent_count,
):
    reasoner = model.reasoner if method in WRAPPED_METHODS else model
    outputs = reasoner.llm(
        inputs_embeds=reasoner.embed(prompt_input_ids),
        attention_mask=prompt_attention_mask,
        position_ids=position_ids_from_mask(prompt_attention_mask),
        output_hidden_states=True,
        use_cache=True,
    )
    attention_mask = prompt_attention_mask
    latents = []
    for _ in range(latent_count):
        latent = reasoner.next_latent(
            outputs.hidden_states[-1][:, -1], outputs.logits[:, -1]
        )
        if method in WRAPPED_METHODS and reasoner.representation_type == "unconstrained":
            latent = latent * model.embedding_scale.to(latent.dtype)
        latents.append(latent)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1
        )
        outputs = reasoner.llm(
            inputs_embeds=latent.unsqueeze(1),
            attention_mask=attention_mask,
            position_ids=position_ids_from_mask(attention_mask)[:, -1:],
            past_key_values=outputs.past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
    return torch.stack(latents, dim=1)


def collect_cot_reference_vectors(
    run,
    records,
    batch_size,
    max_new_tokens,
    model_loader=load_cot_model,
    continuation_generator=generate_continuations,
):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    config = run["config"]
    model, tokenizer = model_loader(
        config["base_model"], config["checkpoint"]["cot_init_checkpoint"]
    )
    prompt_template = load_prompt_template(config["template_path"])
    prompts = [render_prompt(prompt_template, record["question"]) for record in records]
    generations = continuation_generator(
        model,
        tokenizer,
        prompts,
        batch_size,
        max_new_tokens,
    )
    if len(generations) != len(records):
        raise ValueError("CoT generations must align one-to-one with records")

    embedding = model.get_input_embeddings()
    original_vocab_size, hidden_size = embedding.weight.shape
    vectors = []
    example_ids = []
    positions = []
    token_ids = []
    kept_example_ids = []
    dropped_example_ids = []
    for record, generation in zip(records, generations):
        cot = extract_generated_cot(generation)
        encoded = tokenizer.encode(cot, add_special_tokens=False) if cot else []
        kept = [
            (position, token_id)
            for position, token_id in enumerate(encoded, start=1)
            if token_id < original_vocab_size
            and token_id != tokenizer.eos_token_id
            and token_id != tokenizer.pad_token_id
        ]
        if not kept:
            dropped_example_ids.append(record["example_id"])
            continue
        ids = [token_id for _, token_id in kept]
        ids_tensor = torch.tensor(ids, dtype=torch.long, device=embedding.weight.device)
        values = embedding(ids_tensor).detach().float().cpu().numpy()
        vectors.extend(values)
        example_ids.extend([record["example_id"]] * len(ids))
        positions.extend(position for position, _ in kept)
        token_ids.extend(ids)
        kept_example_ids.append(record["example_id"])

    result = {
        "vectors": np.asarray(vectors, dtype=np.float32),
        "example_id": np.asarray(example_ids, dtype=np.int64),
        "source_position": np.asarray(positions, dtype=np.int32),
        "token_id": np.asarray(token_ids, dtype=np.int32),
        "kept_example_ids": np.asarray(kept_example_ids, dtype=np.int64),
        "dropped_example_ids": np.asarray(dropped_example_ids, dtype=np.int64),
        "original_vocab_size": int(original_vocab_size),
        "hidden_size": int(hidden_size),
        "device": str(embedding.weight.device),
        "model_dtype": str(embedding.weight.dtype).removeprefix("torch."),
    }
    no_usable_references = not kept_example_ids
    del embedding, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if no_usable_references:
        raise ValueError("no usable generated CoT references")
    return result


def collect_run_vectors(
    run, records, reasoner_loader=load_analysis_reasoner, device=None
):
    config = run["config"]
    model, tokenizer = reasoner_loader(config, run["checkpoint_path"])
    resolved_device = device or next(model.parameters()).device
    if device is not None:
        model.to(resolved_device)
    model.eval()
    reasoner = model.reasoner if config["method"] in WRAPPED_METHODS else model
    batches = []
    batch_size = config["training"]["per_device_eval_batch_size"]
    latent_count = config["latent"]["num_latent_tokens"]
    for offset in range(0, len(records), batch_size):
        batch_records = records[offset : offset + batch_size]
        batch = move_batch(
            build_prompt_batch(run, tokenizer, batch_records), resolved_device
        )
        latents = rollout_fixed_latents(
            model,
            config["method"],
            batch["prompt_input_ids"],
            batch["prompt_attention_mask"],
            latent_count,
        )
        batches.append(latents.float().cpu().numpy())
    embedding = reasoner.llm.get_input_embeddings().weight
    result = {
        "latents": np.concatenate(batches, axis=0).astype(np.float32, copy=False),
        "original_vocab_size": int(reasoner.original_vocab_size),
        "hidden_size": int(embedding.shape[1]),
        "device": str(embedding.device),
        "model_dtype": str(embedding.dtype).removeprefix("torch."),
    }
    del embedding, latents, batch, batch_records, batches, reasoner, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def validate_vector_spaces(
    unconstrained_bundle, constrained_bundle, cot_reference_bundle
):
    for key in ("original_vocab_size", "hidden_size"):
        if unconstrained_bundle[key] != constrained_bundle[key]:
            raise ValueError(f"collected vector space mismatch for {key}")
        if unconstrained_bundle[key] != cot_reference_bundle[key]:
            raise ValueError(f"CoT reference vector space mismatch for {key}")
    if unconstrained_bundle["latents"].shape != constrained_bundle["latents"].shape:
        raise ValueError("paired latent arrays must have the same shape")


def balanced_sample(
    unconstrained, constrained, reference_bundle, example_ids, points_per_group, seed
):
    if unconstrained.shape != constrained.shape:
        raise ValueError("paired latent arrays must have the same shape")
    if points_per_group < 1:
        raise ValueError("points_per_group must be positive")
    num_examples, latent_count, hidden_size = unconstrained.shape
    if len(example_ids) != num_examples:
        raise ValueError("example IDs must align with latent rows")
    if reference_bundle["vectors"].ndim != 2:
        raise ValueError("CoT reference vectors must be a rank-two array")
    if reference_bundle["vectors"].shape[1] != hidden_size:
        raise ValueError("CoT reference and latent hidden widths must match")
    reference_count = len(reference_bundle["vectors"])
    for key in ("example_id", "source_position", "token_id"):
        if len(reference_bundle[key]) != reference_count:
            raise ValueError(f"CoT reference metadata is misaligned for {key}")
    flat_unconstrained = unconstrained.reshape(-1, hidden_size)
    flat_constrained = constrained.reshape(-1, hidden_size)
    latent_example_id = np.repeat(np.asarray(example_ids, dtype=np.int64), latent_count)
    latent_position = np.tile(
        np.arange(1, latent_count + 1, dtype=np.int32), num_examples
    )
    point_count = min(points_per_group, len(flat_unconstrained), reference_count)
    if point_count < 1:
        raise ValueError("at least one latent and CoT reference point is required")
    rng = np.random.default_rng(seed)
    latent_indices = rng.choice(len(flat_unconstrained), point_count, replace=False)
    reference_indices = rng.choice(reference_count, point_count, replace=False)
    vectors = np.concatenate(
        [
            reference_bundle["vectors"][reference_indices],
            flat_unconstrained[latent_indices],
            flat_constrained[latent_indices],
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    return {
        "vectors": vectors,
        "class_id": np.repeat(np.arange(3, dtype=np.int8), point_count),
        "source_id": np.concatenate(
            [np.zeros(point_count, dtype=np.int8), np.ones(point_count * 2, dtype=np.int8)]
        ),
        "example_id": np.concatenate(
            [
                reference_bundle["example_id"][reference_indices],
                latent_example_id[latent_indices],
                latent_example_id[latent_indices],
            ]
        ),
        "source_position": np.concatenate(
            [
                reference_bundle["source_position"][reference_indices],
                latent_position[latent_indices],
                latent_position[latent_indices],
            ]
        ),
        "token_id": np.concatenate(
            [
                reference_bundle["token_id"][reference_indices],
                np.full(point_count * 2, -1, dtype=np.int32),
            ]
        ),
    }


CLASS_NAMES = np.array(
    ["cot_reference", "latent_unconstrained", "latent_constrained"]
)
SOURCE_NAMES = np.array(["cot", "latent"])


def joint_tsne(vectors, perplexity, seed, tsne_factory=None):
    if perplexity >= len(vectors):
        raise ValueError("perplexity must be smaller than the total point count")
    if tsne_factory is None:
        from sklearn.manifold import TSNE

        tsne_factory = TSNE
    reducer = tsne_factory(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    )
    return reducer.fit_transform(vectors).astype(np.float32, copy=False)


def write_points_csv(path, sampled, coordinates):
    point_count = len(sampled["vectors"])
    if coordinates.shape != (point_count, 2):
        raise ValueError("coordinates must align one-to-one with sampled vectors")
    for key in ("class_id", "source_id", "example_id", "source_position", "token_id"):
        if len(sampled[key]) != point_count:
            raise ValueError(f"sample metadata is misaligned for {key}")
    fieldnames = (
        "class",
        "source",
        "example_id",
        "source_position",
        "token_id",
        "x",
        "y",
    )
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, coordinate in enumerate(coordinates):
            writer.writerow(
                {
                    "class": CLASS_NAMES[sampled["class_id"][index]],
                    "source": SOURCE_NAMES[sampled["source_id"][index]],
                    "example_id": int(sampled["example_id"][index]),
                    "source_position": int(sampled["source_position"][index]),
                    "token_id": int(sampled["token_id"][index]),
                    "x": float(coordinate[0]),
                    "y": float(coordinate[1]),
                }
            )


def write_embedding_pdf(coordinates, class_id, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (point_count, 2)")
    if len(class_id) != len(coordinates):
        raise ValueError("class metadata must align one-to-one with coordinates")
    colors = ("#0072B2", "#D55E00", "#009E73")
    markers = ("o", "^", "s")
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for class_value, name in enumerate(CLASS_NAMES):
        mask = class_id == class_value
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=10,
            alpha=0.45,
            color=colors[class_value],
            marker=markers[class_value],
            label=name,
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def prepare_output_directory(output_dir):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    entries = list(destination.iterdir())
    if entries:
        raise ValueError(
            "output directory must be empty; existing entries: "
            + ", ".join(sorted(entry.name for entry in entries))
        )
    return destination


def package_version(distribution):
    return importlib.metadata.version(distribution)


def git_commit():
    repository = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def manifest_run(run, bundle):
    config = run["config"]
    return {
        "config_path": str(run["config_path"]),
        "checkpoint_path": str(run["checkpoint_path"]),
        "trainer_state": run["trainer_state"],
        "training_complete": run["training_complete"],
        "base_model": config["base_model"],
        "method": config["method"],
        "representation": config["latent_representation_type"],
        "latent": config["latent"],
        "configured_dtype": "bfloat16" if config["training"]["bf16"] else "float32",
        "actual_model_dtype": bundle["model_dtype"],
        "device": bundle["device"],
        "original_vocab_size": bundle["original_vocab_size"],
        "hidden_size": bundle["hidden_size"],
    }


def build_manifest(
    unconstrained_run,
    constrained_run,
    unconstrained_bundle,
    constrained_bundle,
    cot_reference_bundle,
    records,
    sampled,
    perplexity,
    seed,
    cot_batch_size,
    cot_max_new_tokens,
    argv,
):
    config = unconstrained_run["config"]
    points_per_group = len(sampled["vectors"]) // 3
    return {
        "analysis": "embedding_shift_tsne",
        "command": argv,
        "git_commit": git_commit(),
        "example_ids": [int(record["example_id"]) for record in records],
        "data": {
            "path": config["data"]["train_path"],
            "validation_ratio": config["data"]["validation_ratio"],
            "split_seed": 42,
        },
        "cot_reference": {
            "base_model": config["base_model"],
            "adapter_path": config["checkpoint"]["cot_init_checkpoint"],
            "batch_size": cot_batch_size,
            "max_new_tokens": cot_max_new_tokens,
            "do_sample": False,
            "answer_marker": ANSWER_MARKER,
            "kept_example_ids": cot_reference_bundle[
                "kept_example_ids"
            ].tolist(),
            "dropped_example_ids": cot_reference_bundle[
                "dropped_example_ids"
            ].tolist(),
            "device": cot_reference_bundle["device"],
            "model_dtype": cot_reference_bundle["model_dtype"],
            "original_vocab_size": cot_reference_bundle["original_vocab_size"],
            "hidden_size": cot_reference_bundle["hidden_size"],
        },
        "runs": [
            manifest_run(unconstrained_run, unconstrained_bundle),
            manifest_run(constrained_run, constrained_bundle),
        ],
        "tsne": {
            "points_per_group_actual": points_per_group,
            "sampling_seed": seed,
            "n_components": 2,
            "perplexity": perplexity,
            "init": "pca",
            "learning_rate": "auto",
            "max_iter": 1000,
            "random_state": seed,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "matplotlib": package_version("matplotlib"),
            "scikit_learn": package_version("scikit-learn"),
        },
        "interpretation": (
            "Qualitative local-neighborhood visualization only; two-dimensional "
            "inter-cluster distances are not original-space distribution distances "
            "or significance tests."
        ),
    }


def run_embedding_tsne(
    unconstrained_config,
    constrained_config,
    unconstrained_checkpoint=None,
    constrained_checkpoint=None,
    num_samples=128,
    points_per_group=1000,
    perplexity=30,
    seed=42,
    cot_batch_size=16,
    cot_max_new_tokens=256,
    output_dir="outputs/analysis/embedding_shift_tsne",
    tsne_factory=None,
    mmd_bandwidth_multipliers=MMD_BANDWIDTH_MULTIPLIERS,
    mmd_bandwidth_max_points=2000,
    mmd_bootstrap_resamples=10000,
):
    _validate_mmd_options(
        mmd_bandwidth_multipliers, mmd_bandwidth_max_points, mmd_bootstrap_resamples
    )
    destination = prepare_output_directory(output_dir)
    unconstrained_run = load_analysis_run(
        unconstrained_config, unconstrained_checkpoint, "unconstrained"
    )
    constrained_run = load_analysis_run(
        constrained_config, constrained_checkpoint, "constrained"
    )
    validate_tsne_runs(unconstrained_run, constrained_run)
    candidate_records = select_validation_records(
        unconstrained_run["config"], num_samples
    )
    cot_reference_bundle = collect_cot_reference_vectors(
        unconstrained_run,
        candidate_records,
        cot_batch_size,
        cot_max_new_tokens,
    )
    kept_example_ids = set(cot_reference_bundle["kept_example_ids"].tolist())
    records = [
        record
        for record in candidate_records
        if record["example_id"] in kept_example_ids
    ]
    unconstrained_bundle = collect_run_vectors(unconstrained_run, records)
    constrained_bundle = collect_run_vectors(constrained_run, records)
    validate_vector_spaces(
        unconstrained_bundle, constrained_bundle, cot_reference_bundle
    )
    mmd = compute_question_mmd(
        unconstrained_bundle["latents"],
        constrained_bundle["latents"],
        cot_reference_bundle,
        [record["example_id"] for record in records],
        bandwidth_multipliers=mmd_bandwidth_multipliers,
        bandwidth_max_points=mmd_bandwidth_max_points,
        bootstrap_resamples=mmd_bootstrap_resamples,
        seed=seed,
    )
    sampled = balanced_sample(
        unconstrained_bundle["latents"],
        constrained_bundle["latents"],
        cot_reference_bundle,
        [record["example_id"] for record in records],
        points_per_group,
        seed,
    )
    coordinates = joint_tsne(sampled["vectors"], perplexity, seed, tsne_factory)
    paths = {
        "npz_path": destination / "embeddings.npz",
        "points_path": destination / "points.csv",
        "manifest_path": destination / "manifest.json",
        "pdf_path": destination / "embedding_shift_tsne.pdf",
        "mmd_path": destination / "mmd.json",
    }
    np.savez(
        paths["npz_path"],
        vectors=sampled["vectors"],
        coordinates=coordinates,
        class_id=sampled["class_id"],
        source_id=sampled["source_id"],
        example_id=sampled["example_id"],
        source_position=sampled["source_position"],
        token_id=sampled["token_id"],
    )
    write_points_csv(paths["points_path"], sampled, coordinates)
    write_embedding_pdf(coordinates, sampled["class_id"], paths["pdf_path"])
    manifest = build_manifest(
        unconstrained_run,
        constrained_run,
        unconstrained_bundle,
        constrained_bundle,
        cot_reference_bundle,
        records,
        sampled,
        perplexity,
        seed,
        cot_batch_size,
        cot_max_new_tokens,
        list(sys.argv),
    )
    manifest["mmd"] = {
        "result_file": paths["mmd_path"].name,
        **{key: value for key, value in mmd.items() if key != "per_question"},
    }
    paths["mmd_path"].write_text(
        json.dumps(mmd, indent=2), encoding="utf-8"
    )
    paths["manifest_path"].write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Jointly visualize original-model CoT references and paired latent "
            "embeddings with t-SNE and compute question-level MMD squared."
        )
    )
    parser.add_argument("--unconstrained-config", required=True)
    parser.add_argument("--constrained-config", required=True)
    parser.add_argument("--unconstrained-checkpoint")
    parser.add_argument("--constrained-checkpoint")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--points-per-group", type=int, default=1000)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cot-batch-size", type=int, default=32)
    parser.add_argument("--cot-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--mmd-bandwidth-multipliers", type=float, nargs="+",
        default=MMD_BANDWIDTH_MULTIPLIERS,
        help="RBF sigma multipliers of the pooled median distance (default: 0.25 0.5 1 2 4)",
    )
    parser.add_argument(
        "--mmd-bandwidth-max-points", type=int, default=2000,
        help="Maximum pooled points for bandwidth selection only; MMD uses full trajectories",
    )
    parser.add_argument("--mmd-bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--output-dir", default="outputs/analysis/embedding_shift_tsne")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    paths = run_embedding_tsne(
        unconstrained_config=args.unconstrained_config,
        constrained_config=args.constrained_config,
        unconstrained_checkpoint=args.unconstrained_checkpoint,
        constrained_checkpoint=args.constrained_checkpoint,
        num_samples=args.num_samples,
        points_per_group=args.points_per_group,
        perplexity=args.perplexity,
        seed=args.seed,
        cot_batch_size=args.cot_batch_size,
        cot_max_new_tokens=args.cot_max_new_tokens,
        output_dir=args.output_dir,
        mmd_bandwidth_multipliers=args.mmd_bandwidth_multipliers,
        mmd_bandwidth_max_points=args.mmd_bandwidth_max_points,
        mmd_bootstrap_resamples=args.mmd_bootstrap_resamples,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
