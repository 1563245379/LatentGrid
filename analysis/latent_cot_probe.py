import argparse
import copy
import gc
import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

from accelerate import Accelerator
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
import yaml

from src.compression import load_compression_reasoner
from src.data import load_prompt_template, load_train_validation, render_prompt
from src.evaluation import ANSWER_MARKER, extract_numeric_answer
from src.latent_model import (
    START_LATENT_TOKEN,
    load_latent_reasoner,
    position_ids_from_mask,
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
REPRESENTATIONS = frozenset({"constrained", "unconstrained"})
DEFAULT_BERTSCORE_MODEL = "bert-base-uncased"
PROBE_SPLIT_SEED = 42
CACHE_SPLITS = ("train", "validation", "test")
REASONER_ANSWER_CACHE_SPLITS = ("validation", "test")
REASONER_ANSWER_CACHE_VERSION = 1


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def split_probe_template(template):
    if template.count("{latents}") != 1:
        raise ValueError("probe_template must contain exactly one {latents} placeholder")
    return tuple(template.split("{latents}"))


def _complete_probe_checkpoints(output_dir):
    checkpoints = []
    for path in Path(output_dir).glob("checkpoint-*"):
        if not (path / ".complete").is_file():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        checkpoints.append((step, path.resolve()))
    return [path for _, path in sorted(checkpoints)]


def resolve_probe_checkpoint(path, mode):
    if mode not in {"resume", "eval"}:
        raise ValueError(f"unsupported probe checkpoint mode: {mode}")
    selected = Path(path).resolve()
    if (selected / ".complete").is_file():
        return selected
    if not selected.is_dir():
        raise ValueError(f"probe checkpoint path does not exist: {selected}")
    if mode == "eval":
        pointer = selected / "best_checkpoint.txt"
        if pointer.is_file():
            pointer_value = pointer.read_text(encoding="utf-8").strip()
            best = Path(pointer_value)
            if not best.is_absolute():
                best = selected / best
            best = best.resolve()
            if (best / ".complete").is_file():
                return best
            raise ValueError(f"best probe checkpoint is incomplete: {best}")
    checkpoints = _complete_probe_checkpoints(selected)
    if not checkpoints:
        raise ValueError(f"no complete probe checkpoints found in: {selected}")
    return checkpoints[-1]


def read_probe_trainer_state(checkpoint_dir):
    state_path = Path(checkpoint_dir) / "trainer_state.json"
    if not state_path.is_file():
        raise ValueError(f"probe trainer state is missing: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_probe_checkpoint(
    output_dir,
    model,
    tokenizer,
    optimizer,
    scheduler,
    trainer_state,
    decoder_config,
):
    checkpoint_dir = Path(output_dir) / f"checkpoint-{trainer_state['global_step']}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    model.llm.save_pretrained(checkpoint_dir / "adapter")
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps(trainer_state), encoding="utf-8"
    )
    (checkpoint_dir / "config.yaml").write_text(
        yaml.safe_dump(decoder_config, sort_keys=False), encoding="utf-8"
    )
    return checkpoint_dir


def save_probe_process_rng(checkpoint_dir, process_index):
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(state, Path(checkpoint_dir) / f"rng-{process_index}.pt")


def restore_probe_training_state(
    checkpoint_dir, optimizer, scheduler, process_index=0
):
    checkpoint_dir = Path(checkpoint_dir)
    optimizer.load_state_dict(
        torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu")
    )
    scheduler.load_state_dict(
        torch.load(checkpoint_dir / "scheduler.pt", map_location="cpu")
    )
    rng = torch.load(
        checkpoint_dir / f"rng-{process_index}.pt", map_location="cpu"
    )
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return read_probe_trainer_state(checkpoint_dir)


def rotate_probe_checkpoints(output_dir, save_total_limit):
    save_total_limit = int(save_total_limit)
    if save_total_limit <= 0:
        raise ValueError("training.save_total_limit must be positive")
    checkpoints = _complete_probe_checkpoints(output_dir)
    if not checkpoints:
        return
    states = {
        path: read_probe_trainer_state(path) for path in checkpoints
    }
    best = min(
        checkpoints,
        key=lambda path: (
            float(states[path]["validation_loss"]),
            -int(states[path]["global_step"]),
        ),
    )
    recent_count = max(save_total_limit - 1, 0)
    keep = {best, *checkpoints[-recent_count:]} if recent_count else {best}
    for checkpoint in checkpoints:
        if checkpoint not in keep:
            shutil.rmtree(checkpoint)
    (Path(output_dir) / "best_checkpoint.txt").write_text(
        best.name + "\n", encoding="utf-8"
    )


def finalize_probe_checkpoint(checkpoint_dir, save_total_limit):
    checkpoint_dir = Path(checkpoint_dir)
    (checkpoint_dir / ".complete").touch()
    rotate_probe_checkpoints(checkpoint_dir.parent, save_total_limit)


def resolve_run_inputs(
    paradigm_config_path,
    decoder_config_path,
    reasoner_checkpoint=None,
    output_dir=None,
    bertscore_model=None,
    mode="train",
):
    if mode not in {"train", "eval"}:
        raise ValueError(f"unsupported probe mode: {mode}")
    paradigm_path = Path(paradigm_config_path).resolve()
    decoder_path = Path(decoder_config_path).resolve()
    paradigm = load_yaml(paradigm_path)
    decoder = copy.deepcopy(load_yaml(decoder_path))
    method = paradigm["method"]
    representation = paradigm["latent_representation_type"]
    if method not in FIXED_METHODS | ADAPTIVE_METHODS:
        raise ValueError(f"unsupported latent method: {method}")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unsupported latent representation: {representation}")
    split_probe_template(decoder["probe_template"])
    configured_cache = decoder.get("latent_cache_dir")
    latent_cache_dir = (
        Path(configured_cache).resolve() if configured_cache else None
    )
    configured_answer_cache = decoder.get("reasoner_answer_cache_path")
    reasoner_answer_cache_path = (
        Path(configured_answer_cache).resolve()
        if configured_answer_cache
        else None
    )
    if latent_cache_dir is not None and not (latent_cache_dir / ".complete").is_file():
        raise ValueError(f"latent cache is incomplete: {latent_cache_dir}")
    selected_checkpoint = (
        None
        if latent_cache_dir is not None
        else reasoner_checkpoint
        or paradigm["checkpoint"].get("eval_checkpoint")
    )
    if selected_checkpoint is None and latent_cache_dir is None:
        raise ValueError(
            "--reasoner-checkpoint or paradigm checkpoint.eval_checkpoint is "
            "required when latent_cache_dir is not configured"
        )
    checkpoint_path = None
    if selected_checkpoint is not None:
        checkpoint_path = Path(selected_checkpoint).resolve()
        if not (checkpoint_path / ".complete").is_file():
            raise ValueError(f"reasoner checkpoint is incomplete: {checkpoint_path}")
        checkpoint_config = load_yaml(checkpoint_path / "config.yaml")
        for key in ("method", "latent_representation_type", "base_model"):
            if checkpoint_config[key] != paradigm[key]:
                raise ValueError(f"checkpoint config mismatch for {key}")
    probe_checkpoints = decoder.get("checkpoint", {})
    resume_value = probe_checkpoints.get("resume_from_checkpoint")
    eval_value = probe_checkpoints.get("eval_checkpoint")
    if mode == "eval" and not eval_value:
        raise ValueError(
            "checkpoint.eval_checkpoint is required in eval mode"
        )
    selected_output = Path(output_dir or decoder["output_dir"]).resolve()
    selected_bert = (
        bertscore_model
        or decoder["evaluation"].get("bertscore_model")
        or DEFAULT_BERTSCORE_MODEL
    )
    if not selected_bert:
        raise ValueError("BERTScore model must not be empty")
    return {
        "paradigm_config_path": paradigm_path,
        "decoder_config_path": decoder_path,
        "paradigm_config": paradigm,
        "decoder_config": decoder,
        "checkpoint_path": checkpoint_path,
        "latent_cache_dir": latent_cache_dir,
        "reasoner_answer_cache_path": reasoner_answer_cache_path,
        "resume_checkpoint": (
            resolve_probe_checkpoint(resume_value, mode="resume")
            if mode == "train" and resume_value
            else None
        ),
        "eval_checkpoint": (
            resolve_probe_checkpoint(eval_value, mode="eval")
            if mode == "eval"
            else None
        ),
        "mode": mode,
        "output_dir": selected_output,
        "bertscore_model": selected_bert,
    }


def load_eval_records(path, max_eval_samples=None):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            if record["cot"].strip():
                records.append(dict(record, example_id=line_number))
    if max_eval_samples is not None:
        records = records[: int(max_eval_samples)]
    return records


def select_probe_records(decoder_config):
    data = decoder_config["data"]
    split_seed = int(decoder_config.get("seed", PROBE_SPLIT_SEED))
    train, validation = load_train_validation(
        data["train_path"],
        data["validation_ratio"],
        max_train_samples=data.get("max_train_samples"),
        seed=split_seed,
    )
    test = load_eval_records(data["eval_path"], data.get("max_eval_samples"))
    if not train or not validation or not test:
        raise ValueError(
            "probe train, validation, and test splits must all be non-empty"
        )
    return train, validation, test


def load_reasoner_for_probe(config, checkpoint_path, device=None):
    loaders = {
        "answer": load_latent_reasoner,
        "curriculum": load_latent_reasoner,
        "reconstruction": load_latent_reasoner,
        "curriculum_reconstruction": load_latent_reasoner,
        "token_reconstruction": load_latent_reasoner,
        "curriculum_token_reconstruction": load_latent_reasoner,
        "compression": load_compression_reasoner,
        "curriculum_compression": load_compression_reasoner,
    }
    loader = loaders[config["method"]]
    model, tokenizer = loader(
        config,
        checkpoint_path=checkpoint_path,
        is_trainable=device is not None,
    )
    if device is not None:
        model.requires_grad_(False)
        model.to(device)
    return model, tokenizer


@torch.inference_mode()
def rollout_latent_trajectories(
    model, method, input_ids, attention_mask, latent_config
):
    adaptive = method in ADAPTIVE_METHODS and latent_config["enable_token_ce"]
    core = model.reasoner if method in WRAPPED_METHODS else model
    limit = (
        latent_config["max_latent_tokens"]
        if adaptive
        else latent_config["num_latent_tokens"]
    )
    outputs = core.llm(
        inputs_embeds=core.embed(input_ids),
        attention_mask=attention_mask,
        position_ids=position_ids_from_mask(attention_mask),
        output_hidden_states=True,
        use_cache=True,
    )
    finished = torch.zeros(
        input_ids.shape[0], dtype=torch.bool, device=input_ids.device
    )
    counts = torch.zeros(
        input_ids.shape[0], dtype=torch.long, device=input_ids.device
    )
    trajectory = []
    for _ in range(limit):
        latent = core.next_latent(
            outputs.hidden_states[-1][:, -1], outputs.logits[:, -1]
        )
        if method in WRAPPED_METHODS and core.representation_type == "unconstrained":
            latent = latent * model.embedding_scale.to(latent.dtype)
        active = ~finished if adaptive else torch.ones_like(finished)
        trajectory.append(latent * active.to(latent.dtype).unsqueeze(1))
        counts += active.long()
        attention_mask = torch.cat(
            [attention_mask, active.to(attention_mask.dtype).unsqueeze(1)], dim=1
        )
        outputs = core.llm(
            inputs_embeds=latent.unsqueeze(1),
            attention_mask=attention_mask,
            position_ids=position_ids_from_mask(attention_mask)[:, -1:],
            past_key_values=outputs.past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        if adaptive:
            end_logits = core.boundary.append_logits(
                outputs.hidden_states[-1][:, -1], outputs.logits[:, -1]
            )
            finished |= active & end_logits.argmax(-1).eq(core.end_latent_id)
            if finished.all():
                break
    latents = torch.stack(trajectory, dim=1)
    truncated = ~finished if adaptive else torch.zeros_like(finished)
    return latents, counts, truncated


class _LatentCacheState:
    def __init__(self):
        self.example_ids = []
        self.offsets = [0]
        self.latent_counts = []
        self.truncated = []
        self.records = []


def _write_bf16(handle, tensor):
    value = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    handle.write(value.view(torch.uint16).numpy().tobytes())


class LatentCacheWriter:
    def __init__(
        self, destination, hidden_size, manifest, splits=CACHE_SPLITS
    ):
        self.destination = Path(destination)
        self.destination.mkdir(parents=True, exist_ok=False)
        self.hidden_size = int(hidden_size)
        self.manifest = dict(manifest, hidden_size=self.hidden_size, dtype="bfloat16")
        self.states = {split: _LatentCacheState() for split in splits}
        self.handles = {
            split: (self.destination / f"{split}_latents.bin").open("wb")
            for split in self.states
        }

    def append(self, split, records, latents, latent_counts, truncated):
        if split not in self.states:
            raise ValueError(f"unsupported cache split: {split}")
        state = self.states[split]
        if not isinstance(latents, torch.Tensor) or latents.ndim != 3:
            raise ValueError("latents must have shape [batch, width, hidden_size]")
        if latents.shape[2] != self.hidden_size:
            raise ValueError("latent hidden size does not match cache hidden size")
        batch_size, width, _ = latents.shape
        if len(records) != batch_size:
            raise ValueError("records and latents batch sizes differ")
        if latent_counts.ndim != 1 or latent_counts.shape[0] != batch_size:
            raise ValueError("latent_counts must have one value per record")
        if truncated.ndim != 1 or truncated.shape[0] != batch_size:
            raise ValueError("truncated must have one value per record")

        known_ids = set(state.example_ids)
        example_ids = [record["example_id"] for record in records]
        if len(set(example_ids)) != len(example_ids) or any(
            example_id in known_ids for example_id in example_ids
        ):
            raise ValueError(f"duplicate example_id in {split}: {example_ids}")
        counts = [int(latent_counts[row].item()) for row in range(batch_size)]
        if any(count < 0 or count > width for count in counts):
            raise ValueError("latent count exceeds batch trajectory width")
        for row, record in enumerate(records):
            example_id = record["example_id"]
            count = counts[row]
            _write_bf16(self.handles[split], latents[row, :count])
            state.example_ids.append(example_id)
            state.offsets.append(state.offsets[-1] + count)
            state.latent_counts.append(count)
            state.truncated.append(bool(truncated[row].item()))
            state.records.append(
                {
                    "split": split,
                    "example_id": example_id,
                    "cot": record["cot"],
                }
            )

    def finalize(self):
        for handle in self.handles.values():
            handle.close()
        metadata = {
            split: {
                "example_ids": torch.tensor(state.example_ids, dtype=torch.int64),
                "offsets": torch.tensor(state.offsets, dtype=torch.int64),
                "latent_counts": torch.tensor(
                    state.latent_counts, dtype=torch.int32
                ),
                "truncated": torch.tensor(state.truncated, dtype=torch.bool),
            }
            for split, state in self.states.items()
        }
        torch.save(metadata, self.destination / "metadata.pt")
        with (self.destination / "records.jsonl").open("w", encoding="utf-8") as handle:
            for split in self.states:
                for record in self.states[split].records:
                    handle.write(json.dumps(record) + "\n")
        self.manifest["split_counts"] = {
            split: len(state.example_ids) for split, state in self.states.items()
        }
        (self.destination / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8"
        )
        (self.destination / ".complete").touch()
        return self.destination


class LatentTrajectoryCache:
    def __init__(self, cache_dir, manifest, metadata, records, mapped):
        self.cache_dir = Path(cache_dir)
        self.manifest = manifest
        self.metadata = metadata
        self.records = records
        self.mapped = mapped
        self.hidden_size = int(manifest["hidden_size"])

    @classmethod
    def open(cls, cache_dir, expected_manifest=None, required_splits=None):
        cache_dir = Path(cache_dir)
        required_splits = tuple(required_splits or CACHE_SPLITS)
        unsupported_splits = set(required_splits) - set(CACHE_SPLITS)
        if unsupported_splits:
            raise ValueError(
                f"unsupported required cache splits: {sorted(unsupported_splits)}"
            )
        if not (cache_dir / ".complete").is_file():
            raise ValueError(f"latent cache is incomplete: {cache_dir}")
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"latent cache manifest is missing: {cache_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if expected_manifest is not None:
            mismatches = {
                key: (expected_manifest[key], manifest.get(key))
                for key in expected_manifest
                if manifest.get(key) != expected_manifest[key]
            }
            if mismatches:
                raise ValueError(f"latent cache manifest mismatch: {mismatches}")
        hidden_size = manifest.get("hidden_size")
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("latent cache hidden_size must be a positive integer")
        if manifest.get("dtype") != "bfloat16":
            raise ValueError("latent cache dtype must be bfloat16")
        metadata_path = cache_dir / "metadata.pt"
        records_path = cache_dir / "records.jsonl"
        if not metadata_path.is_file() or not records_path.is_file():
            raise ValueError(f"latent cache files are missing: {cache_dir}")
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
        records = {split: [] for split in CACHE_SPLITS}
        with records_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                split = record.get("split")
                if split not in records:
                    raise ValueError(f"unsupported cache split in records: {split}")
                records[split].append(record)

        mapped = {}
        for split in CACHE_SPLITS:
            if split not in metadata:
                raise ValueError(f"latent cache metadata missing split: {split}")
            split_metadata = metadata[split]
            example_ids = split_metadata["example_ids"]
            offsets = split_metadata["offsets"]
            latent_counts = split_metadata["latent_counts"]
            truncated = split_metadata["truncated"]
            row_count = int(example_ids.shape[0])
            if (
                offsets.ndim != 1
                or offsets.shape[0] != row_count + 1
                or latent_counts.shape[0] != row_count
                or truncated.shape[0] != row_count
                or offsets[0].item() != 0
                or torch.any(offsets[1:] < offsets[:-1])
                or len(records[split]) != row_count
            ):
                raise ValueError(f"latent cache metadata is malformed for {split}")
            record_example_ids = [record.get("example_id") for record in records[split]]
            metadata_example_ids = example_ids.tolist()
            if record_example_ids != metadata_example_ids:
                raise ValueError(
                    f"latent cache example_id order mismatch for {split}: "
                    f"metadata={metadata_example_ids}, records={record_example_ids}"
                )
            manifest_split_counts = manifest.get("split_counts")
            if (
                manifest_split_counts is not None
                and manifest_split_counts.get(split) != row_count
            ):
                raise ValueError(f"latent cache split count mismatch for {split}")
            target_count = int(offsets[-1].item())
            if torch.any((offsets[1:] - offsets[:-1]) != latent_counts.to(offsets.dtype)):
                raise ValueError(f"latent cache offsets disagree for {split}")
            if split in required_splits:
                mapped[split] = torch.from_file(
                    str(cache_dir / f"{split}_latents.bin"),
                    shared=False,
                    size=target_count * hidden_size,
                    dtype=torch.bfloat16,
                ).view(target_count, hidden_size)
        return cls(cache_dir, manifest, metadata, records, mapped)

    def row(self, split, index):
        split_metadata = self.metadata[split]
        start = int(split_metadata["offsets"][index].item())
        end = int(split_metadata["offsets"][index + 1].item())
        result = dict(self.records[split][index])
        result["latents"] = self.mapped[split][start:end]
        result["latent_count"] = int(split_metadata["latent_counts"][index].item())
        result["truncated"] = bool(split_metadata["truncated"][index].item())
        return result


class LatentProbeDataset(torch.utils.data.Dataset):
    def __init__(self, cache: LatentTrajectoryCache, split: str):
        self.cache = cache
        self.split = split

    def __len__(self):
        return len(self.cache.records[self.split])

    def __getitem__(self, index):
        return self.cache.row(self.split, index)


class FullTrajectoryCollator:
    def __init__(self, tokenizer, probe_template: str):
        self.tokenizer = tokenizer
        prefix, suffix = split_probe_template(probe_template)
        self.prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)

    def __call__(self, rows):
        if not rows:
            raise ValueError("cannot collate an empty row list")

        max_latents = max(row["latent_count"] for row in rows)
        context_width = max_latents + len(self.prefix_ids) + len(self.suffix_ids)
        context_ids = torch.full(
            (len(rows), context_width),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        context_mask = torch.zeros_like(context_ids)
        latent_positions = torch.zeros_like(context_ids, dtype=torch.bool)
        latent_values = []
        for row_index, row in enumerate(rows):
            count = row["latent_count"]
            start = max_latents - count
            latent_start = start + len(self.prefix_ids)
            suffix_start = latent_start + count
            end = suffix_start + len(self.suffix_ids)
            context_ids[row_index, start:latent_start] = torch.tensor(
                self.prefix_ids, dtype=torch.long
            )
            context_ids[row_index, suffix_start:end] = torch.tensor(
                self.suffix_ids, dtype=torch.long
            )
            context_mask[row_index, start:end] = 1
            latent_positions[row_index, latent_start:suffix_start] = True
            latent_values.append(row["latents"])

        encoded_targets = [
            self.tokenizer.encode(row["cot"], add_special_tokens=False)
            + [self.tokenizer.eos_token_id]
            for row in rows
        ]
        target_width = max(len(target) for target in encoded_targets)
        target_ids = torch.full(
            (len(rows), target_width),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
        )
        target_mask = torch.zeros(
            (len(rows), target_width), dtype=torch.bool
        )
        for row_index, target in enumerate(encoded_targets):
            target_width_for_row = len(target)
            target_ids[row_index, :target_width_for_row] = torch.tensor(
                target, dtype=torch.long
            )
            target_mask[row_index, :target_width_for_row] = True

        latent_counts = torch.tensor(
            [row["latent_count"] for row in rows], dtype=torch.long
        )
        return {
            "context_input_ids": context_ids,
            "context_attention_mask": context_mask,
            "latent_positions": latent_positions,
            "latent_values": torch.cat(latent_values),
            "target_ids": target_ids,
            "target_mask": target_mask,
            "example_ids": torch.tensor(
                [row["example_id"] for row in rows], dtype=torch.long
            ),
            "source_cots": [row["cot"] for row in rows],
            "latent_counts": latent_counts,
            "truncated": torch.tensor(
                [row["truncated"] for row in rows], dtype=torch.bool
            ),
            "eos_token_id": self.tokenizer.eos_token_id,
        }


class FullTrajectoryProbe(nn.Module):
    def __init__(self, llm: nn.Module):
        super().__init__()
        self.llm = llm

    def _context_embeddings(self, batch):
        input_embeddings = self.llm.get_input_embeddings()
        context_input_ids = batch["context_input_ids"]
        try:
            embedding_device = input_embeddings.weight.device
        except AttributeError:
            embedding_device = context_input_ids.device
        context_input_ids = context_input_ids.to(embedding_device)
        context_embeddings = input_embeddings(context_input_ids)
        latent_values = batch["latent_values"]
        if latent_values.ndim != 2:
            raise ValueError("latent values must have shape [count, hidden width]")
        if latent_values.shape[-1] != context_embeddings.shape[-1]:
            raise ValueError(
                "latent hidden width does not match decoder embedding hidden width"
            )
        latent_positions = batch["latent_positions"].to(
            device=context_embeddings.device
        )
        if int(latent_positions.sum().item()) != latent_values.shape[0]:
            raise ValueError(
                "latent positions and latent values have different counts"
            )
        context_embeddings = context_embeddings.clone()
        context_embeddings[latent_positions] = latent_values.to(
            device=context_embeddings.device, dtype=context_embeddings.dtype
        )
        return context_embeddings

    def forward(self, batch) -> dict[str, Tensor]:
        context_embeddings = self._context_embeddings(batch)
        embedding = self.llm.get_input_embeddings()
        target_ids = batch["target_ids"].to(context_embeddings.device)
        target_mask = batch["target_mask"].to(context_embeddings.device).bool()
        target_embeddings = embedding(target_ids)
        inputs_embeds = torch.cat([context_embeddings, target_embeddings], dim=1)
        context_mask = batch["context_attention_mask"].to(
            device=context_embeddings.device, dtype=target_mask.dtype
        )
        attention_mask = torch.cat([context_mask, target_mask], dim=1)
        labels = torch.cat(
            [
                torch.full(
                    context_mask.shape,
                    -100,
                    dtype=target_ids.dtype,
                    device=target_ids.device,
                ),
                target_ids.masked_fill(~target_mask, -100),
            ],
            dim=1,
        )
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids_from_mask(attention_mask),
            use_cache=False,
        )
        shifted_logits = outputs.logits[:, :-1]
        shifted_labels = labels[:, 1:]
        token_losses = F.cross_entropy(
            shifted_logits.flatten(0, 1),
            shifted_labels.flatten(),
            ignore_index=-100,
            reduction="none",
        ).view_as(shifted_labels)
        valid = shifted_labels.ne(-100)
        probe_loss_sums = (token_losses * valid).sum(dim=1)
        probe_token_counts = valid.sum(dim=1)
        return {
            "loss": probe_loss_sums.sum()
            / probe_token_counts.sum().clamp_min(1),
            "probe_loss_sums": probe_loss_sums,
            "probe_token_counts": probe_token_counts,
        }

    def _eos_token_ids(self, batch):
        config = getattr(self.llm, "config", None)
        configured = getattr(config, "eos_token_id", None)
        if configured is None:
            configured = batch.get("eos_token_id")
        if configured is None:
            configured = 1
        if isinstance(configured, Tensor):
            configured = configured.detach().cpu().reshape(-1).tolist()
        elif isinstance(configured, int):
            configured = [configured]
        else:
            configured = list(configured)
        return tuple(int(token_id) for token_id in configured)

    def generate(self, batch, max_new_tokens) -> tuple[Tensor, Tensor]:
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        context_embeddings = self._context_embeddings(batch)
        batch_size = context_embeddings.shape[0]
        context_mask = batch["context_attention_mask"].to(
            device=context_embeddings.device
        )
        eos_token_ids = self._eos_token_ids(batch)
        eos_tensor = torch.tensor(
            eos_token_ids, dtype=torch.long, device=context_embeddings.device
        )
        finished = torch.zeros(
            batch_size, dtype=torch.bool, device=context_embeddings.device
        )
        generated = []
        if max_new_tokens == 0:
            return (
                torch.empty(
                    (batch_size, 0), dtype=torch.long, device=context_embeddings.device
                ),
                ~finished,
            )

        outputs = self.llm(
            inputs_embeds=context_embeddings,
            attention_mask=context_mask,
            position_ids=position_ids_from_mask(context_mask),
            use_cache=True,
        )
        embedding = self.llm.get_input_embeddings()
        try:
            vocab_size = int(embedding.num_embeddings)
        except AttributeError:
            vocab_size = int(outputs.logits.shape[-1])
        for step in range(max_new_tokens):
            next_ids = outputs.logits[:, -1, :vocab_size].argmax(dim=-1)
            if finished.any():
                next_ids = torch.where(
                    finished, eos_tensor[0].expand_as(next_ids), next_ids
                )
            generated.append(next_ids)
            finished |= torch.isin(next_ids, eos_tensor)
            if finished.all() or step + 1 == max_new_tokens:
                break
            context_mask = torch.cat(
                [
                    context_mask,
                    torch.ones(
                        (batch_size, 1),
                        dtype=context_mask.dtype,
                        device=context_mask.device,
                    ),
                ],
                dim=1,
            )
            next_embeddings = embedding(next_ids)
            outputs = self.llm(
                inputs_embeds=next_embeddings.unsqueeze(1),
                attention_mask=context_mask,
                position_ids=position_ids_from_mask(context_mask)[:, -1:],
                past_key_values=outputs.past_key_values,
                use_cache=True,
            )
        return torch.stack(generated, dim=1), ~finished


def load_probe_decoder(
    paradigm_config: dict[str, Any],
    decoder_config: dict[str, Any],
    checkpoint_path=None,
    is_trainable=True,
) -> tuple[FullTrajectoryProbe, object]:
    base_model_path = paradigm_config["base_model"]
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    tokenizer_path = (
        checkpoint_path / "tokenizer"
        if checkpoint_path is not None
        and (checkpoint_path / "tokenizer").is_dir()
        else base_model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (
        torch.bfloat16
        if decoder_config["training"]["bf16"]
        else torch.float32
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, dtype=dtype
    )
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    if checkpoint_path is not None:
        adapter_path = checkpoint_path / "adapter"
        if not adapter_path.is_dir():
            raise ValueError(f"probe checkpoint adapter is missing: {adapter_path}")
        llm = PeftModel.from_pretrained(
            base_model, adapter_path, is_trainable=is_trainable
        )
    else:
        decoder_lora = decoder_config["decoder_lora"]
        llm = get_peft_model(
            base_model,
            LoraConfig(
                r=decoder_lora["r"],
                lora_alpha=decoder_lora["lora_alpha"],
                lora_dropout=decoder_lora["lora_dropout"],
                target_modules=decoder_lora["target_modules"],
                task_type="CAUSAL_LM",
                bias="none",
            ),
        )
    return FullTrajectoryProbe(llm), tokenizer


def summarize_probe_losses(loss_sums, token_counts):
    loss_sums = loss_sums.double()
    token_counts = token_counts.long()
    per_token = float(loss_sums.sum().item() / token_counts.sum().item())
    per_sequence = float(loss_sums.mean().item())
    return {
        "loss_nats_per_token": per_token,
        "nll_nats_per_sequence": per_sequence,
        "perplexity": math.exp(per_token),
        "examples": int(loss_sums.numel()),
        "tokens": int(token_counts.sum().item()),
    }


def _resolve_probe_warmup_steps(value, total_updates):
    return int(value) if value >= 1 else math.ceil(total_updates * value)


def _probe_dataloader(cache, split, tokenizer, decoder_config, batch_size, shuffle):
    return DataLoader(
        LatentProbeDataset(cache, split),
        batch_size=int(batch_size),
        shuffle=shuffle,
        collate_fn=FullTrajectoryCollator(
            tokenizer, decoder_config["probe_template"]
        ),
    )


def _probe_tracking_kwargs(training_config, output_dir):
    report_to = training_config.get("report_to")
    if not report_to:
        return {}
    return {
        "log_with": report_to,
        "project_dir": str(Path(output_dir) / "runs"),
    }


def _probe_progress(iterable, description, accelerator=None):
    show_progress = accelerator is None or getattr(
        accelerator, "is_local_main_process", True
    )
    return tqdm(
        iterable,
        desc=description,
        disable=not show_progress,
        dynamic_ncols=True,
    )


@torch.inference_mode()
def _validation_probe_loss(
    accelerator, model, validation_loader, epoch, total_epochs
):
    model.eval()
    loss_sums = []
    token_counts = []
    progress = _probe_progress(
        validation_loader,
        f"Validating probe {epoch}/{total_epochs}",
        accelerator,
    )
    for batch in progress:
        outputs = model(batch)
        gathered_sums, gathered_counts = accelerator.gather_for_metrics(
            (
                outputs["probe_loss_sums"].detach(),
                outputs["probe_token_counts"].detach(),
            )
        )
        loss_sums.append(gathered_sums.cpu())
        token_counts.append(gathered_counts.cpu())
    if not loss_sums:
        raise ValueError("probe validation loader must not be empty")
    return summarize_probe_losses(
        torch.cat(loss_sums), torch.cat(token_counts)
    )["loss_nats_per_token"]


def train_probe_decoder(
    model,
    tokenizer,
    cache,
    decoder_config,
    output_dir,
    resume_checkpoint=None,
):
    training_config = decoder_config["training"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gradient_accumulation_steps = int(
        training_config["gradient_accumulation_steps"]
    )
    num_train_epochs = int(training_config["num_train_epochs"])
    train_loader = _probe_dataloader(
        cache,
        "train",
        tokenizer,
        decoder_config,
        training_config["per_device_train_batch_size"],
        shuffle=True,
    )
    validation_loader = _probe_dataloader(
        cache,
        "validation",
        tokenizer,
        decoder_config,
        training_config["per_device_eval_batch_size"],
        shuffle=False,
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * num_train_epochs
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    scheduler = get_scheduler(
        training_config["lr_scheduler_type"],
        optimizer,
        num_warmup_steps=_resolve_probe_warmup_steps(
            training_config["warmup_steps"], total_updates
        ),
        num_training_steps=total_updates,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="bf16" if training_config["bf16"] else "no",
        step_scheduler_with_optimizer=False,
        **_probe_tracking_kwargs(training_config, output_dir),
    )
    tracking_enabled = bool(training_config.get("report_to"))
    if tracking_enabled:
        accelerator.init_trackers("latent-cot-probe")
    model, optimizer, scheduler, train_loader, validation_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, validation_loader
    )

    completed_epoch = 0
    global_step = 0
    validation_history = []
    final_train_loss = None
    if resume_checkpoint is not None:
        restored = restore_probe_training_state(
            resume_checkpoint,
            optimizer,
            scheduler,
            process_index=accelerator.process_index,
        )
        completed_epoch = int(restored["completed_epoch"])
        global_step = int(restored["global_step"])
        validation_history = list(restored.get("validation_history", []))
        final_train_loss = restored.get("train_loss")

    optimizer.zero_grad()
    for epoch in range(completed_epoch + 1, num_train_epochs + 1):
        model.train()
        epoch_loss_sums = []
        epoch_token_counts = []
        progress = _probe_progress(
            train_loader,
            f"Training probe {epoch}/{num_train_epochs}",
            accelerator,
        )
        for batch in progress:
            with accelerator.accumulate(model):
                outputs = model(batch)
                accelerator.backward(outputs["loss"])
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad()
            gathered_loss_sums, gathered_token_counts = (
                accelerator.gather_for_metrics(
                    (
                        outputs["probe_loss_sums"].detach(),
                        outputs["probe_token_counts"].detach(),
                    )
                )
            )
            epoch_loss_sums.append(gathered_loss_sums.cpu())
            epoch_token_counts.append(gathered_token_counts.cpu())
            if accelerator.sync_gradients:
                global_step += 1
                if (
                    tracking_enabled
                    and global_step % int(training_config["logging_steps"]) == 0
                ):
                    accelerator.log(
                        {
                            "train/loss": float(outputs["loss"].detach().item()),
                            "train/learning_rate": float(
                                scheduler.get_last_lr()[0]
                            ),
                        },
                        step=global_step,
                    )

        epoch_metrics = summarize_probe_losses(
            torch.cat(epoch_loss_sums), torch.cat(epoch_token_counts)
        )
        final_train_loss = epoch_metrics["loss_nats_per_token"]
        validation_loss = _validation_probe_loss(
            accelerator,
            model,
            validation_loader,
            epoch,
            num_train_epochs,
        )
        validation_history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "validation_loss": validation_loss,
            }
        )
        if tracking_enabled:
            accelerator.log(
                {
                    "train/epoch_loss": final_train_loss,
                    "validation/loss": validation_loss,
                },
                step=global_step,
            )
        trainer_state = {
            "completed_epoch": epoch,
            "global_step": global_step,
            "train_loss": final_train_loss,
            "validation_loss": validation_loss,
            "validation_history": validation_history,
        }
        checkpoint_dir = output_dir / f"checkpoint-{global_step}"
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_probe_checkpoint(
                output_dir,
                accelerator.unwrap_model(model),
                tokenizer,
                optimizer,
                scheduler,
                trainer_state,
                decoder_config,
            )
        accelerator.wait_for_everyone()
        save_probe_process_rng(checkpoint_dir, accelerator.process_index)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            finalize_probe_checkpoint(
                checkpoint_dir, training_config["save_total_limit"]
            )
        accelerator.wait_for_everyone()

    if tracking_enabled:
        accelerator.end_training()
    if final_train_loss is None:
        raise ValueError(
            "resume checkpoint has no train_loss and no epochs remain to train"
        )
    best_checkpoint = resolve_probe_checkpoint(output_dir, mode="eval")
    return best_checkpoint, float(final_train_loss)


def _probe_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_probe_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@torch.inference_mode()
def evaluate_probe_decoder(
    model, tokenizer, cache, decoder_config, split="validation"
):
    if split not in {"validation", "test"}:
        raise ValueError(f"unsupported evaluation split: {split}")
    evaluation_config = decoder_config["evaluation"]
    evaluation_loader = _probe_dataloader(
        cache,
        split,
        tokenizer,
        decoder_config,
        decoder_config["training"]["per_device_eval_batch_size"],
        shuffle=False,
    )
    accelerator = Accelerator()
    evaluation_loader = accelerator.prepare_data_loader(evaluation_loader)
    model = accelerator.prepare_model(model, evaluation_mode=True)
    model = accelerator.unwrap_model(model)
    device = _probe_model_device(model)
    model.eval()
    rows = []
    progress = _probe_progress(
        evaluation_loader,
        f"Evaluating probe {split}",
        accelerator,
    )
    for batch in progress:
        model_batch = _move_probe_batch(batch, device)
        outputs = model(model_batch)
        batch_loss_sums = outputs["probe_loss_sums"].detach().cpu()
        batch_token_counts = outputs["probe_token_counts"].detach().cpu()

        generation_batch = {
            key: value
            for key, value in model_batch.items()
            if key not in {"target_ids", "target_mask"}
        }
        generated_ids, generation_truncated = model.generate(
            generation_batch, evaluation_config["max_new_tokens"]
        )
        generated_text = tokenizer.batch_decode(
            generated_ids.detach().cpu().tolist(), skip_special_tokens=True
        )
        generation_truncated = generation_truncated.detach().cpu().bool()
        example_ids = model_batch["example_ids"].detach().cpu()
        latent_counts = model_batch["latent_counts"].detach().cpu()
        reasoner_truncated = model_batch["truncated"].detach().cpu().bool()
        local_rows = []
        for row_index, reconstructed_cot in enumerate(generated_text):
            local_rows.append(
                {
                    "example_id": int(example_ids[row_index].item()),
                    "reference_cot": model_batch["source_cots"][row_index],
                    "reconstructed_cot": reconstructed_cot,
                    "latent_count": int(latent_counts[row_index].item()),
                    "reasoner_truncated": bool(
                        reasoner_truncated[row_index].item()
                    ),
                    "generation_truncated": bool(
                        generation_truncated[row_index].item()
                    ),
                    "probe_nll_nats": float(batch_loss_sums[row_index].item()),
                    "probe_token_count": int(batch_token_counts[row_index].item()),
                }
            )
        gathered_rows = accelerator.gather_for_metrics(
            local_rows, use_gather_object=True
        )
        rows.extend(gathered_rows)

    expected_ids = [
        int(record["example_id"]) for record in cache.records[split]
    ]
    row_ids = [int(row["example_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)) or sorted(row_ids) != sorted(expected_ids):
        raise ValueError(
            f"distributed {split} gathering produced missing or duplicate examples"
        )
    expected_order = {example_id: index for index, example_id in enumerate(expected_ids)}
    rows.sort(key=lambda row: expected_order[int(row["example_id"])])
    summary = summarize_probe_losses(
        torch.tensor(
            [row["probe_nll_nats"] for row in rows], dtype=torch.float64
        ),
        torch.tensor(
            [row["probe_token_count"] for row in rows], dtype=torch.long
        ),
    )
    evaluation_examples = len(expected_ids)
    reasoner_truncated_count = sum(
        row["reasoner_truncated"] for row in rows
    )
    generation_truncated_count = sum(
        row["generation_truncated"] for row in rows
    )
    accelerator.wait_for_everyone()
    return {
        f"{split}_loss_nats_per_token": summary["loss_nats_per_token"],
        f"{split}_nll_nats_per_sequence": summary[
            "nll_nats_per_sequence"
        ],
        f"{split}_perplexity": summary["perplexity"],
        f"{split}_examples": summary["examples"],
        f"{split}_tokens": summary["tokens"],
        f"{split}_reasoner_truncated_count": int(reasoner_truncated_count),
        f"{split}_reasoner_truncated_rate": reasoner_truncated_count
        / max(evaluation_examples, 1),
        f"{split}_generation_truncated_count": int(generation_truncated_count),
        f"{split}_generation_truncated_rate": generation_truncated_count
        / max(evaluation_examples, 1),
    }, rows


def load_bertscorer():
    from bert_score import BERTScorer

    return BERTScorer


def score_reconstructions(rows, model_type, batch_size, device=None):
    scored_rows = [dict(row) for row in rows]
    scorer = load_bertscorer()(
        model_type=model_type,
        batch_size=batch_size,
        device=device,
    )
    candidates = [row["reconstructed_cot"] for row in scored_rows]
    references = [row["reference_cot"] for row in scored_rows]
    precision, recall, f1 = scorer.score(candidates, references, verbose=True)
    precision = precision.detach().cpu().tolist()
    recall = recall.detach().cpu().tolist()
    f1 = f1.detach().cpu().tolist()
    for index, row in enumerate(scored_rows):
        row["bertscore_precision"] = float(precision[index])
        row["bertscore_recall"] = float(recall[index])
        row["bertscore_f1"] = float(f1[index])
    return {
        "bertscore_precision": float(sum(precision) / len(precision)),
        "bertscore_recall": float(sum(recall) / len(recall)),
        "bertscore_f1": float(sum(f1) / len(f1)),
        "bertscore_hash": scorer.hash,
    }, scored_rows


@torch.inference_mode()
def generate_reasoner_answers(
    reasoner,
    tokenizer,
    records,
    config,
    prompt_template,
    batch_size,
    max_new_tokens=64,
):
    device = next(reasoner.parameters()).device
    answer_prefix_ids = torch.tensor(
        tokenizer.encode(ANSWER_MARKER, add_special_tokens=False),
        dtype=torch.long,
        device=device,
    )
    answers = {}
    offsets = range(0, len(records), batch_size)
    for offset in _probe_progress(offsets, "Generating original answers"):
        batch_records = records[offset : offset + batch_size]
        prompts = [
            render_prompt(prompt_template, record["question"])
            + START_LATENT_TOKEN
            for record in batch_records
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        adaptive = (
            config["method"] in ADAPTIVE_METHODS
            and config["latent"]["enable_token_ce"]
        )
        if adaptive:
            generated_ids, _, _ = reasoner.generate_adaptive_answer_ids(
                input_ids,
                attention_mask,
                answer_prefix_ids,
                max_latent_tokens=config["latent"]["max_latent_tokens"],
                max_new_tokens=max_new_tokens,
            )
        else:
            generated_ids, _ = reasoner.generate_answer_ids(
                input_ids,
                attention_mask,
                config["latent"]["num_latent_tokens"],
                answer_prefix_ids,
                max_new_tokens=max_new_tokens,
            )
        decoded = tokenizer.batch_decode(
            generated_ids.detach().cpu(), skip_special_tokens=True
        )
        answers.update(
            (record["example_id"], extract_numeric_answer(text))
            for record, text in zip(batch_records, decoded)
        )
    return answers


def score_answer_consistency(rows, original_answers):
    scored_rows = []
    consistent_count = 0
    original_missing_count = 0
    reconstructed_missing_count = 0
    for row in rows:
        scored = dict(row)
        original_answer = original_answers[row["example_id"]]
        reconstructed_answer = extract_numeric_answer(row["reconstructed_cot"])
        consistent = (
            original_answer is not None
            and reconstructed_answer is not None
            and original_answer == reconstructed_answer
        )
        scored.update(
            {
                "original_answer": original_answer,
                "reconstructed_answer": reconstructed_answer,
                "answer_consistent": consistent,
            }
        )
        scored_rows.append(scored)
        consistent_count += consistent
        original_missing_count += original_answer is None
        reconstructed_missing_count += reconstructed_answer is None
    total = len(scored_rows)
    return {
        "answer_consistent_count": int(consistent_count),
        "answer_consistency_rate": consistent_count / max(total, 1),
        "original_answer_missing_count": int(original_missing_count),
        "reconstructed_answer_missing_count": int(reconstructed_missing_count),
    }, scored_rows


def write_reasoner_answer_cache(path, manifest, answers):
    path = Path(path)
    if set(answers) != set(REASONER_ANSWER_CACHE_SPLITS):
        raise ValueError(
            "reasoner answer cache requires validation and test splits"
        )
    serialized_answers = {}
    for split in REASONER_ANSWER_CACHE_SPLITS:
        rows = []
        for example_id, answer in sorted(answers[split].items()):
            if answer is not None and not isinstance(answer, str):
                raise ValueError("cached reasoner answers must be strings or null")
            rows.append({"example_id": int(example_id), "answer": answer})
        serialized_answers[split] = rows
    payload = {
        "version": REASONER_ANSWER_CACHE_VERSION,
        "manifest": manifest,
        "answers": serialized_answers,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


def load_reasoner_answer_cache(path, expected_manifest, expected_ids):
    path = Path(path)
    encoded = path.read_bytes()
    payload = json.loads(encoded)
    if payload.get("version") != REASONER_ANSWER_CACHE_VERSION:
        raise ValueError("unsupported reasoner answer cache version")
    if payload.get("manifest") != expected_manifest:
        raise ValueError("reasoner answer cache manifest mismatch")
    cached_splits = payload.get("answers", {})
    if set(cached_splits) != set(REASONER_ANSWER_CACHE_SPLITS):
        raise ValueError("reasoner answer cache split mismatch")
    answers = {}
    for split in REASONER_ANSWER_CACHE_SPLITS:
        rows = cached_splits[split]
        split_answers = {}
        for row in rows:
            example_id = int(row["example_id"])
            answer = row["answer"]
            if example_id in split_answers:
                raise ValueError(
                    f"duplicate example ID in reasoner answer cache: {example_id}"
                )
            if answer is not None and not isinstance(answer, str):
                raise ValueError("cached reasoner answers must be strings or null")
            split_answers[example_id] = answer
        if set(split_answers) != set(expected_ids[split]):
            raise ValueError(
                f"reasoner answer cache example IDs mismatch for {split}"
            )
        answers[split] = split_answers
    return answers, hashlib.sha256(encoded).hexdigest()


def load_or_create_reasoner_answers(
    cache_path, manifest, records_by_split, generate_split_answers
):
    expected_ids = {
        split: [int(record["example_id"]) for record in records_by_split[split]]
        for split in REASONER_ANSWER_CACHE_SPLITS
    }
    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.is_file():
        return load_reasoner_answer_cache(path, manifest, expected_ids)
    answers = {
        split: generate_split_answers(split, records_by_split[split])
        for split in REASONER_ANSWER_CACHE_SPLITS
    }
    for split in REASONER_ANSWER_CACHE_SPLITS:
        if set(answers[split]) != set(expected_ids[split]):
            raise ValueError(
                f"generated reasoner answer IDs mismatch for {split}"
            )
    digest = (
        write_reasoner_answer_cache(path, manifest, answers)
        if path is not None
        else None
    )
    return answers, digest


def _fingerprint_records(records):
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_reasoner_answer_cache_manifest(
    reasoner_checkpoint,
    paradigm_config,
    validation_records,
    test_records,
    answer_batch_size,
    answer_max_new_tokens,
):
    return {
        "reasoner_checkpoint_path": str(Path(reasoner_checkpoint).resolve()),
        "method": paradigm_config["method"],
        "representation": paradigm_config["latent_representation_type"],
        "answer_batch_size": int(answer_batch_size),
        "answer_max_new_tokens": int(answer_max_new_tokens),
        "split_fingerprints": {
            "validation": _fingerprint_records(validation_records),
            "test": _fingerprint_records(test_records),
        },
    }


def _cache_manifest_for_run(
    run, train_records=None, validation_records=None, test_records=None
):
    config = run["paradigm_config"]
    decoder_config = run["decoder_config"]
    data = decoder_config["data"]
    manifest = {
        "method": config["method"],
        "representation": config["latent_representation_type"],
        "checkpoint_path": str(Path(run["checkpoint_path"]).resolve()),
        "train_path": str(Path(data["train_path"]).resolve()),
        "eval_path": str(Path(data["eval_path"]).resolve()),
        "split_seed": int(decoder_config.get("seed", PROBE_SPLIT_SEED)),
        "validation_ratio": data["validation_ratio"],
        "max_train_samples": data.get("max_train_samples"),
        "max_eval_samples": data.get("max_eval_samples"),
    }
    split_records = {
        "train": (
            []
            if run.get("mode", "train") == "eval" and train_records is not None
            else train_records
        ),
        "validation": validation_records,
        "test": test_records,
    }
    if any(records is not None for records in split_records.values()):
        if any(records is None for records in split_records.values()):
            raise ValueError("all cache split records are required together")
        manifest["split_counts"] = {
            split: len(records) for split, records in split_records.items()
        }
        manifest["split_fingerprints"] = {
            split: _fingerprint_records(records)
            for split, records in split_records.items()
        }
    return manifest


def _expected_direct_cache_manifest(run):
    config = run["paradigm_config"]
    decoder_config = run["decoder_config"]
    data = decoder_config["data"]
    return {
        "method": config["method"],
        "representation": config["latent_representation_type"],
        "train_path": str(Path(data["train_path"]).resolve()),
        "eval_path": str(Path(data["eval_path"]).resolve()),
        "split_seed": int(decoder_config.get("seed", PROBE_SPLIT_SEED)),
        "validation_ratio": data["validation_ratio"],
        "max_train_samples": data.get("max_train_samples"),
        "max_eval_samples": data.get("max_eval_samples"),
    }


def _collect_latent_split(
    writer,
    split,
    records,
    reasoner,
    tokenizer,
    config,
    prompt_template,
    batch_size,
    accelerator=None,
):
    device = next(reasoner.parameters()).device
    offsets = range(0, len(records), batch_size)
    progress = _probe_progress(
        offsets, f"Collecting {split} latents", accelerator
    )
    for offset in progress:
        batch_records = records[offset : offset + batch_size]
        prompts = [
            render_prompt(prompt_template, record["question"])
            + START_LATENT_TOKEN
            for record in batch_records
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        latents, latent_counts, truncated = rollout_latent_trajectories(
            reasoner,
            config["method"],
            input_ids,
            attention_mask,
            config["latent"],
        )
        writer.append(split, batch_records, latents, latent_counts, truncated)


def _process_record_slice(records, process_index, num_processes):
    start = len(records) * process_index // num_processes
    end = len(records) * (process_index + 1) // num_processes
    return records[start:end]


def _merge_latent_cache_shards(shard_dirs, destination, manifest):
    shards = []
    for shard_dir in shard_dirs:
        metadata = torch.load(
            shard_dir / "metadata.pt", map_location="cpu", weights_only=True
        )
        records = {split: [] for split in CACHE_SPLITS}
        with (shard_dir / "records.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    records[record["split"]].append(record)
        shards.append((shard_dir, metadata, records))

    shard_manifest = json.loads(
        (shard_dirs[0] / "manifest.json").read_text(encoding="utf-8")
    )
    destination.mkdir(parents=True, exist_ok=False)
    merged_metadata = {}
    merged_records = {split: [] for split in CACHE_SPLITS}
    for split in CACHE_SPLITS:
        example_ids = torch.cat(
            [metadata[split]["example_ids"] for _, metadata, _ in shards]
        )
        latent_counts = torch.cat(
            [metadata[split]["latent_counts"] for _, metadata, _ in shards]
        )
        truncated = torch.cat(
            [metadata[split]["truncated"] for _, metadata, _ in shards]
        )
        merged_metadata[split] = {
            "example_ids": example_ids,
            "offsets": torch.cat(
                [
                    torch.zeros(1, dtype=torch.int64),
                    latent_counts.to(torch.int64).cumsum(dim=0),
                ]
            ),
            "latent_counts": latent_counts,
            "truncated": truncated,
        }
        with (destination / f"{split}_latents.bin").open("wb") as output:
            for shard_dir, _, records in shards:
                with (shard_dir / f"{split}_latents.bin").open("rb") as source:
                    shutil.copyfileobj(source, output)
                merged_records[split].extend(records[split])

    torch.save(merged_metadata, destination / "metadata.pt")
    with (destination / "records.jsonl").open("w", encoding="utf-8") as handle:
        for split in CACHE_SPLITS:
            for record in merged_records[split]:
                handle.write(json.dumps(record) + "\n")
    merged_manifest = dict(
        manifest,
        hidden_size=shard_manifest["hidden_size"],
        dtype="bfloat16",
        split_counts={
            split: len(merged_records[split]) for split in CACHE_SPLITS
        },
    )
    (destination / "manifest.json").write_text(
        json.dumps(merged_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (destination / ".complete").touch()
    return destination


def collect_or_load_latent_cache(
    run, train_records, validation_records, test_records, accelerator=None
):
    expected_manifest = _cache_manifest_for_run(
        run, train_records, validation_records, test_records
    )
    cache_dir = Path(run["output_dir"]) / "latent_cache"
    process_index = int(getattr(accelerator, "process_index", 0))
    num_processes = int(getattr(accelerator, "num_processes", 1))
    is_main_process = bool(getattr(accelerator, "is_main_process", True))
    wait_for_everyone = (
        accelerator.wait_for_everyone if accelerator is not None else lambda: None
    )
    shards_dir = cache_dir.parent / f".{cache_dir.name}.shards"

    if is_main_process:
        if (cache_dir / ".complete").is_file():
            try:
                LatentTrajectoryCache.open(cache_dir, expected_manifest)
            except ValueError as error:
                if "manifest mismatch" not in str(error):
                    raise
                shutil.rmtree(cache_dir)
        elif cache_dir.exists():
            shutil.rmtree(cache_dir)
        if num_processes > 1:
            if shards_dir.exists():
                shutil.rmtree(shards_dir)
            shards_dir.mkdir(parents=True)
    wait_for_everyone()
    if (cache_dir / ".complete").is_file():
        return LatentTrajectoryCache.open(cache_dir, expected_manifest)

    config = run["paradigm_config"]
    if num_processes > 1:
        reasoner, tokenizer = load_reasoner_for_probe(
            config,
            Path(run["checkpoint_path"]),
            device=accelerator.device,
        )
    else:
        reasoner, tokenizer = load_reasoner_for_probe(
            config, Path(run["checkpoint_path"])
        )
    reasoner.eval()
    core = reasoner.reasoner if config["method"] in WRAPPED_METHODS else reasoner
    hidden_size = int(core.llm.get_input_embeddings().weight.shape[1])
    writer_dir = (
        shards_dir / f"process-{process_index}"
        if num_processes > 1
        else cache_dir
    )
    writer = LatentCacheWriter(writer_dir, hidden_size, expected_manifest)
    prompt_template = load_prompt_template(config["template_path"])
    batch_size = int(run["decoder_config"]["collection"]["batch_size"])
    split_records = {
        "train": train_records if run.get("mode", "train") == "train" else [],
        "validation": validation_records,
        "test": test_records,
    }
    try:
        for split, records in split_records.items():
            process_records = _process_record_slice(
                records, process_index, num_processes
            )
            if not process_records:
                continue
            _collect_latent_split(
                writer,
                split,
                process_records,
                reasoner,
                tokenizer,
                config,
                prompt_template,
                batch_size,
                accelerator,
            )
        writer.finalize()
    finally:
        for handle in writer.handles.values():
            if not handle.closed:
                handle.close()
    del reasoner, tokenizer, writer, core
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    wait_for_everyone()
    if num_processes > 1 and is_main_process:
        shard_dirs = [
            shards_dir / f"process-{index}" for index in range(num_processes)
        ]
        _merge_latent_cache_shards(shard_dirs, cache_dir, expected_manifest)
        shutil.rmtree(shards_dir)
    wait_for_everyone()
    return LatentTrajectoryCache.open(cache_dir, expected_manifest)


def run_latent_cot_probe(
    paradigm_config_path,
    decoder_config_path,
    reasoner_checkpoint=None,
    output_dir=None,
    bertscore_model=None,
    mode="train",
) -> dict[str, Path]:
    run = resolve_run_inputs(
        paradigm_config_path,
        decoder_config_path,
        reasoner_checkpoint=reasoner_checkpoint,
        output_dir=output_dir,
        bertscore_model=bertscore_model,
        mode=mode,
    )
    decoder_config = run["decoder_config"]
    orchestration_accelerator = Accelerator(
        mixed_precision="bf16" if decoder_config["training"]["bf16"] else "no"
    )
    seed = int(decoder_config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    destination = Path(run["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    if run["latent_cache_dir"] is not None:
        cache = LatentTrajectoryCache.open(
            run["latent_cache_dir"],
            _expected_direct_cache_manifest(run),
            required_splits=("validation", "test") if mode == "eval" else None,
        )
    else:
        train_records, validation_records, test_records = select_probe_records(
            decoder_config
        )
        cache = collect_or_load_latent_cache(
            run,
            train_records,
            validation_records,
            test_records,
            orchestration_accelerator,
        )

    selected_probe_checkpoint = (
        run["eval_checkpoint"] if mode == "eval" else run["resume_checkpoint"]
    )
    model, tokenizer = load_probe_decoder(
        run["paradigm_config"],
        decoder_config,
        checkpoint_path=selected_probe_checkpoint,
        is_trainable=mode == "train",
    )
    if mode == "eval":
        best_checkpoint = run["eval_checkpoint"]
        final_train_loss = read_probe_trainer_state(best_checkpoint).get(
            "train_loss"
        )
    else:
        best_checkpoint, final_train_loss = train_probe_decoder(
            model,
            tokenizer,
            cache,
            decoder_config,
            destination,
            resume_checkpoint=run["resume_checkpoint"],
        )
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, tokenizer = load_probe_decoder(
            run["paradigm_config"],
            decoder_config,
            checkpoint_path=best_checkpoint,
            is_trainable=False,
        )

    validation_metrics, validation_rows = evaluate_probe_decoder(
        model, tokenizer, cache, decoder_config, split="validation"
    )
    test_metrics, test_rows = evaluate_probe_decoder(
        model, tokenizer, cache, decoder_config, split="test"
    )

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    evaluation_config = decoder_config["evaluation"]
    bertscore_batch_size = int(evaluation_config["bertscore_batch_size"])
    answer_batch_size = int(
        evaluation_config.get(
            "answer_batch_size",
            decoder_config["training"]["per_device_eval_batch_size"],
        )
    )
    answer_max_new_tokens = int(
        evaluation_config.get("answer_max_new_tokens", 64)
    )
    reasoner_answer_cache_path = run.get(
        "reasoner_answer_cache_path"
    ) or decoder_config.get("reasoner_answer_cache_path")
    reasoner_answer_cache_sha256 = None
    orchestration_accelerator.wait_for_everyone()
    if orchestration_accelerator.is_main_process:
        validation_bertscore, scored_validation_rows = score_reconstructions(
            validation_rows,
            model_type=run["bertscore_model"],
            batch_size=bertscore_batch_size,
        )
        test_bertscore, scored_test_rows = score_reconstructions(
            test_rows,
            model_type=run["bertscore_model"],
            batch_size=bertscore_batch_size,
        )
        if validation_rows or test_rows:
            _, validation_records, test_records = select_probe_records(
                decoder_config
            )
            reasoner_checkpoint = (
                run["checkpoint_path"] or cache.manifest["checkpoint_path"]
            )
            answer_cache_manifest = build_reasoner_answer_cache_manifest(
                reasoner_checkpoint,
                run["paradigm_config"],
                validation_records,
                test_records,
                answer_batch_size,
                answer_max_new_tokens,
            )
            reasoner_bundle = None

            def generate_split_answers(_split, records):
                nonlocal reasoner_bundle
                if reasoner_bundle is None:
                    reasoner_bundle = load_reasoner_for_probe(
                        run["paradigm_config"], Path(reasoner_checkpoint)
                    )
                    reasoner_bundle[0].eval()
                prompt_template = load_prompt_template(
                    run["paradigm_config"]["template_path"]
                )
                return generate_reasoner_answers(
                    reasoner_bundle[0],
                    reasoner_bundle[1],
                    records,
                    run["paradigm_config"],
                    prompt_template,
                    answer_batch_size,
                    answer_max_new_tokens,
                )

            answers, reasoner_answer_cache_sha256 = (
                load_or_create_reasoner_answers(
                    reasoner_answer_cache_path,
                    answer_cache_manifest,
                    {
                        "validation": validation_records,
                        "test": test_records,
                    },
                    generate_split_answers,
                )
            )
            validation_answers = answers["validation"]
            test_answers = answers["test"]
            if reasoner_bundle is not None:
                del reasoner_bundle
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            validation_answers = {}
            test_answers = {}
        validation_consistency, scored_validation_rows = score_answer_consistency(
            scored_validation_rows, validation_answers
        )
        test_consistency, scored_test_rows = score_answer_consistency(
            scored_test_rows, test_answers
        )
    else:
        validation_bertscore = {}
        test_bertscore = {}
        validation_consistency = {}
        test_consistency = {}
        scored_validation_rows = validation_rows
        scored_test_rows = test_rows
    metrics = {
        "final_train_loss_nats_per_token": (
            float(final_train_loss) if final_train_loss is not None else None
        ),
        **validation_metrics,
        **test_metrics,
        **{
            f"validation_{name}": value
            for name, value in validation_bertscore.items()
        },
        **{f"test_{name}": value for name, value in test_bertscore.items()},
        **{
            f"validation_{name}": value
            for name, value in validation_consistency.items()
        },
        **{f"test_{name}": value for name, value in test_consistency.items()},
        "train_examples": len(cache.records["train"]),
        "validation_examples": len(cache.records["validation"]),
        "test_examples": len(cache.records["test"]),
        "bertscore_model": run["bertscore_model"],
        "bertscore_batch_size": bertscore_batch_size,
        "answer_batch_size": answer_batch_size,
        "answer_max_new_tokens": answer_max_new_tokens,
    }
    metrics_path = destination / "metrics.json"
    validation_predictions_path = destination / "validation_predictions.jsonl"
    test_predictions_path = destination / "test_predictions.jsonl"
    manifest_path = destination / "run_manifest.json"
    latent_cache_path = Path(cache.cache_dir)
    best_checkpoint = Path(best_checkpoint)
    artifact_paths = {
        "latent_cache": latent_cache_path,
        "best_checkpoint": best_checkpoint,
        "decoder_adapter": best_checkpoint / "adapter",
        "metrics": metrics_path,
        "validation_predictions": validation_predictions_path,
        "test_predictions": test_predictions_path,
        "manifest": manifest_path,
    }
    if reasoner_answer_cache_path is not None:
        artifact_paths["reasoner_answer_cache"] = Path(
            reasoner_answer_cache_path
        )
    manifest = {
        "paradigm_config_path": str(Path(run["paradigm_config_path"]).resolve()),
        "decoder_config_path": str(Path(run["decoder_config_path"]).resolve()),
        "reasoner_checkpoint_path": (
            str(Path(run["checkpoint_path"]).resolve())
            if run["checkpoint_path"] is not None
            else cache.manifest.get("checkpoint_path")
        ),
        "probe_checkpoint_path": str(best_checkpoint.resolve()),
        "method": run["paradigm_config"]["method"],
        "representation": run["paradigm_config"]["latent_representation_type"],
        "base_model": run["paradigm_config"]["base_model"],
        "seed": seed,
        "bertscore_model": run["bertscore_model"],
        "bertscore_batch_size": bertscore_batch_size,
        "answer_batch_size": answer_batch_size,
        "answer_max_new_tokens": answer_max_new_tokens,
        "reasoner_answer_cache_path": (
            str(Path(reasoner_answer_cache_path).resolve())
            if reasoner_answer_cache_path is not None
            else None
        ),
        "reasoner_answer_cache_sha256": reasoner_answer_cache_sha256,
        "probe_template": decoder_config["probe_template"],
        "artifacts": {
            key: str(path.resolve()) for key, path in artifact_paths.items()
        },
    }
    if orchestration_accelerator.is_main_process:
        metrics_path.write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        with validation_predictions_path.open("w", encoding="utf-8") as handle:
            for row in scored_validation_rows:
                handle.write(json.dumps(row) + "\n")
        with test_predictions_path.open("w", encoding="utf-8") as handle:
            for row in scored_test_rows:
                handle.write(json.dumps(row) + "\n")
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    orchestration_accelerator.wait_for_everyone()
    return artifact_paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Probe explicit chain-of-thought information in latent trajectories."
    )
    parser.add_argument("--paradigm-config", required=True)
    parser.add_argument("--decoder-config", required=True)
    parser.add_argument("--reasoner-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--bertscore-model")
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    paths = run_latent_cot_probe(
        args.paradigm_config,
        args.decoder_config,
        reasoner_checkpoint=args.reasoner_checkpoint,
        output_dir=args.output_dir,
        bertscore_model=args.bertscore_model,
        mode=args.mode,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
