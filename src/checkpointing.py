import copy
import json
import random
import shutil
from pathlib import Path

import torch
import yaml


def read_trainer_state(checkpoint_dir: str | Path) -> dict:
    return json.loads(
        (Path(checkpoint_dir) / "trainer_state.json").read_text(
            encoding="utf-8"
        )
    )


def require_complete_checkpoint(checkpoint_dir: str | Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / ".complete").is_file():
        raise ValueError(f"checkpoint is not complete: {checkpoint_dir}")
    return checkpoint_dir


def save_checkpoint_at(
    checkpoint_dir,
    model,
    tokenizer,
    optimizer,
    scheduler,
    trainer_state,
    config,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    reasoner = model.module if hasattr(model, "module") else model
    reasoner.save_trainable(checkpoint_dir, tokenizer)
    saved_config = copy.deepcopy(config)
    for block_name, actual in getattr(
        reasoner, "lora_training_config", {}
    ).items():
        existing = saved_config.get(block_name)
        saved_config[block_name] = {
            **(existing if isinstance(existing, dict) else {}),
            **actual,
        }
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps(trainer_state), encoding="utf-8"
    )
    (checkpoint_dir / "config.yaml").write_text(
        yaml.safe_dump(saved_config, sort_keys=False), encoding="utf-8"
    )
    return checkpoint_dir


def save_checkpoint(
    output_dir,
    model,
    tokenizer,
    optimizer,
    scheduler,
    trainer_state,
    config,
) -> Path:
    checkpoint_dir = Path(output_dir) / f"checkpoint-{trainer_state['global_step']}"
    return save_checkpoint_at(
        checkpoint_dir,
        model,
        tokenizer,
        optimizer,
        scheduler,
        trainer_state,
        config,
    )


def mark_checkpoint_complete(checkpoint_dir: str | Path) -> None:
    (Path(checkpoint_dir) / ".complete").touch()


def save_process_rng(checkpoint_dir, process_index) -> None:
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(state, Path(checkpoint_dir) / f"rng-{process_index}.pt")


def restore_rng(state) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def restore_training_state(
    checkpoint_dir, optimizer, scheduler, process_index=0
) -> dict:
    checkpoint_dir = require_complete_checkpoint(checkpoint_dir)
    optimizer.load_state_dict(
        torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu")
    )
    scheduler.load_state_dict(
        torch.load(checkpoint_dir / "scheduler.pt", map_location="cpu")
    )
    restore_rng(
        torch.load(
            checkpoint_dir / f"rng-{process_index}.pt", map_location="cpu"
        )
    )
    return read_trainer_state(checkpoint_dir)


def finalize_checkpoint(checkpoint_dir, save_total_limit) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    mark_checkpoint_complete(checkpoint_dir)
    rotate_checkpoints(checkpoint_dir.parent, save_total_limit)


def rotate_checkpoints(output_dir, save_total_limit) -> None:
    paths = sorted(
        (
            path
            for path in Path(output_dir).glob("checkpoint-*")
            if (path / ".complete").exists()
        ),
        key=lambda path: int(path.name.rsplit("-", 1)[1]),
    )
    if not paths:
        return

    states = {}
    for path in paths:
        state_path = path / "trainer_state.json"
        states[path] = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.exists()
            else {}
        )

    grouped = {}
    for path in paths:
        stage = states[path].get("curriculum_stage")
        grouped.setdefault(stage, []).append(path)

    metric_best_paths = set()
    for stage_paths in grouped.values():
        loss_paths = [
            path
            for path in stage_paths
            if states[path].get("validation_loss") is not None
        ]
        if loss_paths:
            metric_best_paths.add(
                max(
                    loss_paths,
                    key=lambda path: (
                        -states[path]["validation_loss"],
                        int(path.name.rsplit("-", 1)[1]),
                    ),
                )
            )

        accuracy_paths = [
            path
            for path in stage_paths
            if states[path].get("validation_accuracy") is not None
        ]
        if accuracy_paths:
            metric_best_paths.add(
                max(
                    accuracy_paths,
                    key=lambda path: (
                        states[path]["validation_accuracy"],
                        int(path.name.rsplit("-", 1)[1]),
                    ),
                )
            )

    if metric_best_paths:
        recent_count = max(save_total_limit - 2, 1)
    else:
        recent_count = max(save_total_limit, 0)
    recent_paths = set(paths[-recent_count:] if recent_count else ())
    keep = metric_best_paths | recent_paths
    for path in paths:
        if path not in keep:
            shutil.rmtree(path)
