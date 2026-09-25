import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm

from src.checkpointing import (
    finalize_checkpoint,
    mark_checkpoint_complete,
    require_complete_checkpoint,
    restore_training_state,
    save_checkpoint_at,
    save_process_rng,
)
from src.cot import CotCollator, load_cot_reasoner
from src.compression import (
    CompressionCollator,
    CurriculumCompressionCollator,
    load_compression_reasoner,
)
from src.data import (
    load_prompt_response_templates,
    load_prompt_template,
    load_train_validation,
)
from src.evaluation import ANSWER_MARKER, extract_numeric_answer
from src.indirect import IndirectCollator
from src.latent_model import load_latent_reasoner
from src.no_cot import NoCotCollator, load_no_cot_reasoner
from src.reconstruction import (
    CurriculumReconstructionCollator,
    ReconstructionCollator,
    load_reconstruction_reasoner,
)
from src.token_reconstruction import (
    CurriculumTokenReconstructionCollator,
    TokenReconstructionCollator,
    load_token_reconstruction_reasoner,
)


DIRECT_CURRICULUM_METHODS = {
    "curriculum_reconstruction",
    "curriculum_compression",
    "curriculum_token_reconstruction",
}

VARIABLE_LATENT_METHODS = {
    "compression",
    "curriculum_compression",
    "token_reconstruction",
    "curriculum_token_reconstruction",
}

LOSS_STATS = {
    "answer": ("answer_loss_sums", "answer_token_counts"),
    "cot": ("cot_loss_sums", "cot_token_counts"),
    "response": ("response_loss_sums", "response_token_counts"),
    "reconstruction": (
        "reconstruction_loss_sums",
        "reconstruction_token_counts",
    ),
    "latent": ("latent_loss_sums", "latent_counts"),
    "token_ce": ("token_ce_loss_sums", "token_ce_counts"),
}


@dataclass(frozen=True)
class OnlineTraceRuntime:
    boundary_by_epoch: dict[int, int]
    optimizer_steps: int
    seed: int
    stage_checkpoint_dir: Path
    on_backward: Callable[..., None]


def _epoch_generator(seed: int, epoch: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return generator


def resolve_warmup_steps(value: float, total_updates: int) -> int:
    return int(value) if value >= 1 else math.ceil(total_updates * value)


def _curriculum_final_stage(config: dict) -> int:
    if config["method"] in VARIABLE_LATENT_METHODS:
        return 6
    latent_tokens_per_stage = config["curriculum"][
        "latent_tokens_per_stage"
    ]
    return config["latent"]["num_latent_tokens"] // latent_tokens_per_stage


def curriculum_reset_epochs(config: dict) -> frozenset[int]:
    stage_epochs = config["curriculum"]["stage_epochs"]
    return frozenset(
        stage * stage_epochs + 1
        for stage in range(1, _curriculum_final_stage(config))
    )


def is_curriculum_reset_epoch(config: dict, epoch: int) -> bool:
    return epoch in curriculum_reset_epochs(config)


def reset_optimizer_state(optimizer) -> None:
    optimizer.state.clear()


def _curriculum_policy(config):
    method = config["method"]
    if method in {"curriculum", "curriculum_reconstruction"}:
        return curriculum_reset_epochs(config), True
    if method in {
        "curriculum_compression",
        "curriculum_token_reconstruction",
    }:
        return curriculum_reset_epochs(config), True
    return frozenset(), False


def _loss_components(
    method: str, enable_token_ce: bool = True
) -> tuple[str, ...]:
    components = {
        "no_cot": ("answer",),
        "cot": ("response",),
        "answer": ("answer",),
        "curriculum": ("answer", "cot"),
        "reconstruction": ("answer", "reconstruction"),
        "compression": ("latent", "answer"),
        "curriculum_reconstruction": ("cot", "reconstruction", "answer"),
        "curriculum_compression": ("cot", "latent", "answer"),
        "token_reconstruction": ("answer", "reconstruction"),
        "curriculum_token_reconstruction": (
            "cot",
            "reconstruction",
            "answer",
        ),
    }[method]
    if enable_token_ce and method == "token_reconstruction":
        return components + ("token_ce",)
    if enable_token_ce and method in {
        "compression",
        "curriculum_compression",
        "curriculum_token_reconstruction",
    }:
        return components[:-1] + ("token_ce", components[-1])
    return components


def run_validation(
    accelerator,
    model,
    dataloader,
    method: str,
    epoch=1,
    num_train_epochs=1,
    enable_token_ce=True,
    tokenizer=None,
    max_latent_tokens=None,
    num_latent_tokens=None,
    max_new_tokens=64,
) -> dict[str, float]:
    model.eval()
    unwrapped_model = (
        accelerator.unwrap_model(model)
        if hasattr(accelerator, "unwrap_model")
        else model
    )
    generation_model = (
        getattr(unwrapped_model, "reasoner", unwrapped_model)
        if method
        in {
            "reconstruction",
            "curriculum_reconstruction",
            "token_reconstruction",
            "curriculum_token_reconstruction",
        }
        else unwrapped_model
    )
    use_adaptive_generation = (
        method in VARIABLE_LATENT_METHODS
        and enable_token_ce
        and hasattr(generation_model, "generate_adaptive_answer_ids")
    )
    evaluate_accuracy = (
        tokenizer is not None
        and (
            use_adaptive_generation
            or hasattr(generation_model, "generate_answer_ids")
        )
    )
    components = _loss_components(method, enable_token_ce)
    names = tuple(
        name
        for component in components
        for name in LOSS_STATS[component]
    )
    totals = {name: 0.0 for name in names}
    totals.update({"latent_length_sums": 0.0, "example_counts": 0.0})
    with torch.no_grad():
        for batch in _validation_progress(
            dataloader, accelerator, epoch, num_train_epochs
        ):
            outputs = model(batch)
            if method in VARIABLE_LATENT_METHODS:
                latent_lengths = batch["latent_counts"].float()
            else:
                shape_anchor = outputs[LOSS_STATS[components[0]][0]]
                latent_lengths = torch.full_like(
                    shape_anchor,
                    batch["latent_count"],
                    dtype=torch.float32,
                )
            example_counts = torch.ones_like(latent_lengths)
            values = tuple(outputs[name].detach() for name in names) + (
                latent_lengths,
                example_counts,
            )
            gathered_names = names + (
                "latent_length_sums",
                "example_counts",
            )
            accuracy_fields = {
                "prompt_input_ids",
                "prompt_attention_mask",
                "suffix_input_ids",
                "answer_mask",
            }
            if evaluate_accuracy and accuracy_fields <= batch.keys():
                answer_prefix_ids = torch.tensor(
                    tokenizer.encode(ANSWER_MARKER, add_special_tokens=False),
                    dtype=torch.long,
                    device=batch["prompt_input_ids"].device,
                )
                if use_adaptive_generation:
                    generated_ids, _, _ = (
                        generation_model.generate_adaptive_answer_ids(
                            batch["prompt_input_ids"],
                            batch["prompt_attention_mask"],
                            answer_prefix_ids,
                            max_latent_tokens=max_latent_tokens,
                        )
                    )
                else:
                    generated_ids, _ = generation_model.generate_answer_ids(
                        batch["prompt_input_ids"],
                        batch["prompt_attention_mask"],
                        batch.get("latent_count", num_latent_tokens),
                        answer_prefix_ids,
                        max_new_tokens=max_new_tokens,
                    )
                predictions = tokenizer.batch_decode(
                    generated_ids.detach().cpu(), skip_special_tokens=True
                )
                gold_ids = [
                    ids[mask].detach().cpu()
                    for ids, mask in zip(
                        batch["suffix_input_ids"], batch["answer_mask"]
                    )
                ]
                gold_answers = tokenizer.batch_decode(
                    gold_ids, skip_special_tokens=True
                )
                normalized_predictions = [
                    extract_numeric_answer(prediction)
                    for prediction in predictions
                ]
                normalized_gold = [
                    extract_numeric_answer(gold) for gold in gold_answers
                ]
                correct = torch.tensor(
                    [
                        predicted is not None and predicted == gold
                        for predicted, gold in zip(
                            normalized_predictions, normalized_gold
                        )
                    ],
                    dtype=torch.float32,
                    device=batch["prompt_input_ids"].device,
                )
                values += (correct,)
                gathered_names += ("correct_counts",)
                totals.setdefault("correct_counts", 0.0)
            gathered = accelerator.gather_for_metrics(
                values
            )
            for name, gathered_values in zip(gathered_names, gathered):
                totals[name] += gathered_values.sum().item()

    metrics = {
        component + "_loss": totals[LOSS_STATS[component][0]]
        / max(totals[LOSS_STATS[component][1]], 1)
        for component in components
    }
    metrics["loss"] = sum(metrics.values())
    metrics["latent_length_mean"] = totals["latent_length_sums"] / max(
        totals["example_counts"], 1
    )
    if "correct_counts" in totals:
        metrics["accuracy"] = totals["correct_counts"] / max(
            totals["example_counts"], 1
        )
    return metrics


def _collator(
    config, tokenizer, prompt_template, epoch, response_template=None
):
    if config["method"] == "cot":
        if response_template is None:
            raise ValueError("cot training requires a response template")
        return CotCollator(
            tokenizer,
            prompt_template,
            response_template,
            config["data"],
            config["training"]["model_max_length"],
        )
    if config["method"] == "no_cot":
        if response_template is None:
            raise ValueError("no_cot training requires a response template")
        return NoCotCollator(
            tokenizer, prompt_template, response_template
        )
    if config["method"] == "curriculum_reconstruction":
        return CurriculumReconstructionCollator(
            tokenizer,
            prompt_template,
            config["latent"],
            config["curriculum"],
            epoch,
        )
    if config["method"] == "curriculum_token_reconstruction":
        return CurriculumTokenReconstructionCollator(
            tokenizer, prompt_template, epoch, config["curriculum"],
            config["latent"].get("tokens_per_latent", 2),
        )
    if config["method"] == "curriculum_compression":
        return CurriculumCompressionCollator(
            tokenizer, prompt_template, epoch, config["curriculum"],
            config["latent"].get("tokens_per_latent", 2),
        )
    if config["method"] == "compression":
        return CompressionCollator(tokenizer, prompt_template, config["latent"].get("tokens_per_latent", 2))
    if config["method"] == "reconstruction":
        return ReconstructionCollator(
            tokenizer, prompt_template, config["latent"]
        )
    if config["method"] == "token_reconstruction":
        return TokenReconstructionCollator(tokenizer, prompt_template, config["latent"].get("tokens_per_latent", 2))
    return IndirectCollator(
        tokenizer,
        prompt_template,
        config["method"],
        epoch,
        config["latent"],
        config.get("curriculum"),
    )


def _dataloader(records, batch_size, collator, shuffle, generator=None):
    return DataLoader(
        records,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        generator=generator,
    )


def _training_progress(dataloader, accelerator, epoch, total_epochs):
    return tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{total_epochs}",
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
    )


def _validation_progress(dataloader, accelerator, epoch, total_epochs):
    return tqdm(
        dataloader,
        desc=f"Validation {epoch}/{total_epochs}",
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
    )


def _curriculum_stage(config, epoch):
    stage_epochs = config["curriculum"]["stage_epochs"]
    return min(
        (epoch - 1) // stage_epochs + 1,
        _curriculum_final_stage(config),
    )


def _save_distributed_checkpoint(
    accelerator,
    checkpoint_dir,
    model,
    tokenizer,
    optimizer,
    scheduler,
    trainer_state,
    config,
    save_total_limit=None,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint_at(
            checkpoint_dir,
            accelerator.unwrap_model(model),
            tokenizer,
            optimizer,
            scheduler,
            trainer_state,
            config,
        )
    accelerator.wait_for_everyone()
    save_process_rng(checkpoint_dir, accelerator.process_index)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if save_total_limit is not None:
            finalize_checkpoint(checkpoint_dir, save_total_limit)
        else:
            mark_checkpoint_complete(checkpoint_dir)
    accelerator.wait_for_everyone()
    return checkpoint_dir


def _train_phase(
    config: dict,
    accelerator: Accelerator,
    train_records: list[dict],
    validation_records: list[dict],
    output_dir: Path,
    num_train_epochs: int,
    model: torch.nn.Module,
    tokenizer,
    collator,
    method: str,
    resume_checkpoint: str | Path | None = None,
    reset_epochs=frozenset(),
    save_curriculum_stage=False,
    runtime: OnlineTraceRuntime | None = None,
):
    training_config = config["training"]
    enable_token_ce = (
        method in VARIABLE_LATENT_METHODS
        and config["latent"]["enable_token_ce"]
    )
    for role, summary in getattr(model, "lora_training_summary", {}).items():
        accelerator.print(
            f"{role} LoRA: target_modules={summary['target_modules']} "
            f"trainable_modules={summary['trainable_modules']} "
            f"trainable_parameters={summary['trainable_parameters']} "
            f"total_parameters={summary['total_parameters']}"
        )
    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    sampler_seed = runtime.seed if runtime is not None else None
    sizing_loader = _dataloader(
        train_records,
        training_config["per_device_train_batch_size"],
        collator(1, "train"),
        shuffle=True,
    )
    sizing_loader = accelerator.prepare(sizing_loader)
    updates_per_epoch = math.ceil(
        len(sizing_loader) / training_config["gradient_accumulation_steps"]
    )
    if runtime is not None and runtime.optimizer_steps > updates_per_epoch:
        raise ValueError(
            "online trace optimizer_steps exceeds updates_per_epoch: "
            f"{runtime.optimizer_steps} > {updates_per_epoch}"
        )
    total_updates = updates_per_epoch * num_train_epochs
    scheduler = get_scheduler(
        training_config["lr_scheduler_type"],
        optimizer,
        num_warmup_steps=resolve_warmup_steps(
            training_config["warmup_steps"], total_updates
        ),
        num_training_steps=total_updates,
    )
    model, optimizer, scheduler = accelerator.prepare(
        model, optimizer, scheduler
    )
    if runtime is not None and not resume_checkpoint:
        set_seed(runtime.seed, device_specific=True)

    completed_epoch = 0
    global_step = 0
    restored = None
    if resume_checkpoint:
        restored = restore_training_state(
            resume_checkpoint,
            optimizer,
            scheduler,
            process_index=accelerator.process_index,
        )
        completed_epoch = restored["completed_epoch"]
        global_step = restored["global_step"]
        if "sampler_seed" in restored:
            sampler_seed = int(restored["sampler_seed"])
        if save_curriculum_stage:
            expected_stage = _curriculum_stage(config, completed_epoch)
            if restored.get("curriculum_stage") != expected_stage:
                raise ValueError(
                    "curriculum stage mismatch: "
                    f"checkpoint has {restored.get('curriculum_stage')}, "
                    f"expected {expected_stage}"
                )

    latest_checkpoint = (
        Path(resume_checkpoint)
        if resume_checkpoint
        else output_dir / f"checkpoint-{global_step}"
    )
    train_loader = sizing_loader
    validation_loader = None
    optimizer.zero_grad()
    max_boundary_epoch = (
        max(runtime.boundary_by_epoch) if runtime is not None else None
    )
    if runtime is not None and resume_checkpoint is None:
        initial_state = {
            "completed_epoch": 0,
            "global_step": 0,
            "sampler_seed": sampler_seed,
        }
        if save_curriculum_stage:
            initial_state["curriculum_stage"] = 0
        latest_checkpoint = _save_distributed_checkpoint(
            accelerator,
            runtime.stage_checkpoint_dir / "epoch-0",
            model,
            tokenizer,
            optimizer,
            scheduler,
            initial_state,
            config,
        )
    for epoch in range(completed_epoch + 1, num_train_epochs + 1):
        if epoch in reset_epochs:
            reset_optimizer_state(optimizer)

        train_loader = _dataloader(
            train_records,
            training_config["per_device_train_batch_size"],
            collator(epoch, "train"),
            shuffle=True,
            generator=(
                _epoch_generator(sampler_seed, epoch)
                if sampler_seed is not None
                else None
            ),
        )
        train_loader = accelerator.prepare(train_loader)
        model.train()
        completed_updates_in_epoch = 0
        micro_batch_in_update = 0
        progress_bar = _training_progress(
            train_loader, accelerator, epoch, num_train_epochs
        )
        for batch in progress_bar:
            micro_batch_in_update += 1
            tracing = (
                runtime is not None
                and epoch in runtime.boundary_by_epoch
                and completed_updates_in_epoch < runtime.optimizer_steps
            )
            boundary_epoch = (
                runtime.boundary_by_epoch[epoch] if tracing else None
            )
            with accelerator.accumulate(model):
                outputs = (
                    model(batch, retain_latent_grad=True)
                    if tracing
                    else model(batch)
                )
                accelerator.backward(outputs["loss"])
                if tracing:
                    runtime.on_backward(
                        accelerator=accelerator,
                        method=method,
                        boundary_epoch=boundary_epoch,
                        training_epoch=epoch,
                        optimizer_update=completed_updates_in_epoch + 1,
                        micro_batch=micro_batch_in_update,
                        update_finished=accelerator.sync_gradients,
                        batch=batch,
                        outputs=outputs,
                    )
                max_grad_norm = training_config.get("max_grad_norm")
                if accelerator.sync_gradients and max_grad_norm is not None:
                    accelerator.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                completed_updates_in_epoch += 1
                global_step += 1
                micro_batch_in_update = 0
                if (
                    runtime is not None
                    and epoch == max_boundary_epoch
                    and completed_updates_in_epoch == runtime.optimizer_steps
                ):
                    return (
                        latest_checkpoint,
                        model,
                        optimizer,
                        scheduler,
                        train_loader,
                        validation_loader,
                    )
                if global_step % training_config["logging_steps"] == 0:
                    metrics = {
                        "train/loss": outputs["loss"]
                        .detach()
                        .item(),
                        "train/learning_rate": scheduler.get_last_lr()[
                            0
                        ],
                    }
                    for component in _loss_components(method, enable_token_ce):
                        metrics[
                            f"train/{component}_loss"
                        ] = outputs[f"{component}_loss"].detach().item()
                    accelerator.log(metrics, step=global_step)
                    progress_bar.set_postfix(
                        step=global_step,
                        loss=f'{metrics["train/loss"]:.4f}',
                        lr=(
                            f'{metrics["train/learning_rate"]:.2e}'
                        ),
                    )

        validation_loader = _dataloader(
            validation_records,
            training_config["per_device_eval_batch_size"],
            collator(epoch, "validation"),
            shuffle=False,
        )
        validation_loader = accelerator.prepare(validation_loader)
        latent_config = config.get("latent", {})
        validation_metrics = run_validation(
            accelerator,
            model,
            validation_loader,
            method=method,
            epoch=epoch,
            num_train_epochs=num_train_epochs,
            enable_token_ce=enable_token_ce,
            tokenizer=tokenizer,
            max_latent_tokens=latent_config.get("max_latent_tokens"),
            num_latent_tokens=latent_config.get("num_latent_tokens"),
            max_new_tokens=config.get("evaluation", {}).get(
                "max_new_tokens", 64
            ),
        )
        accelerator.log(
            {
                f"validation/{name}": value
                for name, value in validation_metrics.items()
            },
            step=global_step,
        )

        trainer_state = {
            "completed_epoch": epoch,
            "global_step": global_step,
            "validation_loss": validation_metrics["loss"],
        }
        if "accuracy" in validation_metrics:
            trainer_state["validation_accuracy"] = validation_metrics[
                "accuracy"
            ]
        if sampler_seed is not None:
            trainer_state["sampler_seed"] = sampler_seed
        if save_curriculum_stage:
            trainer_state["curriculum_stage"] = _curriculum_stage(
                config, epoch
            )

        if runtime is not None:
            if epoch in runtime.boundary_by_epoch.values():
                latest_checkpoint = _save_distributed_checkpoint(
                    accelerator,
                    runtime.stage_checkpoint_dir / f"epoch-{epoch}",
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    trainer_state,
                    config,
                )
        else:
            latest_checkpoint = _save_distributed_checkpoint(
                accelerator,
                output_dir / f"checkpoint-{global_step}",
                model,
                tokenizer,
                optimizer,
                scheduler,
                trainer_state,
                config,
                save_total_limit=training_config["save_total_limit"],
            )

    return (
        latest_checkpoint,
        model,
        optimizer,
        scheduler,
        train_loader,
        validation_loader,
    )


def train_indirect_from_records(
    config: dict,
    accelerator: Accelerator,
    train_records: list[dict],
    validation_records: list[dict],
    prompt_template: str,
    runtime: OnlineTraceRuntime | None = None,
):
    if config["method"] not in {"answer", "curriculum"}:
        raise ValueError(
            "train_indirect_from_records only supports answer and curriculum"
        )
    checkpoint_config = config["checkpoint"]
    resume_checkpoint = checkpoint_config["resume_from_checkpoint"]
    if runtime is not None and resume_checkpoint:
        raise ValueError(
            "online trace training cannot resume from a checkpoint"
        )
    if runtime is not None:
        set_seed(runtime.seed)
    model, tokenizer = load_latent_reasoner(
        config,
        checkpoint_path=resume_checkpoint or None,
        is_trainable=True,
    )
    reset_epochs, save_curriculum_stage = _curriculum_policy(config)
    return _train_phase(
        config,
        accelerator,
        train_records,
        validation_records,
        output_dir=Path(config["output_dir"]),
        num_train_epochs=config["training"]["num_train_epochs"],
        model=model,
        tokenizer=tokenizer,
        collator=lambda epoch, split: _collator(
            config,
            tokenizer,
            prompt_template,
            epoch,
        ),
        method=config["method"],
        resume_checkpoint=resume_checkpoint or None,
        reset_epochs=reset_epochs,
        save_curriculum_stage=save_curriculum_stage,
        runtime=runtime,
    )


def train_from_config(config: dict) -> Path:
    output_dir = Path(config["output_dir"])
    training_config = config["training"]
    data_config = config["data"]
    checkpoint_config = config["checkpoint"]
    resume_checkpoint = checkpoint_config["resume_from_checkpoint"]
    if resume_checkpoint:
        require_complete_checkpoint(resume_checkpoint)

    if "seed" in training_config:
        set_seed(training_config["seed"])

    accelerator = Accelerator(
        gradient_accumulation_steps=training_config[
            "gradient_accumulation_steps"
        ],
        mixed_precision="bf16" if training_config["bf16"] else "no",
        log_with=training_config["report_to"],
        project_dir=output_dir / "runs",
        step_scheduler_with_optimizer=False,
    )
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.init_trackers("latent-cot-bench")

    train_records, validation_records = load_train_validation(
        data_config["train_path"],
        data_config["validation_ratio"],
        max_train_samples=data_config["max_train_samples"],
        seed=42,
    )
    response_template = None
    if config["method"] in {"cot", "no_cot"}:
        prompt_template, response_template = load_prompt_response_templates(
            config["template_path"]
        )
    else:
        prompt_template = load_prompt_template(config["template_path"])
    try:
        if config["method"] in {"answer", "curriculum"}:
            return train_indirect_from_records(
                config,
                accelerator,
                train_records,
                validation_records,
                prompt_template,
            )[0]

        if config["method"] == "cot":
            model, tokenizer = load_cot_reasoner(
                config,
                checkpoint_path=resume_checkpoint or None,
                is_trainable=True,
            )
        elif config["method"] == "no_cot":
            model, tokenizer = load_no_cot_reasoner(
                config,
                checkpoint_path=resume_checkpoint or None,
                is_trainable=True,
            )
        elif config["method"] in {"compression", "curriculum_compression"}:
            model, tokenizer = load_compression_reasoner(
                config,
                checkpoint_path=resume_checkpoint or None,
                is_trainable=True,
            )
        elif config["method"] in {
            "reconstruction",
            "curriculum_reconstruction",
        }:
            model, tokenizer = load_reconstruction_reasoner(
                config,
                checkpoint_path=resume_checkpoint or None,
                is_trainable=True,
            )
        elif config["method"] in {
            "token_reconstruction",
            "curriculum_token_reconstruction",
        }:
            model, tokenizer = load_token_reconstruction_reasoner(
                config,
                checkpoint_path=resume_checkpoint or None,
                is_trainable=True,
            )
        else:
            model, tokenizer = load_latent_reasoner(
                config,
                checkpoint_path=resume_checkpoint or None,
                is_trainable=True,
            )
        reset_epochs, save_curriculum_stage = _curriculum_policy(config)
        resources = _train_phase(
            config,
            accelerator,
            train_records,
            validation_records,
            output_dir=output_dir,
            num_train_epochs=training_config["num_train_epochs"],
            model=model,
            tokenizer=tokenizer,
            collator=lambda epoch, split: _collator(
                config,
                tokenizer,
                prompt_template,
                epoch,
                response_template=response_template,
            ),
            method=config["method"],
            resume_checkpoint=resume_checkpoint,
            reset_epochs=reset_epochs,
            save_curriculum_stage=save_curriculum_stage,
        )
        return resources[0]
    finally:
        accelerator.end_training()
