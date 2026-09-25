from pathlib import Path
from typing import Any

import torch
import yaml


def normalize_trainable_modules(value: object) -> list[str] | None:
    """Return a canonical main-model LoRA module selection."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(
            "reasoner_lora.trainable_modules must be null or a list of strings"
        )
    if any(not isinstance(module, str) or not module for module in value):
        raise ValueError(
            "reasoner_lora.trainable_modules must contain non-empty strings"
        )
    return sorted(set(value))


def lora_parameter_groups(
    llm: torch.nn.Module,
) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    """Group mounted LoRA parameters by their owning module type."""
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = {}
    for name, parameter in llm.named_parameters():
        if ".lora_" not in name:
            continue
        owner = name.split(".lora_", 1)[0]
        module_type = owner.rsplit(".", 1)[-1]
        groups.setdefault(module_type, []).append((name, parameter))
    return groups


def _configured_selection(config: dict[str, Any]) -> list[str] | None:
    reasoner_lora = config.get("reasoner_lora") or {}
    return normalize_trainable_modules(
        reasoner_lora.get("trainable_modules")
    )


def _historical_selection(
    checkpoint_path: str | Path,
) -> list[str] | None:
    config_path = Path(checkpoint_path) / "config.yaml"
    if not config_path.is_file():
        return None
    with config_path.open(encoding="utf-8") as stream:
        checkpoint_config = yaml.safe_load(stream) or {}
    return _configured_selection(checkpoint_config)


def _validate_selection(
    selection: list[str] | None,
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]],
) -> None:
    if selection is None:
        return
    unknown = sorted(set(selection) - set(groups))
    if unknown:
        raise ValueError(
            "reasoner_lora.trainable_modules contains unknown module(s) "
            f"{unknown}; available modules: {sorted(groups)}"
        )


def configure_reasoner_lora(
    llm: torch.nn.Module,
    config: dict,
    checkpoint_path: str | Path | None = None,
    is_trainable: bool = True,
) -> list[str] | None:
    """Apply the configured or historical selection to a loaded CoT adapter."""
    groups = lora_parameter_groups(llm)
    if not is_trainable:
        for parameters in groups.values():
            for _, parameter in parameters:
                parameter.requires_grad_(False)
        return None

    current = _configured_selection(config)
    historical = (
        _historical_selection(checkpoint_path)
        if checkpoint_path is not None
        else None
    )
    _validate_selection(current, groups)
    _validate_selection(historical, groups)

    if checkpoint_path is not None and current is not None:
        historical_set = set(groups) if historical is None else set(historical)
        if set(current) != historical_set:
            raise ValueError(
                "reasoner_lora.trainable_modules conflicts with the historical "
                f"selection: current={current}, historical="
                f"{historical if historical is not None else sorted(groups)}"
            )

    selection = historical if checkpoint_path is not None else current
    selected = set(groups) if selection is None else set(selection)
    for module_type, parameters in groups.items():
        trainable = module_type in selected
        for _, parameter in parameters:
            parameter.requires_grad_(trainable)
    return selection


def summarize_lora(llm: torch.nn.Module) -> dict:
    """Summarize mounted LoRA targets and actual trainable parameter counts."""
    groups = lora_parameter_groups(llm)
    trainable_modules = sorted(
        module_type
        for module_type, parameters in groups.items()
        if any(parameter.requires_grad for _, parameter in parameters)
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameters in groups.values()
        for _, parameter in parameters
        if parameter.requires_grad
    )
    total_parameters = sum(
        parameter.numel()
        for parameters in groups.values()
        for _, parameter in parameters
    )
    return {
        "target_modules": sorted(groups),
        "trainable_modules": trainable_modules,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
    }


def attach_lora_training_metadata(
    model: torch.nn.Module, selection: list[str] | None
) -> None:
    """Record the effective LoRA configuration and parameter summaries."""
    adapter = model.decoder.llm.peft_config["default"]
    targets = adapter.target_modules
    if isinstance(targets, set):
        targets = sorted(targets)
    model.lora_training_config = {
        "reasoner_lora": {"trainable_modules": selection},
        "decoder_lora": {
            "r": adapter.r,
            "lora_alpha": adapter.lora_alpha,
            "lora_dropout": adapter.lora_dropout,
            "target_modules": targets,
        },
    }
    model.lora_training_summary = {
        "reasoner": summarize_lora(model.reasoner.llm),
        "decoder": summarize_lora(model.decoder.llm),
    }
