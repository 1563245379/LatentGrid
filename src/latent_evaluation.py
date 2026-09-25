from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.compression import load_compression_reasoner
from src.data import (
    load_jsonl,
    load_prompt_response_templates,
    load_prompt_template,
    render_no_cot,
    render_prompt,
)
from src.evaluation import (
    ANSWER_MARKER,
    score_generations,
    score_latent_generations,
    write_results,
)
from src.latent_model import START_LATENT_TOKEN, load_latent_reasoner
from src.model import generate_continuations, load_cot_model


def evaluate_from_config(config: dict, checkpoint_path=None) -> dict:
    records = load_jsonl(config["data"]["eval_path"])
    max_eval_samples = config["data"]["max_eval_samples"]
    if max_eval_samples is not None:
        records = records[:max_eval_samples]

    resolved_checkpoint = (
        checkpoint_path
        if checkpoint_path is not None
        else config["checkpoint"]["eval_checkpoint"]
    )
    if resolved_checkpoint is None:
        raise ValueError("checkpoint.eval_checkpoint is required for evaluation")
    if config["method"] in {"cot", "no_cot"}:
        checkpoint_dir = Path(resolved_checkpoint)
        model, tokenizer = load_cot_model(
            config["base_model"],
            checkpoint_dir / "adapter",
            tokenizer_path=checkpoint_dir / "tokenizer",
        )
        if config["method"] == "no_cot":
            prompt_template, response_template = load_prompt_response_templates(
                config["template_path"]
            )
            prompts = [
                render_no_cot(
                    prompt_template,
                    response_template,
                    record["question"],
                )
                for record in records
            ]
        else:
            prompt_template = load_prompt_template(config["template_path"])
            prompts = [
                render_prompt(prompt_template, record["question"])
                for record in records
            ]
        generations = generate_continuations(
            model,
            tokenizer,
            prompts,
            config["training"]["per_device_eval_batch_size"],
            config.get("evaluation", {}).get("max_new_tokens", 64),
            add_special_tokens=False,
        )
        predictions, metrics = score_generations(records, generations)
        write_results(
            Path(config["output_dir"]) / "eval", predictions, metrics
        )
        return metrics

    adaptive_methods = {
        "compression",
        "curriculum_compression",
        "token_reconstruction",
        "curriculum_token_reconstruction",
    }
    is_adaptive = (
        config["method"] in adaptive_methods
        and config["latent"]["enable_token_ce"]
    )
    loader = {
        "compression": load_compression_reasoner,
        "curriculum_compression": load_compression_reasoner,
    }.get(config["method"], load_latent_reasoner)
    reasoner, tokenizer = loader(
        config,
        checkpoint_path=resolved_checkpoint,
        is_trainable=False,
    )
    reasoner.eval()
    device = next(reasoner.parameters()).device

    prompt_template = load_prompt_template(config["template_path"])
    prompts = [
        render_prompt(prompt_template, record["question"]) + START_LATENT_TOKEN
        for record in records
    ]
    answer_prefix_ids = torch.tensor(
        tokenizer.encode(ANSWER_MARKER, add_special_tokens=False),
        dtype=torch.long,
        device=device,
    )
    batch_size = config["training"]["per_device_eval_batch_size"]
    generations = []
    latent_lengths = []
    latent_truncated = []
    for offset in tqdm(
        range(0, len(prompts), batch_size), desc="Evaluating", dynamic_ncols=True
    ):
        encoded = tokenizer(
            prompts[offset : offset + batch_size],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        if is_adaptive:
            generated_ids, lengths, truncated = reasoner.generate_adaptive_answer_ids(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
                answer_prefix_ids,
                max_latent_tokens=config["latent"]["max_latent_tokens"],
                max_new_tokens=64,
            )
            latent_truncated.extend(truncated.detach().cpu().tolist())
        else:
            generated_ids, lengths = reasoner.generate_answer_ids(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
                config["latent"].get("num_latent_tokens", 12),
                answer_prefix_ids,
            )
        decoded = tokenizer.batch_decode(
            generated_ids.detach().cpu(), skip_special_tokens=True
        )
        generations.extend(ANSWER_MARKER + text for text in decoded)
        latent_lengths.extend(lengths.detach().cpu().tolist())

    predictions, metrics = score_latent_generations(
        records,
        generations,
        latent_lengths,
        latent_truncated if is_adaptive else None,
    )
    write_results(Path(config["output_dir"]) / "eval", predictions, metrics)
    return metrics
