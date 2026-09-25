from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_cot_model(
    base_model: str,
    adapter_path: str | Path,
    tokenizer_path: str | Path | None = None,
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or adapter_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_continuations(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    add_special_tokens: bool | None = None,
) -> list[str]:
    generations = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        tokenizer_kwargs = {"return_tensors": "pt", "padding": True}
        if add_special_tokens is not None:
            tokenizer_kwargs["add_special_tokens"] = add_special_tokens
        encoded = tokenizer(batch_prompts, **tokenizer_kwargs)
        encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}
        prompt_width = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        continuation_ids = output_ids[:, prompt_width:]
        generations.extend(
            tokenizer.batch_decode(continuation_ids, skip_special_tokens=True)
        )
    return generations
