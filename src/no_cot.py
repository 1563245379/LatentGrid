from pathlib import Path
from typing import Any

import torch
from peft import PeftConfig, PeftModel
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import render_no_cot


class NoCotCollator:
    def __init__(
        self,
        tokenizer,
        prompt_template: str,
        response_template: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.response_template = response_template

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _pad(
        sequences: list[list[int]], pad_id: int, left: bool = False
    ) -> tuple[Tensor, Tensor]:
        width = max(map(len, sequences))
        padded, masks = [], []
        for sequence in sequences:
            padding = width - len(sequence)
            pads = [pad_id] * padding
            zeros = [0] * padding
            ones = [1] * len(sequence)
            padded.append(pads + sequence if left else sequence + pads)
            masks.append(zeros + ones if left else ones + zeros)
        return torch.tensor(padded), torch.tensor(masks)

    def __call__(self, records: list[dict]) -> dict[str, Tensor | int]:
        prompt_texts = [
            render_no_cot(
                self.prompt_template,
                self.response_template,
                record["question"],
            )
            for record in records
        ]
        full_texts = [
            render_no_cot(
                self.prompt_template,
                self.response_template,
                record["question"],
                answer=record["answer"],
            )
            for record in records
        ]
        answer_texts = []
        for prompt, full in zip(prompt_texts, full_texts):
            if not full.startswith(prompt):
                raise ValueError(
                    "no_cot response template must place {answer} at the end"
                )
            answer_texts.append(full[len(prompt) :])

        prompt_ids = [self._encode(text) for text in prompt_texts]
        answer_ids = [
            self._encode(text) + [self.tokenizer.eos_token_id]
            for text in answer_texts
        ]
        full_ids = [
            prompt + answer
            for prompt, answer in zip(prompt_ids, answer_ids)
        ]
        labels = [
            [-100] * len(prompt) + answer
            for prompt, answer in zip(prompt_ids, answer_ids)
        ]

        input_ids, attention_mask = self._pad(
            full_ids, self.tokenizer.pad_token_id
        )
        label_ids, _ = self._pad(labels, -100)
        prompt_input_ids, prompt_attention_mask = self._pad(
            prompt_ids, self.tokenizer.pad_token_id, left=True
        )
        suffix_input_ids, suffix_attention_mask = self._pad(
            answer_ids, self.tokenizer.pad_token_id
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "suffix_input_ids": suffix_input_ids,
            "suffix_attention_mask": suffix_attention_mask,
            "answer_mask": suffix_attention_mask.bool(),
            "latent_count": 0,
        }


class NoCotReasoner(nn.Module):
    def __init__(
        self,
        llm: nn.Module,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.llm = llm
        config = llm.config
        self.eos_token_id = (
            eos_token_id
            if eos_token_id is not None
            else config.eos_token_id
        )
        self.pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else config.pad_token_id
        )
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id

    def save_trainable(self, checkpoint_dir, tokenizer) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.llm.save_pretrained(checkpoint_dir / "adapter")
        tokenizer.save_pretrained(checkpoint_dir / "tokenizer")

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        outputs = self.llm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits[:, :-1]
        target_ids = batch["input_ids"][:, 1:]
        target_mask = batch["labels"][:, 1:].ne(-100)
        token_losses = F.cross_entropy(
            logits.flatten(0, 1),
            target_ids.flatten(),
            reduction="none",
        ).view_as(target_ids)
        answer_loss_sums = (
            token_losses * target_mask.to(token_losses.dtype)
        ).sum(dim=1)
        answer_token_counts = target_mask.sum(dim=1)
        answer_loss_sum = answer_loss_sums.sum()
        answer_token_count = answer_token_counts.sum()
        answer_loss = answer_loss_sum / answer_token_count.clamp_min(1)
        return {
            "loss": answer_loss,
            "answer_loss": answer_loss,
            "answer_loss_sum": answer_loss_sum,
            "answer_token_count": answer_token_count,
            "answer_loss_sums": answer_loss_sums,
            "answer_token_counts": answer_token_counts,
        }

    @torch.inference_mode()
    def generate_answer_ids(
        self,
        prompt_input_ids: Tensor,
        prompt_attention_mask: Tensor,
        latent_count: int,
        answer_prefix_ids: Tensor,
        max_new_tokens: int = 64,
    ) -> tuple[Tensor, Tensor]:
        del latent_count, answer_prefix_ids
        prompt_width = prompt_input_ids.shape[1]
        output_ids = self.llm.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        generated_ids = output_ids[:, prompt_width:]
        latent_lengths = torch.zeros(
            prompt_input_ids.shape[0],
            dtype=torch.long,
            device=prompt_input_ids.device,
        )
        return generated_ids, latent_lengths


def load_no_cot_reasoner(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    is_trainable: bool = True,
) -> tuple[NoCotReasoner, Any]:
    if checkpoint_path is None:
        adapter_path = Path(config["checkpoint"]["cot_init_checkpoint"])
        tokenizer_path = adapter_path
        base_model_name = config["base_model"]
    else:
        checkpoint_path = Path(checkpoint_path)
        adapter_path = checkpoint_path / "adapter"
        tokenizer_path = checkpoint_path / "tokenizer"
        adapter_config = PeftConfig.from_pretrained(adapter_path)
        base_model_name = adapter_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "dtype": torch.bfloat16
        if config["training"]["bf16"]
        else torch.float32
    }
    if not is_trainable:
        load_kwargs["device_map"] = "auto"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, **load_kwargs
    )
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    llm = PeftModel.from_pretrained(
        base_model, adapter_path, is_trainable=is_trainable
    )
    return (
        NoCotReasoner(
            llm,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ),
        tokenizer,
    )
