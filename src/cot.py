from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class CotCollator:
    def __init__(
        self,
        tokenizer,
        prompt_template: str,
        response_template: str,
        data_config: dict[str, Any],
        model_max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.response_template = response_template
        self.data_config = data_config
        self.model_max_length = model_max_length

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _truncate(sequence: list[int], limit: int) -> list[int]:
        return sequence[: max(0, limit)]

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
        full_ids = []
        labels = []
        prompt_ids = []
        response_ids = []
        answer_masks = []
        max_source_length = self.data_config["max_source_length"]
        max_target_length = self.data_config["max_target_length"]

        for record in records:
            prompt = self.prompt_template.format(
                question=record["question"], response=""
            )
            response_prefix = self.response_template.format(
                cot=record["cot"], answer=""
            )
            full_response = self.response_template.format(
                cot=record["cot"], answer=record["answer"]
            )
            if not full_response.startswith(response_prefix):
                raise ValueError(
                    "CoT response template must place {answer} at the end"
                )
            answer = full_response[len(response_prefix) :]

            encoded_prompt = self._encode(prompt)
            prompt_limit = min(
                max_source_length, max(0, self.model_max_length - 1)
            )
            encoded_prompt = self._truncate(encoded_prompt, prompt_limit)
            prefix_ids = self._encode(response_prefix)
            answer_ids = self._encode(answer)

            response_budget = min(
                max_target_length - 1,
                self.model_max_length - len(encoded_prompt) - 1,
            )
            response_body = self._truncate(
                prefix_ids + answer_ids, response_budget
            )
            response = response_body + [self.tokenizer.eos_token_id]
            body_answer_mask = [
                index >= len(prefix_ids)
                for index in range(len(response_body))
            ]
            answer_mask = body_answer_mask + [True]

            prompt_ids.append(encoded_prompt)
            response_ids.append(response)
            answer_masks.append(answer_mask)
            full_ids.append(encoded_prompt + response)
            labels.append([-100] * len(encoded_prompt) + response)

        input_ids, attention_mask = self._pad(
            full_ids, self.tokenizer.pad_token_id
        )
        label_ids, _ = self._pad(labels, -100)
        prompt_input_ids, prompt_attention_mask = self._pad(
            prompt_ids, self.tokenizer.pad_token_id, left=True
        )
        suffix_input_ids, suffix_attention_mask = self._pad(
            response_ids, self.tokenizer.pad_token_id
        )
        answer_mask, _ = self._pad(answer_masks, 0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "suffix_input_ids": suffix_input_ids,
            "suffix_attention_mask": suffix_attention_mask,
            "answer_mask": answer_mask.bool(),
            "latent_count": 0,
        }


class CotReasoner(nn.Module):
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
        response_loss_sums = (
            token_losses * target_mask.to(token_losses.dtype)
        ).sum(dim=1)
        response_token_counts = target_mask.sum(dim=1)
        response_loss_sum = response_loss_sums.sum()
        response_token_count = response_token_counts.sum()
        response_loss = response_loss_sum / response_token_count.clamp_min(1)
        return {
            "loss": response_loss,
            "response_loss": response_loss,
            "response_loss_sum": response_loss_sum,
            "response_token_count": response_token_count,
            "response_loss_sums": response_loss_sums,
            "response_token_counts": response_token_counts,
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
        zero_lengths = torch.zeros(
            prompt_input_ids.shape[0],
            dtype=torch.long,
            device=prompt_input_ids.device,
        )
        return generated_ids, zero_lengths


def load_cot_reasoner(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    is_trainable: bool = True,
) -> tuple[CotReasoner, Any]:
    if checkpoint_path is None:
        adapter_path = None
        tokenizer_path = config["base_model"]
        base_model_name = config["base_model"]
    else:
        checkpoint_path = Path(checkpoint_path)
        adapter_path = checkpoint_path / "adapter"
        tokenizer_path = checkpoint_path / "tokenizer"
        adapter_config = PeftConfig.from_pretrained(adapter_path)
        base_model_name = adapter_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=config["training"]["model_max_length"],
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

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

    if adapter_path is None:
        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            target_modules=config["lora"]["target_modules"],
            lora_dropout=config["lora"]["dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(base_model, lora_config)
    else:
        llm = PeftModel.from_pretrained(
            base_model, adapter_path, is_trainable=is_trainable
        )

    return (
        CotReasoner(
            llm,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ),
        tokenizer,
    )
