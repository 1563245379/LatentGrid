from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

from src.direct_curriculum import (
    build_reconstruction_curriculum_plan,
)
from src.indirect import IndirectCollator, parse_cot_steps
from src.latent_model import (
    load_latent_reasoner,
    position_ids_from_mask,
)
from src.lora_training import (
    attach_lora_training_metadata,
    configure_reasoner_lora,
)


def build_reconstruction_text_targets(
    record: dict, group_count: int
) -> tuple[str, ...]:
    steps = parse_cot_steps(record["cot"])
    targets = []
    for group_index in range(group_count):
        if group_index == group_count - 1 and group_index < len(steps):
            targets.append("\n".join(steps[group_index:]))
        elif group_index < len(steps):
            targets.append(steps[group_index])
        else:
            targets.append(record["answer"])
    return tuple(targets)


def _attach_reconstruction_targets(tokenizer, batch, text_targets):
    encoded = [
        [
            tokenizer.encode(target, add_special_tokens=False)
            + [tokenizer.eos_token_id]
            for target in sample
        ]
        for sample in text_targets
    ]
    width = max(len(target) for sample in encoded for target in sample)
    ids = []
    masks = []
    for sample in encoded:
        sample_ids = []
        sample_masks = []
        for target in sample:
            padding = width - len(target)
            sample_ids.append(target + [tokenizer.pad_token_id] * padding)
            sample_masks.append([True] * len(target) + [False] * padding)
        ids.append(sample_ids)
        masks.append(sample_masks)
    batch["reconstruction_target_ids"] = torch.tensor(ids, dtype=torch.long)
    batch["reconstruction_target_mask"] = torch.tensor(
        masks, dtype=torch.bool
    )
    return batch


class ReconstructionCollator:
    def __init__(self, tokenizer, prompt_template: str, latent_config: dict):
        self.tokenizer = tokenizer
        self.latent_config = latent_config
        self.answer_collator = IndirectCollator(
            tokenizer,
            prompt_template,
            method="answer",
            epoch=1,
            latent_config=latent_config,
        )

    def __call__(self, records):
        batch = self.answer_collator(records)
        group_count = self.latent_config["num_latent_tokens"] // 2
        return _attach_reconstruction_targets(
            self.tokenizer,
            batch,
            [
                build_reconstruction_text_targets(record, group_count)
                for record in records
            ],
        )


class CurriculumReconstructionCollator:
    def __init__(
        self, tokenizer, prompt_template, latent_config, curriculum_config, epoch
    ):
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.curriculum_config = curriculum_config
        self.main_collator = IndirectCollator(
            tokenizer,
            prompt_template,
            method="curriculum",
            epoch=epoch,
            latent_config=latent_config,
            curriculum_config=curriculum_config,
        )

    def __call__(self, records):
        batch = self.main_collator(records)
        plans = [
            build_reconstruction_curriculum_plan(
                record,
                self.epoch,
                self.curriculum_config["stage_epochs"],
                self.curriculum_config["latent_tokens_per_stage"],
            )
            for record in records
        ]
        return _attach_reconstruction_targets(
            self.tokenizer,
            batch,
            [plan["reconstruction_targets"] for plan in plans],
        )


class ReconstructionDecoder(nn.Module):
    def __init__(
        self,
        llm: nn.Module,
        latent_prefix_width: int = 2,
        decoder_batch_size: int | None = None,
    ):
        super().__init__()
        if (
            decoder_batch_size is not None
            and (
                isinstance(decoder_batch_size, bool)
                or not isinstance(decoder_batch_size, Integral)
                or decoder_batch_size <= 0
            )
        ):
            raise ValueError(
                "decoder_batch_size must be None or a positive integer"
            )
        self.llm = llm
        self.latent_prefix_width = latent_prefix_width
        self.decoder_batch_size = (
            int(decoder_batch_size)
            if decoder_batch_size is not None
            else None
        )

    def _forward_chunk(
        self,
        latent_groups: Tensor,
        target_ids: Tensor,
        target_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        target_embeddings = self.llm.get_input_embeddings()(target_ids)
        inputs_embeds = torch.cat([latent_groups, target_embeddings], dim=1)
        prefix_mask = torch.ones(
            (target_mask.shape[0], self.latent_prefix_width),
            dtype=target_mask.dtype,
            device=target_mask.device,
        )
        attention_mask = torch.cat([prefix_mask, target_mask], dim=1)
        labels = torch.cat(
            [
                torch.full(
                    (target_ids.shape[0], self.latent_prefix_width),
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
        return (token_losses * valid).sum(dim=1), valid.sum(dim=1)

    def forward(
        self,
        latents: Tensor,
        target_ids: Tensor,
        target_mask: Tensor,
    ) -> dict[str, Tensor]:
        batch_size, group_count, target_width = target_ids.shape
        hidden_size = latents.shape[-1]
        latent_groups = latents.reshape(
            batch_size, group_count, self.latent_prefix_width, hidden_size
        ).reshape(
            batch_size * group_count, self.latent_prefix_width, hidden_size
        )
        flat_target_ids = target_ids.reshape(
            batch_size * group_count, target_width
        )
        flat_target_mask = target_mask.reshape(
            batch_size * group_count, target_width
        )
        total_group_count = batch_size * group_count
        if (
            self.decoder_batch_size is None
            or total_group_count <= self.decoder_batch_size
        ):
            group_loss_sums, group_token_counts = self._forward_chunk(
                latent_groups, flat_target_ids, flat_target_mask
            )
        else:
            loss_parts = []
            token_count_parts = []
            for start in range(0, total_group_count, self.decoder_batch_size):
                end = min(start + self.decoder_batch_size, total_group_count)
                group_loss_sums, group_token_counts = self._forward_chunk(
                    latent_groups[start:end],
                    flat_target_ids[start:end],
                    flat_target_mask[start:end],
                )
                loss_parts.append(group_loss_sums)
                token_count_parts.append(group_token_counts)
            group_loss_sums = torch.cat(loss_parts)
            group_token_counts = torch.cat(token_count_parts)
        loss_sums = group_loss_sums.reshape(batch_size, group_count).sum(dim=1)
        token_counts = group_token_counts.reshape(
            batch_size, group_count
        ).sum(dim=1)
        return {
            "reconstruction_loss": loss_sums.sum()
            / token_counts.sum().clamp_min(1),
            "reconstruction_loss_sums": loss_sums,
            "reconstruction_token_counts": token_counts,
        }


class ReconstructionReasoner(nn.Module):
    def __init__(
        self,
        reasoner: nn.Module,
        decoder: ReconstructionDecoder,
        include_cot_loss: bool = False,
    ) -> None:
        super().__init__()
        self.reasoner = reasoner
        self.decoder = decoder
        self.include_cot_loss = include_cot_loss

    def forward(self, batch, retain_latent_grad=False):
        main_outputs = self.reasoner(
            batch, retain_latent_grad=retain_latent_grad
        )
        reconstruction_outputs = self.decoder(
            torch.stack(main_outputs["latents"], dim=1),
            batch["reconstruction_target_ids"],
            batch["reconstruction_target_mask"],
        )
        result = {
            "answer_loss": main_outputs["answer_loss"],
            "reconstruction_loss": reconstruction_outputs[
                "reconstruction_loss"
            ],
            "answer_loss_sum": main_outputs["answer_loss_sum"],
            "answer_token_count": main_outputs["answer_token_count"],
            "reconstruction_loss_sum": reconstruction_outputs[
                "reconstruction_loss_sums"
            ].sum(),
            "reconstruction_token_count": reconstruction_outputs[
                "reconstruction_token_counts"
            ].sum(),
            "answer_loss_sums": main_outputs["answer_loss_sums"],
            "answer_token_counts": main_outputs["answer_token_counts"],
            "reconstruction_loss_sums": reconstruction_outputs[
                "reconstruction_loss_sums"
            ],
            "reconstruction_token_counts": reconstruction_outputs[
                "reconstruction_token_counts"
            ],
            "latents": main_outputs["latents"],
        }
        total_loss = main_outputs["answer_loss"] + reconstruction_outputs[
            "reconstruction_loss"
        ]
        if self.include_cot_loss:
            total_loss = total_loss + main_outputs["cot_loss"]
            result["cot_loss"] = main_outputs["cot_loss"]
            result["cot_loss_sums"] = main_outputs["cot_loss_sums"]
            result["cot_token_counts"] = main_outputs["cot_token_counts"]
        result["loss"] = total_loss
        return result

    def save_trainable(self, checkpoint_dir, tokenizer) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.reasoner.save_trainable(checkpoint_dir, tokenizer)
        self.decoder.llm.save_pretrained(checkpoint_dir / "decoder_adapter")


def load_reconstruction_decoder(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    is_trainable: bool = True,
    latent_prefix_width: int = 2,
    decoder_batch_size: int | None = None,
) -> ReconstructionDecoder:
    dtype = torch.bfloat16 if config["training"]["bf16"] else torch.float32
    adapter_path = (
        Path(checkpoint_path) / "decoder_adapter"
        if checkpoint_path is not None
        else None
    )
    base_model_path = config["base_model"]
    if adapter_path is not None:
        base_model_path = PeftConfig.from_pretrained(
            adapter_path
        ).base_model_name_or_path
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, dtype=dtype
    )
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    if adapter_path is not None:
        llm = PeftModel.from_pretrained(
            base_model, adapter_path, is_trainable=is_trainable
        )
    else:
        decoder_lora = config["decoder_lora"]
        llm = get_peft_model(
            base_model,
            LoraConfig(
                r=decoder_lora["r"],
                lora_alpha=decoder_lora["lora_alpha"],
                lora_dropout=decoder_lora["lora_dropout"],
                target_modules=decoder_lora.get("target_modules"),
                task_type="CAUSAL_LM",
                bias="none",
            ),
        )
        if not is_trainable:
            for parameter in llm.parameters():
                parameter.requires_grad_(False)
    return ReconstructionDecoder(
        llm,
        latent_prefix_width=latent_prefix_width,
        decoder_batch_size=decoder_batch_size,
    )


def load_reconstruction_reasoner(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    is_trainable: bool = True,
):
    reasoner, tokenizer = load_latent_reasoner(
        config,
        checkpoint_path=checkpoint_path,
        is_trainable=is_trainable,
    )
    selection = configure_reasoner_lora(
        reasoner.llm,
        config,
        checkpoint_path=checkpoint_path,
        is_trainable=is_trainable,
    )
    decoder = load_reconstruction_decoder(
        config,
        checkpoint_path=checkpoint_path,
        is_trainable=is_trainable,
        decoder_batch_size=config.get("training", {}).get("decoder_batch_size"),
    )
    model = ReconstructionReasoner(
        reasoner,
        decoder,
        include_cot_loss=config.get("method") == "curriculum_reconstruction",
    )
    if is_trainable:
        attach_lora_training_metadata(model, selection)
    return model, tokenizer
