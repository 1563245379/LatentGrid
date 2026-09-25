from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.compression import (
    CompressionCollator,
    CurriculumCompressionCollator,
)
from src.compression import sample_source_token_targets
from src.latent_model import (
    load_latent_reasoner,
    masked_token_mean,
    position_ids_from_mask,
    shifted_suffix_logits,
)
from src.lora_training import (
    attach_lora_training_metadata,
    configure_reasoner_lora,
)
from src.reconstruction import ReconstructionDecoder, load_reconstruction_decoder


def attach_token_reconstruction_targets(tokenizer, batch: dict) -> dict:
    source_ids = batch["source_group_ids"]
    source_mask = batch["source_group_mask"]
    batch_size, group_count, source_width = source_ids.shape
    target_ids = torch.full(
        (batch_size, group_count, source_width + 1),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )
    target_mask = torch.zeros(
        (batch_size, group_count, source_width + 1), dtype=torch.bool
    )
    for batch_index in range(batch_size):
        for group_index in range(group_count):
            count = int(source_mask[batch_index, group_index].sum().item())
            if count == 0:
                continue
            target_ids[batch_index, group_index, :count] = source_ids[
                batch_index, group_index, :count
            ]
            target_ids[batch_index, group_index, count] = tokenizer.eos_token_id
            target_mask[batch_index, group_index, : count + 1] = True
    batch["reconstruction_target_ids"] = target_ids
    batch["reconstruction_target_mask"] = target_mask
    return batch


class TokenReconstructionCollator(CompressionCollator):
    def __call__(self, records):
        return attach_token_reconstruction_targets(
            self.tokenizer, super().__call__(records)
        )


class CurriculumTokenReconstructionCollator(CurriculumCompressionCollator):
    def __call__(self, records):
        return attach_token_reconstruction_targets(
            self.tokenizer, super().__call__(records)
        )


class TokenReconstructionReasoner(nn.Module):
    def __init__(
        self,
        reasoner: nn.Module,
        decoder: ReconstructionDecoder,
        enable_token_ce: bool,
        include_cot_loss: bool = False,
    ) -> None:
        super().__init__()
        self.reasoner = reasoner
        self.decoder = decoder
        self.enable_token_ce = enable_token_ce
        self.include_cot_loss = include_cot_loss

    def forward(self, batch, retain_latent_grad=False):
        latent_mask = batch["latent_mask"]
        prompt_attention_mask = batch["prompt_attention_mask"]
        outputs = self.reasoner.llm(
            inputs_embeds=self.reasoner.embed(batch["prompt_input_ids"]),
            attention_mask=prompt_attention_mask,
            position_ids=position_ids_from_mask(prompt_attention_mask),
            output_hidden_states=True,
            use_cache=True,
        )

        attention_mask = prompt_attention_mask
        last_hidden = outputs.hidden_states[-1][:, -1]
        last_base_logits = outputs.logits[:, -1]
        group_count = latent_mask.shape[1]
        source_targets = None
        source_loss_sums = last_hidden.new_zeros(last_hidden.shape[0])
        if self.enable_token_ce and group_count:
            source_targets = sample_source_token_targets(
                batch["source_group_ids"], batch["source_group_mask"]
            )
        latents = []
        for group_index in range(group_count):
            predictor_hidden = outputs.hidden_states[-1][:, -1]
            predictor_base_logits = outputs.logits[:, -1]
            active = latent_mask[:, group_index]
            if self.enable_token_ce:
                source_logits = self.reasoner.boundary.append_logits(
                    predictor_hidden, predictor_base_logits
                )
                source_losses = F.cross_entropy(
                    source_logits,
                    source_targets[:, group_index],
                    reduction="none",
                )
                source_loss_sums = source_loss_sums + source_losses * active.to(
                    source_losses.dtype
                )
            latent = self.reasoner.next_latent(
                predictor_hidden, predictor_base_logits
            )
            if retain_latent_grad:
                latent.retain_grad()
            latents.append(latent)
            active_column = active.to(attention_mask.dtype).unsqueeze(1)
            attention_mask = torch.cat(
                [attention_mask, active_column], dim=1
            )
            outputs = self.reasoner.llm(
                inputs_embeds=(
                    latent * active.to(latent.dtype).unsqueeze(1)
                ).unsqueeze(1),
                attention_mask=attention_mask,
                position_ids=position_ids_from_mask(attention_mask)[:, -1:],
                past_key_values=outputs.past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )
            last_hidden = torch.where(
                active.unsqueeze(1), outputs.hidden_states[-1][:, -1], last_hidden
            )
            last_base_logits = torch.where(
                active.unsqueeze(1), outputs.logits[:, -1], last_base_logits
            )

        suffix_attention_mask = batch["suffix_attention_mask"]
        full_attention_mask = torch.cat(
            [attention_mask, suffix_attention_mask], dim=1
        )
        suffix_outputs = self.reasoner.llm(
            inputs_embeds=self.reasoner.embed(batch["suffix_input_ids"]),
            attention_mask=full_attention_mask,
            position_ids=position_ids_from_mask(full_attention_mask)[
                :, -suffix_attention_mask.shape[1] :
            ],
            past_key_values=outputs.past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        suffix_logits = self.reasoner.boundary.append_logits(
            suffix_outputs.hidden_states[-1], suffix_outputs.logits
        )
        predictor_logits = shifted_suffix_logits(
            self.reasoner.boundary.append_logits(last_hidden, last_base_logits),
            suffix_logits,
        )
        token_losses = F.cross_entropy(
            predictor_logits.flatten(0, 1),
            batch["suffix_input_ids"].flatten(),
            reduction="none",
        ).view_as(batch["suffix_input_ids"])

        answer_mask = batch["answer_mask"]
        answer_loss_sums = (
            token_losses * answer_mask.to(token_losses.dtype)
        ).sum(dim=1)
        answer_token_counts = answer_mask.sum(dim=1)
        answer_loss = masked_token_mean(token_losses, answer_mask)

        if group_count:
            reconstruction_outputs = self.decoder(
                torch.stack(latents, dim=1),
                batch["reconstruction_target_ids"],
                batch["reconstruction_target_mask"],
            )
            reconstruction_loss = reconstruction_outputs[
                "reconstruction_loss"
            ]
            reconstruction_loss_sums = reconstruction_outputs[
                "reconstruction_loss_sums"
            ]
            reconstruction_token_counts = reconstruction_outputs[
                "reconstruction_token_counts"
            ]
        else:
            reconstruction_loss_sums = answer_loss.new_zeros(
                batch["latent_counts"].shape[0]
            )
            reconstruction_token_counts = batch["latent_counts"].new_zeros(
                batch["latent_counts"].shape[0]
            )
            reconstruction_loss = answer_loss * 0.0
            skipped_parameters = self.decoder.parameters()
            if self.reasoner.latent_head is not None:
                skipped_parameters = (
                    *skipped_parameters,
                    *self.reasoner.latent_head.parameters(),
                )
            for parameter in skipped_parameters:
                if parameter.requires_grad:
                    reconstruction_loss = (
                        reconstruction_loss + parameter.sum() * 0.0
                    )

        result = {
            "loss": answer_loss + reconstruction_loss,
            "answer_loss": answer_loss,
            "answer_loss_sums": answer_loss_sums,
            "answer_token_counts": answer_token_counts,
            "reconstruction_loss": reconstruction_loss,
            "reconstruction_loss_sums": reconstruction_loss_sums,
            "reconstruction_token_counts": reconstruction_token_counts,
            "latents": latents,
        }
        if self.enable_token_ce:
            end_logits = self.reasoner.boundary.append_logits(
                last_hidden, last_base_logits
            )
            end_targets = torch.full_like(
                batch["latent_counts"], self.reasoner.end_latent_id
            )
            end_losses = F.cross_entropy(
                end_logits, end_targets, reduction="none"
            )
            token_ce_loss_sums = source_loss_sums + end_losses
            token_ce_counts = batch["latent_counts"] + 1
            token_ce_loss = token_ce_loss_sums.sum() / token_ce_counts.sum().clamp_min(
                1
            )
            result["loss"] = result["loss"] + token_ce_loss
            result.update(
                {
                    "token_ce_loss": token_ce_loss,
                    "token_ce_loss_sums": token_ce_loss_sums,
                    "token_ce_counts": token_ce_counts,
                }
            )
        if self.include_cot_loss:
            cot_mask = batch["cot_mask"]
            cot_loss_sums = (
                token_losses * cot_mask.to(token_losses.dtype)
            ).sum(dim=1)
            cot_token_counts = cot_mask.sum(dim=1)
            cot_loss = masked_token_mean(token_losses, cot_mask)
            result["loss"] = result["loss"] + cot_loss
            result.update(
                {
                    "cot_loss": cot_loss,
                    "cot_loss_sums": cot_loss_sums,
                    "cot_token_counts": cot_token_counts,
                }
            )
        return result

    def save_trainable(self, checkpoint_dir, tokenizer) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        self.reasoner.save_trainable(checkpoint_dir, tokenizer)
        self.decoder.llm.save_pretrained(checkpoint_dir / "decoder_adapter")


def load_token_reconstruction_reasoner(
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
        latent_prefix_width=1,
        decoder_batch_size=config.get("training", {}).get("decoder_batch_size"),
    )
    model = TokenReconstructionReasoner(
        reasoner,
        decoder,
        enable_token_ce=config["latent"]["enable_token_ce"],
        include_cot_loss=(
            config["method"] == "curriculum_token_reconstruction"
        ),
    )
    if is_trainable:
        attach_lora_training_metadata(model, selection)
    return model, tokenizer
