import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.data import render_prompt
from src.direct_curriculum import (
    curriculum_stage,
    validate_tokens_per_latent,
    group_cot_with_steps,
    split_curriculum_groups,
)
from src.evaluation import ANSWER_MARKER
from src.latent_model import (
    END_LATENT_TOKEN,
    START_LATENT_TOKEN,
    load_latent_reasoner,
    masked_token_mean,
    position_ids_from_mask,
)


def group_cot_token_ids(
    token_ids: list[int], tokens_per_latent: int = 2
) -> tuple[tuple[int, ...], ...]:
    validate_tokens_per_latent(tokens_per_latent)
    return tuple(
        tuple(token_ids[offset : offset + tokens_per_latent])
        for offset in range(0, len(token_ids), tokens_per_latent)
    )


def sample_source_token_targets(
    source_ids: Tensor, source_mask: Tensor
) -> Tensor:
    counts = source_mask.sum(dim=-1)
    offsets = (
        torch.rand_like(counts, dtype=torch.float32) * counts.clamp_min(1)
    ).long()
    return source_ids.gather(dim=-1, index=offsets.unsqueeze(-1)).squeeze(-1)


def pool_group_embeddings(
    source_ids: Tensor,
    source_mask: Tensor,
    embedding_weight: Tensor,
) -> Tensor:
    source_embeddings = F.embedding(source_ids, embedding_weight)
    weights = source_mask.to(source_embeddings.dtype).unsqueeze(-1)
    counts = source_mask.sum(dim=-1).clamp_min(1).to(source_embeddings.dtype)
    return (source_embeddings * weights).sum(dim=-2) / counts.sqrt().unsqueeze(-1)


def apply_gumbel_noise(
    probabilities: Tensor,
    temperature: float,
    noise_scale: float,
    add_noise: bool,
) -> Tensor:
    probabilities = probabilities.float()
    probability_sums = probabilities.sum(dim=-1, keepdim=True)
    valid = probability_sums > 0
    probabilities = probabilities / probability_sums.clamp_min(1.0e-12)
    if not add_noise:
        return probabilities
    uniform = torch.rand_like(probabilities).clamp_min(1.0e-10)
    gumbel = -torch.log(-torch.log(uniform))
    noisy = torch.softmax(
        (
            torch.log(probabilities + 1.0e-10)
            + noise_scale * gumbel
        )
        / temperature,
        dim=-1,
    )
    return torch.where(valid, noisy, torch.zeros_like(noisy))


def build_sparse_teacher(
    raw_targets: Tensor,
    embedding_weight: Tensor,
    top_k: int,
    gumbel_temperature: float,
    gumbel_noise_scale: float,
    add_noise: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    logits = raw_targets.float() @ embedding_weight.float().t()
    top_logits, indices = torch.topk(logits, top_k, dim=-1)
    probabilities = torch.softmax(top_logits, dim=-1)
    probabilities = apply_gumbel_noise(
        probabilities,
        temperature=gumbel_temperature,
        noise_scale=gumbel_noise_scale,
        add_noise=add_noise,
    )
    top_embeddings = embedding_weight[indices]
    weighted = (
        probabilities.to(top_embeddings.dtype).unsqueeze(-1) * top_embeddings
    ).sum(dim=-2)
    return indices, probabilities, weighted


def sparse_teacher_kl(
    student_logits: Tensor,
    teacher_indices: Tensor,
    teacher_probs: Tensor,
    original_vocab_size: int,
) -> Tensor:
    student_log_probs = F.log_softmax(
        student_logits[..., :original_vocab_size].float(), dim=-1
    )
    selected = student_log_probs.gather(dim=-1, index=teacher_indices)
    return (
        teacher_probs
        * (torch.log(teacher_probs.clamp_min(1.0e-12)) - selected)
    ).sum(dim=-1)


class CompressionCollator:
    def __init__(self, tokenizer, prompt_template: str, tokens_per_latent: int = 2):
        validate_tokens_per_latent(tokens_per_latent)
        self.tokens_per_latent = tokens_per_latent
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template

    @staticmethod
    def _pad(sequences, pad_id, left=False):
        width = max(len(sequence) for sequence in sequences)
        ids, masks = [], []
        for sequence in sequences:
            padding = width - len(sequence)
            pads = [pad_id] * padding
            zeros = [0] * padding
            ones = [1] * len(sequence)
            ids.append(pads + sequence if left else sequence + pads)
            masks.append(zeros + ones if left else ones + zeros)
        return torch.tensor(ids), torch.tensor(masks)

    def _prompt_suffix(self, records, remaining_cot_ids=None):
        start_id = self.tokenizer.convert_tokens_to_ids(START_LATENT_TOKEN)
        end_id = self.tokenizer.convert_tokens_to_ids(END_LATENT_TOKEN)
        prompt_ids = [
            self.tokenizer.encode(
                render_prompt(self.prompt_template, record["question"]),
                add_special_tokens=False,
            )
            + [start_id]
            for record in records
        ]
        marker_ids = self.tokenizer.encode(ANSWER_MARKER, add_special_tokens=False)
        suffix_ids, answer_masks, cot_masks = [], [], []
        for index, record in enumerate(records):
            answer_ids = self.tokenizer.encode(
                record["answer"], add_special_tokens=False
            ) + [self.tokenizer.eos_token_id]
            remaining_ids = (
                () if remaining_cot_ids is None else remaining_cot_ids[index]
            )
            suffix_ids.append([end_id] + list(remaining_ids) + marker_ids + answer_ids)
            answer_masks.append(
                [False] * (1 + len(remaining_ids) + len(marker_ids))
                + [True] * len(answer_ids)
            )
            cot_masks.append(
                [False]
                + [True] * len(remaining_ids)
                + [False] * (len(marker_ids) + len(answer_ids))
            )
        prompt_input_ids, prompt_attention_mask = self._pad(
            prompt_ids, self.tokenizer.pad_token_id, left=True
        )
        suffix_input_ids, suffix_attention_mask = self._pad(
            suffix_ids, self.tokenizer.pad_token_id
        )
        suffix_width = suffix_input_ids.shape[1]
        answer_mask = torch.tensor(
            [mask + [False] * (suffix_width - len(mask)) for mask in answer_masks],
            dtype=torch.bool,
        )
        batch = {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "suffix_input_ids": suffix_input_ids,
            "suffix_attention_mask": suffix_attention_mask,
            "answer_mask": answer_mask,
        }
        if remaining_cot_ids is not None:
            batch["cot_mask"] = torch.tensor(
                [mask + [False] * (suffix_width - len(mask)) for mask in cot_masks],
                dtype=torch.bool,
            )
        return batch

    def _collate_groups(self, batch, groups):
        max_groups = max(len(sample_groups) for sample_groups in groups)
        source_group_ids = torch.full(
            (len(groups), max_groups, self.tokens_per_latent), self.tokenizer.pad_token_id
        )
        source_group_mask = torch.zeros(
            (len(groups), max_groups, self.tokens_per_latent), dtype=torch.bool
        )
        latent_mask = torch.zeros((len(groups), max_groups), dtype=torch.bool)
        for batch_index, sample_groups in enumerate(groups):
            for group_index, group in enumerate(sample_groups):
                source_group_ids[batch_index, group_index, : len(group)] = torch.tensor(group)
                source_group_mask[batch_index, group_index, : len(group)] = True
                latent_mask[batch_index, group_index] = True
        batch.update(
            {
                "source_group_ids": source_group_ids,
                "source_group_mask": source_group_mask,
                "latent_mask": latent_mask,
                "latent_counts": latent_mask.sum(dim=1),
            }
        )
        return batch

    def __call__(self, records):
        batch = self._prompt_suffix(records)
        groups = [
            group_cot_token_ids(
                self.tokenizer.encode(record["cot"], add_special_tokens=False),
                self.tokens_per_latent,
            )
            for record in records
        ]
        return self._collate_groups(batch, groups)


class CurriculumCompressionCollator(CompressionCollator):
    def __init__(self, tokenizer, prompt_template, epoch, curriculum_config, tokens_per_latent=2):
        super().__init__(tokenizer, prompt_template, tokens_per_latent)
        self.stage = curriculum_stage(epoch, curriculum_config["stage_epochs"])

    def __call__(self, records):
        grouped = [
            group_cot_with_steps(self.tokenizer, record["cot"], self.tokens_per_latent)
            for record in records
        ]
        selections = [
            split_curriculum_groups(groups, step_groups, self.stage)
            for groups, step_groups in grouped
        ]
        selected_groups = [
            tuple(groups[index] for index in selected)
            for (groups, _), (selected, _) in zip(grouped, selections)
        ]
        remaining_ids = [remaining for _, remaining in selections]
        batch = self._prompt_suffix(records, remaining_ids)
        return self._collate_groups(batch, selected_groups)


class CompressionReasoner(nn.Module):
    def __init__(
        self,
        reasoner: nn.Module,
        enable_token_ce: bool,
        fixed_variance: float,
        gumbel_temperature: float,
        gumbel_noise_scale: float,
    ) -> None:
        super().__init__()
        self.reasoner = reasoner
        self.enable_token_ce = enable_token_ce
        self.fixed_variance = fixed_variance
        self.gumbel_temperature = gumbel_temperature
        self.gumbel_noise_scale = gumbel_noise_scale
        embedding = reasoner.llm.get_input_embeddings().weight[
            : reasoner.original_vocab_size
        ]
        self.register_buffer(
            "embedding_scale",
            embedding.detach().float().std(correction=0),
            persistent=False,
        )

    def save_trainable(self, checkpoint_dir, tokenizer) -> None:
        self.reasoner.save_trainable(checkpoint_dir, tokenizer)

    @torch.inference_mode()
    def generate_adaptive_answer_ids(
        self,
        prompt_input_ids: Tensor,
        prompt_attention_mask: Tensor,
        answer_prefix_ids: Tensor,
        max_latent_tokens: int,
        max_new_tokens: int = 64,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if max_latent_tokens < 1:
            raise ValueError("max_latent_tokens must be at least 1")
        return self._generate_answer_ids(
            prompt_input_ids,
            prompt_attention_mask,
            answer_prefix_ids,
            max_latent_tokens,
            max_new_tokens,
            stop_on_end=True,
        )

    @torch.inference_mode()
    def generate_answer_ids(
        self,
        prompt_input_ids: Tensor,
        prompt_attention_mask: Tensor,
        latent_count: int,
        answer_prefix_ids: Tensor,
        max_new_tokens: int = 64,
    ) -> tuple[Tensor, Tensor]:
        generated, latent_lengths, _ = self._generate_answer_ids(
            prompt_input_ids,
            prompt_attention_mask,
            answer_prefix_ids,
            latent_count,
            max_new_tokens,
            stop_on_end=False,
        )
        return generated, latent_lengths

    def _generate_answer_ids(
        self,
        prompt_input_ids: Tensor,
        prompt_attention_mask: Tensor,
        answer_prefix_ids: Tensor,
        latent_limit: int,
        max_new_tokens: int,
        stop_on_end: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        outputs = self.reasoner.llm(
            inputs_embeds=self.reasoner.embed(prompt_input_ids),
            attention_mask=prompt_attention_mask,
            position_ids=position_ids_from_mask(prompt_attention_mask),
            output_hidden_states=True,
            use_cache=True,
        )
        attention_mask = prompt_attention_mask
        finished = torch.zeros(
            prompt_input_ids.shape[0],
            dtype=torch.bool,
            device=prompt_input_ids.device,
        )
        latent_lengths = torch.zeros_like(finished, dtype=torch.long)
        for _ in range(latent_limit):
            latent = self.reasoner.next_latent(
                outputs.hidden_states[-1][:, -1], outputs.logits[:, -1]
            )
            if self.reasoner.representation_type == "unconstrained":
                latent = latent * self.embedding_scale.to(latent.dtype)
            active = ~finished
            active_mask = active.to(attention_mask.dtype).unsqueeze(1)
            attention_mask = torch.cat([attention_mask, active_mask], dim=1)
            latent_lengths += active.long()
            outputs = self.reasoner.llm(
                inputs_embeds=latent.unsqueeze(1),
                attention_mask=attention_mask,
                position_ids=position_ids_from_mask(attention_mask)[:, -1:],
                past_key_values=outputs.past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )
            if stop_on_end:
                end_logits = self.reasoner.boundary.append_logits(
                    outputs.hidden_states[-1][:, -1], outputs.logits[:, -1]
                )
                finished |= active & end_logits.argmax(dim=-1).eq(
                    self.reasoner.end_latent_id
                )
                if finished.all():
                    break
        truncated = ~finished if stop_on_end else torch.zeros_like(finished)
        generated = self.reasoner._generate_answer_from_cache(
            outputs,
            attention_mask,
            answer_prefix_ids,
            max_new_tokens,
        )
        return generated, latent_lengths, truncated

    def _gold_targets(self, batch):
        embedding = self.reasoner.llm.get_input_embeddings().weight[
            : self.reasoner.original_vocab_size
        ]
        raw = pool_group_embeddings(
            batch["source_group_ids"], batch["source_group_mask"], embedding
        ).detach()
        if self.reasoner.representation_type == "unconstrained":
            return raw, raw.float() / self.embedding_scale.float(), None, None
        indices, probabilities, weighted = build_sparse_teacher(
            raw,
            embedding,
            self.reasoner.top_k,
            self.gumbel_temperature,
            self.gumbel_noise_scale,
            add_noise=self.training,
        )
        return weighted.detach(), None, indices, probabilities.detach()

    def forward(self, batch):
        gold_latents, normalized_targets, teacher_indices, teacher_probs = (
            self._gold_targets(batch)
        )
        latent_mask = batch["latent_mask"]
        gold_latents = gold_latents * latent_mask.unsqueeze(-1)
        prompt_embeds = self.reasoner.embed(batch["prompt_input_ids"])
        suffix_embeds = self.reasoner.embed(batch["suffix_input_ids"])
        inputs_embeds = torch.cat(
            [prompt_embeds, gold_latents.to(prompt_embeds.dtype), suffix_embeds],
            dim=1,
        )
        attention_mask = torch.cat(
            [
                batch["prompt_attention_mask"],
                latent_mask.to(batch["prompt_attention_mask"].dtype),
                batch["suffix_attention_mask"],
            ],
            dim=1,
        )
        outputs = self.reasoner.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids_from_mask(attention_mask),
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]
        base_logits = outputs.logits
        prompt_width = batch["prompt_input_ids"].shape[1]
        max_latents = latent_mask.shape[1]
        predictor_hidden = torch.cat(
            [
                hidden[:, prompt_width - 1 : prompt_width],
                hidden[:, prompt_width : prompt_width + max_latents - 1],
            ],
            dim=1,
        )
        predictor_logits = torch.cat(
            [
                base_logits[:, prompt_width - 1 : prompt_width],
                base_logits[:, prompt_width : prompt_width + max_latents - 1],
            ],
            dim=1,
        )
        if self.reasoner.representation_type == "unconstrained":
            means = self.reasoner.latent_head(predictor_hidden).float()
            sampled = means
            if self.training:
                sampled = sampled + math.sqrt(self.fixed_variance) * torch.randn_like(means)
            per_latent = (sampled - normalized_targets.float()).square().mean(dim=-1)
        else:
            per_latent = sparse_teacher_kl(
                predictor_logits,
                teacher_indices,
                teacher_probs,
                self.reasoner.original_vocab_size,
            )
        latent_loss_sums = (
            per_latent * latent_mask.to(per_latent.dtype)
        ).sum(dim=1)
        latent_counts = latent_mask.sum(dim=1)
        latent_loss = latent_loss_sums.sum() / latent_counts.sum().clamp_min(1)

        suffix_start = prompt_width + max_latents
        suffix_width = batch["suffix_input_ids"].shape[1]
        answer_hidden = hidden[:, suffix_start : suffix_start + suffix_width - 1]
        answer_base_logits = base_logits[
            :, suffix_start : suffix_start + suffix_width - 1
        ]
        answer_logits = self.reasoner.boundary.append_logits(
            answer_hidden, answer_base_logits
        )
        answer_targets = batch["suffix_input_ids"][:, 1:]
        answer_mask = batch["answer_mask"][:, 1:]
        suffix_token_losses = F.cross_entropy(
            answer_logits.flatten(0, 1),
            answer_targets.flatten(),
            reduction="none",
        ).view_as(answer_targets)
        answer_loss_sums = (
            suffix_token_losses * answer_mask.to(suffix_token_losses.dtype)
        ).sum(dim=1)
        answer_token_counts = answer_mask.sum(dim=1)
        answer_loss = masked_token_mean(suffix_token_losses, answer_mask)
        result = {
            "loss": latent_loss + answer_loss,
            "latent_loss": latent_loss,
            "answer_loss": answer_loss,
            "latent_loss_sums": latent_loss_sums,
            "latent_counts": latent_counts,
            "answer_loss_sums": answer_loss_sums,
            "answer_token_counts": answer_token_counts,
        }
        if self.enable_token_ce:
            source_targets = sample_source_token_targets(
                batch["source_group_ids"], batch["source_group_mask"]
            )
            source_logits = self.reasoner.boundary.append_logits(
                predictor_hidden, predictor_logits
            )
            source_losses = F.cross_entropy(
                source_logits.flatten(0, 1),
                source_targets.flatten(),
                reduction="none",
            ).view_as(source_targets)
            source_loss_sums = (
                source_losses * latent_mask.to(source_losses.dtype)
            ).sum(dim=1)

            batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
            last_latent_positions = prompt_width + latent_counts - 1
            last_hidden = hidden[batch_indices, last_latent_positions]
            last_base_logits = base_logits[batch_indices, last_latent_positions]
            end_logits = self.reasoner.boundary.append_logits(
                last_hidden, last_base_logits
            )
            end_targets = torch.full_like(
                latent_counts, self.reasoner.end_latent_id
            )
            end_losses = F.cross_entropy(
                end_logits, end_targets, reduction="none"
            )
            token_ce_loss_sums = source_loss_sums + end_losses
            token_ce_counts = latent_counts + 1
            token_ce_loss = (
                token_ce_loss_sums.sum() / token_ce_counts.sum().clamp_min(1)
            )
            result["loss"] = result["loss"] + token_ce_loss
            result.update(
                {
                    "token_ce_loss": token_ce_loss,
                    "token_ce_loss_sums": token_ce_loss_sums,
                    "token_ce_counts": token_ce_counts,
                }
            )
        if "cot_mask" in batch:
            cot_mask = batch["cot_mask"][:, 1:]
            cot_loss_sums = (
                suffix_token_losses * cot_mask.to(suffix_token_losses.dtype)
            ).sum(dim=1)
            cot_token_counts = cot_mask.sum(dim=1)
            cot_loss = masked_token_mean(suffix_token_losses, cot_mask)
            result["loss"] = result["loss"] + cot_loss
            result.update(
                {
                    "cot_loss": cot_loss,
                    "cot_loss_sums": cot_loss_sums,
                    "cot_token_counts": cot_token_counts,
                }
            )
        return result


def load_compression_reasoner(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    is_trainable: bool = True,
):
    reasoner, tokenizer = load_latent_reasoner(
        config,
        checkpoint_path=checkpoint_path,
        is_trainable=is_trainable,
    )
    latent = config["latent"]
    return (
        CompressionReasoner(
            reasoner,
            enable_token_ce=latent["enable_token_ce"],
            fixed_variance=latent.get("fixed_variance", 1.0e-18),
            gumbel_temperature=latent.get("gumbel_temperature", 1.0),
            gumbel_noise_scale=latent.get("gumbel_noise_scale", 1.0e-3),
        ),
        tokenizer,
    )
