from pathlib import Path
from typing import Any

import torch
from peft import PeftConfig, PeftModel
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


START_LATENT_TOKEN = "<start_latent>"
END_LATENT_TOKEN = "<end_latent>"


class BoundaryTokenParameters(nn.Module):
    """Trainable input and output rows for the two latent boundary tokens."""

    def __init__(self, input_rows: Tensor, output_rows: Tensor) -> None:
        super().__init__()
        self.input_rows = nn.Parameter(input_rows.clone())
        self.output_rows = nn.Parameter(output_rows.clone())

    @classmethod
    def from_model(
        cls, input_embedding: nn.Embedding, output_head: nn.Linear
    ) -> "BoundaryTokenParameters":
        return cls(
            input_embedding.weight.detach().mean(dim=0, keepdim=True).repeat(2, 1),
            output_head.weight.detach().mean(dim=0, keepdim=True).repeat(2, 1),
        )

    def embed(
        self,
        input_ids: Tensor,
        base_embedding: nn.Embedding,
        original_vocab_size: int,
    ) -> Tensor:
        base_ids = input_ids.clamp(max=original_vocab_size - 1)
        embeddings = base_embedding(base_ids)
        start_mask = input_ids == original_vocab_size
        end_mask = input_ids == original_vocab_size + 1
        embeddings = torch.where(start_mask.unsqueeze(-1), self.input_rows[0], embeddings)
        return torch.where(end_mask.unsqueeze(-1), self.input_rows[1], embeddings)

    def append_logits(self, hidden_states: Tensor, base_logits: Tensor) -> Tensor:
        special_logits = hidden_states @ self.output_rows.t()
        return torch.cat([base_logits, special_logits], dim=-1)


class UnconstrainedLatentHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.layers(hidden)


def constrained_latent(
    base_logits: Tensor,
    frozen_embeddings: Tensor,
    top_k: int,
    temperature: float,
) -> Tensor:
    top_logits, top_indices = torch.topk(base_logits, top_k, dim=-1)
    weights = torch.softmax(top_logits / temperature, dim=-1)
    top_embeddings = frozen_embeddings[top_indices]
    return (weights.unsqueeze(-1) * top_embeddings).sum(dim=-2)


def position_ids_from_mask(attention_mask: Tensor) -> Tensor:
    return (attention_mask.long().cumsum(dim=-1) - 1).clamp_min(0)


def masked_token_mean(losses: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1)


def shifted_suffix_logits(
    final_latent_logits: Tensor, suffix_logits: Tensor
) -> Tensor:
    return torch.cat(
        [final_latent_logits.unsqueeze(1), suffix_logits[:, :-1]], dim=1
    )


class LatentReasoner(nn.Module):
    def __init__(
        self,
        llm,
        original_vocab_size,
        start_latent_id,
        end_latent_id,
        representation_type,
        top_k=10,
        representation_temperature=1.0,
    ):
        super().__init__()
        canonical_ids = (original_vocab_size, original_vocab_size + 1)
        if (start_latent_id, end_latent_id) != canonical_ids:
            raise ValueError(
                "boundary tokens must use canonical appended IDs "
                f"{canonical_ids}, got {(start_latent_id, end_latent_id)}"
            )
        self.llm = llm
        self.original_vocab_size = original_vocab_size
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        config = getattr(llm, "config", None)
        if config is None:
            config = llm.model.config
        configured_eos_ids = config.eos_token_id
        self.eos_token_ids = (
            (configured_eos_ids,)
            if isinstance(configured_eos_ids, int)
            else tuple(configured_eos_ids)
        )
        self.eos_token_id = self.eos_token_ids[0]
        self.representation_type = representation_type
        self.top_k = top_k
        self.representation_temperature = representation_temperature

        input_embedding = llm.get_input_embeddings()
        self.boundary = BoundaryTokenParameters.from_model(
            input_embedding, llm.get_output_embeddings()
        )
        self.latent_head = None
        if representation_type == "unconstrained":
            self.latent_head = UnconstrainedLatentHead(
                input_embedding.embedding_dim
            ).to(device=input_embedding.weight.device, dtype=input_embedding.weight.dtype)

    def save_trainable(self, checkpoint_dir, tokenizer) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.llm.save_pretrained(checkpoint_dir / "adapter")
        tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
        torch.save(
            {
                "boundary": self.boundary.state_dict(),
                "latent_head": (
                    self.latent_head.state_dict()
                    if self.latent_head is not None
                    else None
                ),
                "original_vocab_size": self.original_vocab_size,
                "start_latent_id": self.start_latent_id,
                "end_latent_id": self.end_latent_id,
            },
            checkpoint_dir / "latent_state.pt",
        )

    def load_extra_state(self, checkpoint_dir) -> None:
        state = torch.load(
            Path(checkpoint_dir) / "latent_state.pt", map_location="cpu"
        )
        self.boundary.load_state_dict(state["boundary"])
        if self.latent_head is not None:
            self.latent_head.load_state_dict(state["latent_head"])

    def embed(self, input_ids: Tensor) -> Tensor:
        return self.boundary.embed(
            input_ids, self.llm.get_input_embeddings(), self.original_vocab_size
        )

    def next_latent(self, hidden: Tensor, base_logits: Tensor) -> Tensor:
        if self.representation_type == "unconstrained":
            return self.latent_head(hidden)
        return constrained_latent(
            base_logits[:, : self.original_vocab_size],
            self.llm.get_input_embeddings().weight[: self.original_vocab_size],
            self.top_k,
            self.representation_temperature,
        )

    def _generate_answer_from_cache(
        self,
        outputs,
        attention_mask: Tensor,
        answer_prefix_ids: Tensor,
        max_new_tokens: int,
    ) -> Tensor:
        batch_size = attention_mask.shape[0]
        prefix_ids = answer_prefix_ids.to(attention_mask.device).reshape(1, -1)
        suffix_ids = torch.cat(
            [
                torch.full(
                    (1, 1),
                    self.end_latent_id,
                    dtype=prefix_ids.dtype,
                    device=prefix_ids.device,
                ),
                prefix_ids,
            ],
            dim=1,
        ).expand(batch_size, -1)
        suffix_mask = torch.ones(
            suffix_ids.shape,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([attention_mask, suffix_mask], dim=1)
        outputs = self.llm(
            inputs_embeds=self.embed(suffix_ids),
            attention_mask=attention_mask,
            position_ids=position_ids_from_mask(attention_mask)[
                :, -suffix_ids.shape[1] :
            ],
            past_key_values=outputs.past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        generated = []
        finished = torch.zeros(
            batch_size, dtype=torch.bool, device=attention_mask.device
        )
        eos_ids = suffix_ids.new_tensor(self.eos_token_ids)
        for _ in range(max_new_tokens):
            logits = self.boundary.append_logits(
                outputs.hidden_states[-1][:, -1], outputs.logits[:, -1]
            ).clone()
            logits[:, self.start_latent_id] = torch.finfo(logits.dtype).min
            logits[:, self.end_latent_id] = torch.finfo(logits.dtype).min
            next_ids = logits.argmax(dim=-1)
            next_ids = torch.where(
                finished,
                torch.full_like(next_ids, self.eos_token_id),
                next_ids,
            )
            generated.append(next_ids)
            finished |= (next_ids.unsqueeze(-1) == eos_ids).any(dim=-1)
            if finished.all():
                break
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1
            )
            outputs = self.llm(
                inputs_embeds=self.embed(next_ids.unsqueeze(1)),
                attention_mask=attention_mask,
                position_ids=position_ids_from_mask(attention_mask)[:, -1:],
                past_key_values=outputs.past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )
        if generated:
            return torch.stack(generated, dim=1)
        return suffix_ids.new_empty((batch_size, 0))

    @torch.inference_mode()
    def generate_answer_ids(
        self,
        prompt_input_ids: Tensor,
        prompt_attention_mask: Tensor,
        latent_count: int,
        answer_prefix_ids: Tensor,
        max_new_tokens: int = 64,
    ) -> tuple[Tensor, Tensor]:
        outputs = self.llm(
            inputs_embeds=self.embed(prompt_input_ids),
            attention_mask=prompt_attention_mask,
            position_ids=position_ids_from_mask(prompt_attention_mask),
            output_hidden_states=True,
            use_cache=True,
        )

        attention_mask = prompt_attention_mask
        for _ in range(latent_count):
            latent = self.next_latent(
                outputs.hidden_states[-1][:, -1, :],
                outputs.logits[:, -1, :],
            )
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1
            )
            outputs = self.llm(
                inputs_embeds=latent.unsqueeze(1),
                attention_mask=attention_mask,
                position_ids=position_ids_from_mask(attention_mask)[:, -1:],
                past_key_values=outputs.past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )

        generated_ids = self._generate_answer_from_cache(
            outputs,
            attention_mask,
            answer_prefix_ids,
            max_new_tokens,
        )
        latent_lengths = torch.full(
            (prompt_input_ids.shape[0],),
            latent_count,
            dtype=torch.long,
            device=prompt_input_ids.device,
        )
        return generated_ids, latent_lengths

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
        outputs = self.llm(
            inputs_embeds=self.embed(prompt_input_ids),
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
        for _ in range(max_latent_tokens):
            latent = self.next_latent(
                outputs.hidden_states[-1][:, -1], outputs.logits[:, -1]
            )
            active = ~finished
            attention_mask = torch.cat(
                [
                    attention_mask,
                    active.to(attention_mask.dtype).unsqueeze(1),
                ],
                dim=1,
            )
            latent_lengths += active.long()
            outputs = self.llm(
                inputs_embeds=latent.unsqueeze(1),
                attention_mask=attention_mask,
                position_ids=position_ids_from_mask(attention_mask)[:, -1:],
                past_key_values=outputs.past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )
            end_logits = self.boundary.append_logits(
                outputs.hidden_states[-1][:, -1], outputs.logits[:, -1]
            )
            finished |= active & end_logits.argmax(dim=-1).eq(
                self.end_latent_id
            )
            if finished.all():
                break

        generated_ids = self._generate_answer_from_cache(
            outputs,
            attention_mask,
            answer_prefix_ids,
            max_new_tokens,
        )
        return generated_ids, latent_lengths, ~finished

    def forward(self, batch, retain_latent_grad=False):
        prompt_attention_mask = batch["prompt_attention_mask"]
        outputs = self.llm(
            inputs_embeds=self.embed(batch["prompt_input_ids"]),
            attention_mask=prompt_attention_mask,
            position_ids=position_ids_from_mask(prompt_attention_mask),
            output_hidden_states=True,
            use_cache=True,
        )

        attention_mask = prompt_attention_mask
        latents = []
        for _ in range(batch["latent_count"]):
            hidden = outputs.hidden_states[-1][:, -1, :]
            latent = self.next_latent(hidden, outputs.logits[:, -1, :])
            if retain_latent_grad:
                latent.retain_grad()
            latents.append(latent)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1
            )
            outputs = self.llm(
                inputs_embeds=latent.unsqueeze(1),
                attention_mask=attention_mask,
                position_ids=position_ids_from_mask(attention_mask)[:, -1:],
                past_key_values=outputs.past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )

        final_hidden = outputs.hidden_states[-1][:, -1, :]
        final_latent_logits = self.boundary.append_logits(
            final_hidden, outputs.logits[:, -1, :]
        )
        suffix_attention_mask = batch["suffix_attention_mask"]
        full_attention_mask = torch.cat(
            [attention_mask, suffix_attention_mask], dim=1
        )
        suffix_outputs = self.llm(
            inputs_embeds=self.embed(batch["suffix_input_ids"]),
            attention_mask=full_attention_mask,
            position_ids=position_ids_from_mask(full_attention_mask)[
                :, -suffix_attention_mask.shape[1] :
            ],
            past_key_values=outputs.past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        suffix_logits = self.boundary.append_logits(
            suffix_outputs.hidden_states[-1], suffix_outputs.logits
        )
        predictor_logits = shifted_suffix_logits(
            final_latent_logits, suffix_logits
        )
        token_losses = F.cross_entropy(
            predictor_logits.flatten(0, 1),
            batch["suffix_input_ids"].flatten(),
            reduction="none",
        ).view_as(batch["suffix_input_ids"])

        cot_mask = batch["cot_mask"]
        answer_mask = batch["answer_mask"]
        cot_loss_sums = (
            token_losses * cot_mask.to(token_losses.dtype)
        ).sum(dim=1)
        answer_loss_sums = (
            token_losses * answer_mask.to(token_losses.dtype)
        ).sum(dim=1)
        cot_token_counts = cot_mask.sum(dim=1)
        answer_token_counts = answer_mask.sum(dim=1)
        cot_loss_sum = cot_loss_sums.sum()
        answer_loss_sum = answer_loss_sums.sum()
        cot_token_count = cot_token_counts.sum()
        answer_token_count = answer_token_counts.sum()
        cot_loss = masked_token_mean(token_losses, cot_mask)
        answer_loss = masked_token_mean(token_losses, answer_mask)
        return {
            "loss": cot_loss + answer_loss,
            "answer_loss": answer_loss,
            "cot_loss": cot_loss,
            "answer_loss_sum": answer_loss_sum,
            "answer_token_count": answer_token_count,
            "cot_loss_sum": cot_loss_sum,
            "cot_token_count": cot_token_count,
            "answer_loss_sums": answer_loss_sums,
            "answer_token_counts": answer_token_counts,
            "cot_loss_sums": cot_loss_sums,
            "cot_token_counts": cot_token_counts,
            "latents": latents,
        }


def load_cot_backbone(
    config: dict[str, Any], adapter_path: str | None = None, is_trainable: bool = True
) -> tuple[nn.Module, Any, int]:
    """Load the frozen base vocabulary with the trainable CoT adapter attached."""
    adapter_path = adapter_path or config["checkpoint"]["cot_init_checkpoint"]
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
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
        config["base_model"], **load_kwargs
    )
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    input_embedding = base_model.get_input_embeddings()
    output_head = base_model.get_output_embeddings()
    original_vocab_size = input_embedding.weight.shape[0]

    tokenizer.add_tokens(
        [
            f"<|latent_vocab_pad_{token_id}|>"
            for token_id in range(len(tokenizer), original_vocab_size)
        ],
        special_tokens=True,
    )
    tokenizer.add_tokens([START_LATENT_TOKEN, END_LATENT_TOKEN])
    model = PeftModel.from_pretrained(
        base_model, adapter_path, is_trainable=is_trainable
    )
    return model, tokenizer, original_vocab_size


def load_latent_reasoner(
    config: dict[str, Any], checkpoint_path=None, is_trainable: bool = True
) -> tuple[LatentReasoner, Any]:
    latent_config = config.get("latent", {})
    if checkpoint_path is None:
        llm, tokenizer, original_vocab_size = load_cot_backbone(
            config, is_trainable=is_trainable
        )
        reasoner = LatentReasoner(
            llm,
            original_vocab_size,
            tokenizer.convert_tokens_to_ids(START_LATENT_TOKEN),
            tokenizer.convert_tokens_to_ids(END_LATENT_TOKEN),
            config["latent_representation_type"],
            top_k=latent_config.get("top_k", 10),
            representation_temperature=latent_config.get(
                "representation_temperature", 1.0
            ),
        )
        return reasoner, tokenizer

    checkpoint_path = Path(checkpoint_path)
    latent_state = torch.load(
        checkpoint_path / "latent_state.pt", map_location="cpu"
    )
    adapter_path = checkpoint_path / "adapter"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path / "tokenizer")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    adapter_config = PeftConfig.from_pretrained(adapter_path)
    with torch.random.fork_rng():
        load_kwargs = {
            "dtype": torch.bfloat16
            if config["training"]["bf16"]
            else torch.float32
        }
        if not is_trainable:
            load_kwargs["device_map"] = "auto"
        base_model = AutoModelForCausalLM.from_pretrained(
            adapter_config.base_model_name_or_path,
            **load_kwargs,
        )
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)
        llm = PeftModel.from_pretrained(
            base_model, adapter_path, is_trainable=is_trainable
        )
        reasoner = LatentReasoner(
            llm,
            latent_state["original_vocab_size"],
            latent_state["start_latent_id"],
            latent_state["end_latent_id"],
            config["latent_representation_type"],
            top_k=latent_config.get("top_k", 10),
            representation_temperature=latent_config.get(
                "representation_temperature", 1.0
            ),
        )
        reasoner.load_extra_state(checkpoint_path)
    return reasoner, tokenizer
