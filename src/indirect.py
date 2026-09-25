import torch

from src.data import render_prompt
from src.latent_model import END_LATENT_TOKEN, START_LATENT_TOKEN


def parse_cot_steps(cot):
    return tuple(line.strip() for line in cot.splitlines() if line.strip())


def build_indirect_plan(record, method, epoch, num_latent_tokens, stage_epochs=5, latent_tokens_per_stage=2):
    if method == "answer":
        latent_count, remaining = num_latent_tokens, ()
    else:
        final_stage = num_latent_tokens // latent_tokens_per_stage
        stage = min((epoch - 1) // stage_epochs + 1, final_stage)
        latent_count = min(stage * latent_tokens_per_stage, num_latent_tokens)
        steps = parse_cot_steps(record["cot"])
        remaining = () if stage == final_stage else steps[stage:]
    return {
        "example_id": record["example_id"],
        "question": record["question"],
        "answer": record["answer"],
        "remaining_cot": "\n".join(remaining),
        "latent_count": latent_count,
    }


class IndirectCollator:
    def __init__(
        self,
        tokenizer,
        prompt_template,
        method,
        epoch,
        latent_config,
        curriculum_config=None,
    ):
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.method = method
        self.epoch = epoch
        self.latent_config = latent_config
        self.curriculum_config = curriculum_config or {}

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _pad(sequences, pad_id, left=False):
        width = max(len(sequence) for sequence in sequences)
        input_ids = []
        attention_mask = []
        for sequence in sequences:
            padding = width - len(sequence)
            pads = [pad_id] * padding
            zeros = [0] * padding
            ones = [1] * len(sequence)
            input_ids.append(pads + sequence if left else sequence + pads)
            attention_mask.append(zeros + ones if left else ones + zeros)
        return torch.tensor(input_ids), torch.tensor(attention_mask)

    def __call__(self, records):
        plans = [
            build_indirect_plan(
                record,
                self.method,
                self.epoch,
                self.latent_config["num_latent_tokens"],
                stage_epochs=self.curriculum_config.get("stage_epochs", 5),
                latent_tokens_per_stage=self.curriculum_config.get(
                    "latent_tokens_per_stage", 2
                ),
            )
            for record in records
        ]
        start_id = self.tokenizer.convert_tokens_to_ids(START_LATENT_TOKEN)
        end_id = self.tokenizer.convert_tokens_to_ids(END_LATENT_TOKEN)
        prompt_ids = [
            self._encode(render_prompt(self.prompt_template, plan["question"]))
            + [start_id]
            for plan in plans
        ]

        suffix_ids = []
        cot_masks = []
        answer_masks = []
        marker_ids = self._encode("###Answer:")
        for plan in plans:
            cot_ids = self._encode(plan["remaining_cot"])
            answer_ids = self._encode(plan["answer"]) + [self.tokenizer.eos_token_id]
            suffix_ids.append([end_id] + cot_ids + marker_ids + answer_ids)
            cot_masks.append(
                [False] + [True] * len(cot_ids) + [False] * (len(marker_ids) + len(answer_ids))
            )
            answer_masks.append(
                [False] * (1 + len(cot_ids) + len(marker_ids))
                + [True] * len(answer_ids)
            )

        prompt_input_ids, prompt_attention_mask = self._pad(
            prompt_ids, self.tokenizer.pad_token_id, left=True
        )
        suffix_input_ids, suffix_attention_mask = self._pad(
            suffix_ids, self.tokenizer.pad_token_id
        )
        suffix_width = suffix_input_ids.shape[1]
        cot_mask = torch.tensor(
            [mask + [False] * (suffix_width - len(mask)) for mask in cot_masks],
            dtype=torch.bool,
        )
        answer_mask = torch.tensor(
            [mask + [False] * (suffix_width - len(mask)) for mask in answer_masks],
            dtype=torch.bool,
        )
        return {
            "example_ids": torch.tensor(
                [plan["example_id"] for plan in plans], dtype=torch.int64
            ),
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "suffix_input_ids": suffix_input_ids,
            "suffix_attention_mask": suffix_attention_mask,
            "cot_mask": cot_mask,
            "answer_mask": answer_mask,
            "latent_count": plans[0]["latent_count"],
        }
