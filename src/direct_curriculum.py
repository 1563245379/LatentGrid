from src.indirect import parse_cot_steps


MAX_DIRECT_CURRICULUM_STAGE = 6


def curriculum_stage(epoch: int, stage_epochs: int) -> int:
    return min((epoch - 1) // stage_epochs + 1, MAX_DIRECT_CURRICULUM_STAGE)


def validate_tokens_per_latent(tokens_per_latent):
    if type(tokens_per_latent) is not int or tokens_per_latent < 1:
        raise ValueError("tokens_per_latent must be a positive integer")


def group_cot_with_steps(tokenizer, cot: str, tokens_per_latent: int = 2):
    validate_tokens_per_latent(tokens_per_latent)
    encoded = tokenizer(cot, add_special_tokens=False, return_offsets_mapping=True)
    cot_ids = tuple(encoded["input_ids"])
    line_spans = []
    offset = 0
    for line in cot.splitlines(keepends=True):
        line_end = offset + len(line)
        if line.strip():
            line_spans.append((offset, line_end))
        offset = line_end
    assignments = []
    for token_start, token_end in encoded["offset_mapping"]:
        overlaps = [
            max(0, min(token_end, line_end) - max(token_start, line_start))
            for line_start, line_end in line_spans
        ]
        assignments.append(
            max(range(len(overlaps)), key=lambda index: (overlaps[index], index))
        )
    groups = tuple(
        tuple(cot_ids[index : index + tokens_per_latent])
        for index in range(0, len(cot_ids), tokens_per_latent)
    )
    step_groups = tuple(
        max(
            range(max(assignments) + 1),
            key=lambda step: (assignments[index : index + tokens_per_latent].count(step), step),
        )
        for index in range(0, len(cot_ids), tokens_per_latent)
    )
    return groups, step_groups


def split_curriculum_groups(groups, step_groups, stage):
    selected = tuple(
        index
        for index, step_group in enumerate(step_groups)
        if stage == MAX_DIRECT_CURRICULUM_STAGE or step_group < stage
    )
    selected_set = set(selected)
    remaining = tuple(
        token_id
        for index, group in enumerate(groups)
        if index not in selected_set
        for token_id in group
    )
    return selected, remaining


def build_reconstruction_curriculum_plan(
    record, epoch, stage_epochs, latent_tokens_per_stage
):
    stage = curriculum_stage(epoch, stage_epochs)
    consolidation = stage == MAX_DIRECT_CURRICULUM_STAGE
    steps = parse_cot_steps(record["cot"])
    targets = []
    for group_index in range(stage):
        if group_index < len(steps):
            target = (
                "\n".join(steps[group_index:])
                if consolidation and group_index == stage - 1
                else steps[group_index]
            )
        else:
            target = record["answer"]
        targets.append(target)
    return {
        "stage": stage,
        "latent_count": stage * latent_tokens_per_stage,
        "remaining_cot": "" if consolidation else "\n".join(steps[stage:]),
        "reconstruction_targets": tuple(targets),
        "consolidation": consolidation,
    }
