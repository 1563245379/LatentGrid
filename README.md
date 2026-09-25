# LatentGrid

Code for **Revisiting Latent Chain-of-Thought: Disentangling Representation and Supervision**.

This repository implements LatentGrid and Adaptive Curriculum Thinking (ACT), with experiments on answer accuracy, rationale recovery, representation alignment, and training gradients.

## Setup

Run commands from the repository root:

```bash
pip install -r requirements.txt
```

Set `base_model` and `checkpoint.cot_init_checkpoint` to the matching backbone and CoT weights.

## Training and evaluation

Choose a configuration from the table below and train on eight GPUs:

```bash
accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  -m cli train \
  --config configs/answer_unconstrained.yaml

python -m cli eval --config configs/answer_unconstrained.yaml
```

| Experiment | Configuration in `configs/` |
| --- | --- |
| CoT | `cot_reference.yaml` |
| No-CoT | `no_cot.yaml` |
| Answer | `answer_unconstrained.yaml` |
| Curriculum | `curriculum_unconstrained.yaml` |
| Reconstruction | `reconstruction_unconstrained.yaml` |
| Compression | `compression_unconstrained.yaml` |
| ACT-Rec | `curriculum_token_reconstruction_unconstrained.yaml` |
| ACT-Comp | `curriculum_compression_unconstrained.yaml` with Token CE enabled |

Replace `_unconstrained` with `_constrained` to use vocabulary-constrained representations. For ACT-Comp, copy its configuration, set `latent.enable_token_ce: true`, and choose a separate `output_dir`.

Set `checkpoint.eval_checkpoint` for evaluation or `checkpoint.resume_from_checkpoint` to resume training. Results are saved under `output_dir`, with predictions and metrics in `eval/`.

## Rationale recovery probe

Set `checkpoint.eval_checkpoint` in the selected paradigm configuration, then run:

```bash
accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  -m analysis.latent_cot_probe \
  --paradigm-config configs/reconstruction_unconstrained.yaml \
  --decoder-config configs/latent_cot_probe.yaml \
  --output-dir outputs/analysis/probe_reconstruction_unconstrained
```

This trains a decoder on frozen latent trajectories and reports NLL, BERTScore, and answer consistency.

## t-SNE and MMD

Set `checkpoint.eval_checkpoint` in both representation configurations for each method:

```bash
for method in curriculum reconstruction compression; do
  python -m analysis.embedding_shift_tsne \
    --unconstrained-config "configs/${method}_unconstrained.yaml" \
    --constrained-config "configs/${method}_constrained.yaml" \
    --output-dir "outputs/analysis/embedding_shift_${method}"
done
```

Defaults use 128 questions, 1,000 t-SNE points per group, and seed 42. Outputs include the t-SNE figure and question-level MMD² with 95% bootstrap confidence intervals.

## Latent-position gradients

Train paired Answer and Curriculum models and trace the first five optimizer updates at each curriculum stage boundary:

```bash
accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  -m analysis.latent_position_loss \
  --answer-config configs/answer_unconstrained.yaml \
  --curriculum-config configs/curriculum_unconstrained.yaml \
  --trace-optimizer-steps 5 \
  --seed 42 \
  --output-dir outputs/analysis/latent_position_loss
```

Use a fresh output directory for each t-SNE/MMD or gradient-analysis run.
