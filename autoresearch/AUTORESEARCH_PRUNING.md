# autoresearch — pruning edition

An adaptation of the autoresearch loop for the
[Attention_Pruning_Sparsity_Restructure](../Attention_Pruning_Sparsity_Restructure.md)
plan: the LLM autonomously researches **pruning recipes** for HF transformer models
instead of training recipes. Same philosophy — fixed evaluation harness, one editable
file, tight keep/discard loop on a single scalar metric.

## What changed vs the original

| Original (training) | This (pruning) |
|---|---|
| `prepare.py` fixed data/eval | `autoresearch/harness.py` fixed model/data/eval — **read-only** |
| `train.py` is the edit surface | `autoresearch/experiment.py` is the edit surface (plus `sconce/` library code the recipe calls) |
| metric: `val_bpb` after 5-min train | metric: WikiText-2 `val_ppl` of the **pruned, saved, and reloaded** model |
| fixed budget: 5 min wall clock | fixed budget: `params_ratio ≤ 0.75` of dense + recipe wall-clock ≤ 15 min |
| `uv run train.py` | `python autoresearch/harness.py` |

Two deliberate design choices tie the loop to the restructure plan:

1. **The round-trip gate.** A run only scores if `save_pretrained → from_pretrained`
   succeeds and the *reloaded* model is what gets evaluated. This makes the G2
   config-sync gap (missing `head_dim`, `num_hidden_layers`, …) an immediate research
   obstacle rather than deferred cleanup, and guarantees every "keep" is shippable.
2. **Compression is the budget, quality is the score.** Every scored run must remove
   ≥ 25% of parameters; within that, minimize perplexity. This is the direct-slicing
   RFC objective with the NAS loop replaced by *you*.

## Setup

1. **Agree on a run tag** with the user (e.g. `jul27`). Branch
   `autoresearch/<tag>` must not exist yet.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current
   working branch (`experiment/apr14`).
3. **Read the in-scope files**:
   - `Attention_Pruning_Sparsity_Restructure.md` — the gap analysis; your idea backlog.
   - `autoresearch/harness.py` — fixed constants, eval, run protocol. **Do not modify.**
   - `autoresearch/experiment.py` — the recipe you edit.
   - `sconce/pruner.py` — the library the recipe builds on (editable).
4. **Verify data/model access**: first harness run downloads WikiText-2 and the model,
   then caches token ids + baseline ppl in `~/.cache/autoresearch-sconce/`.
   Smoke-test the plumbing anytime with `AR_MODEL=__tiny__ python autoresearch/harness.py`.
5. **Initialize `autoresearch/results.tsv`** with just the header row. (The repo root
   already has a tracked `results.tsv` from earlier work — this one is separate and
   stays untracked; `autoresearch/results.tsv` is gitignored by the loop convention.)
6. **Confirm and go.**

## Experimentation

Each run: `python autoresearch/harness.py > autoresearch/run.log 2>&1`

The harness loads the dense model, hands it plus calibration batches to
`experiment.run(...)`, then enforces the constraints and prints a summary block:

```
---
val_ppl:          14.532101
baseline_ppl:     13.90122
ppl_ratio:        1.0454
params_M:         371.22
params_ratio:     0.7488
prune_seconds:    41.3
peak_mem_gb:      4.2
roundtrip:        ok
status:           valid
```

Extract results: `grep "^val_ppl:\|^params_ratio:\|^peak_mem_gb:\|^status:\|^roundtrip:" autoresearch/run.log`

**What you CAN do**
- Rewrite `autoresearch/experiment.py` freely: importance metrics (weight L2,
  activation norms via calibration hooks, cosine block redundancy), what to cut
  (SwiGLU FFN width, GQA Q-heads, whole layers, combinations), per-layer ratio
  schedules, slicing order, light recovery finetune / distillation on the calibration
  batches — anything that fits the time budget.
- Edit `sconce/` library code when the recipe needs a capability or fix there
  (activation hooks, ConfigUpdater, depth pruning, SparsityPacker, …). That is the
  restructure work happening *inside* the loop; commit library changes together with
  the recipe that proves them.

**What you CANNOT do**
- Modify `autoresearch/harness.py`. Eval, constants, the round-trip gate, and the
  compression target are ground truth.
- Install new packages. Only what's already importable in this environment.
- Game the metric (e.g., special-casing WikiText-2 content in the recipe).

**The goal:** lowest `val_ppl` with `status: valid`. The first run is always the
baseline (identity recipe, as checked in) — it records dense ppl and confirms the
harness works; it reports `params_ratio 1.0` and is exempt from the compression gate.

**VRAM/RAM** is a soft constraint (reported as `peak_mem_gb`); don't blow it up.

**Simplicity criterion** (unchanged from the original): all else equal, simpler wins.
A tiny ppl gain that adds hacky complexity to `sconce/` is not worth it; equal results
with less code is a keep.

## Logging results

Append to `autoresearch/results.tsv` (tab-separated, untracked). Columns:

```
commit	val_ppl	params_ratio	mem_gb	status	description
```

- `commit`: short hash of the recipe commit
- `val_ppl`: reloaded-model perplexity (0.000000 for crash/invalid)
- `params_ratio`: pruned/dense params, 4 decimals (1.0000 for baseline)
- `mem_gb`: peak_mem_gb rounded to .1f
- `status`: `keep` | `discard` | `crash` | `invalid` (constraint or round-trip failure)
- `description`: short; no tabs

Example:

```
commit	val_ppl	params_ratio	mem_gb	status	description
a1b2c3d	13.901220	1.0000	4.1	keep	baseline (identity recipe)
b2c3d4e	0.000000	0.7500	4.2	invalid	head prune 25% — roundtrip fail (head_dim missing)
c3d4e5f	15.220000	0.7490	4.2	keep	uniform FFN 40% + config head_dim fix in sconce
d4e5f6g	14.530000	0.7488	4.3	keep	activation-based FFN scoring beats weight-L2
```

## The experiment loop

On branch `autoresearch/<tag>`, LOOP FOREVER:

1. Check git state (branch/commit).
2. Edit `autoresearch/experiment.py` (and `sconce/` if needed) with one idea.
3. `git commit` (never commit `autoresearch/results.tsv` or `run.log`).
4. `python autoresearch/harness.py > autoresearch/run.log 2>&1`
5. `grep "^val_ppl:\|^status:" autoresearch/run.log` — empty grep ⇒ crash ⇒
   `tail -n 50 autoresearch/run.log`, fix if trivial, else log `crash` and move on.
6. Log the row in `autoresearch/results.tsv`.
7. `status: valid` and val_ppl improved (lower than best-so-far valid run) ⇒ advance
   the branch (keep the commit). Otherwise ⇒ `git reset --hard` back.
8. `status: invalid` runs never advance the branch, but their *reason* is signal —
   a round-trip failure tells you exactly which ConfigUpdater gap to fix next.

**Timeout:** a run should finish in ≲ 20 min on this machine (eval is ~2× ppl passes
+ recipe time). If it exceeds 30 min, kill it, treat as failure.

**Idea backlog** (when stuck, re-read the restructure plan; roughly in order):

1. Baseline (identity) — done first, always.
2. Uniform SwiGLU FFN width prune (weight-L2) at the ratio that hits `params_ratio ≈ 0.75`.
3. GQA Q-head prune — will fail round-trip until `config.head_dim` is written (G2);
   fix in `sconce`, re-run.
4. Depth prune (uniform-spaced, then cosine-redundancy ranked; renumber `layer_idx`, G6).
5. Activation-based FFN neuron scoring from `calib_batches` (hooks on `down_proj`
   input) vs weight-L2 (G5) — the RFC's core bet; measure the delta honestly.
6. Activation-based head scoring (hooked `o_proj` input, per-head L2).
7. Depth + width combos; skew ratios by per-layer sensitivity.
8. Non-uniform per-layer ratios (needs the heterogeneous-config strategy from the plan).
9. Short recovery finetune on calib batches inside the budget; then logit distillation
   from the dense teacher (G8).
10. GMP-mask → SparsityPacker re-pack (G7) as an alternative route to the same
    params_ratio.

**NEVER STOP** (unchanged): once the loop begins, do not pause to ask whether to
continue. If out of ideas, re-read `Attention_Pruning_Sparsity_Restructure.md`,
combine near-misses, or go more radical. The loop runs until manually interrupted.
