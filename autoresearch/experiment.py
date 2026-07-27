"""
autoresearch/experiment.py — THE file the experiment loop edits.

Contract (enforced by harness.py, which is read-only):

    def run(model, tokenizer, calib_batches, device) -> model

  * model:          HF causal LM on CPU, fp32, eval mode.
  * calib_batches:  list of {"input_ids": LongTensor[B, SEQ_LEN]} on CPU
                    (WikiText-2 train) for activation-based importance scoring.
  * device:         preferred compute device (cuda / mps / cpu).
  * return:         the (pruned) model. It will be judged on:
                      - params_ratio <= 0.75  (vs dense baseline)
                      - save_pretrained -> from_pretrained round-trip
                      - WikiText-2 val_ppl of the RELOADED model (lower = better)

Everything about the pruning recipe is fair game: importance metric (weight
magnitude, activation norms, cosine block redundancy, ...), what to cut
(FFN width, Q heads, depth, combinations), ratios per component, ordering,
light recovery finetuning on calib_batches (inside the time budget), etc.
Edits to the sconce/ library are allowed when the recipe needs a capability
or bugfix there — that's the point. harness.py is off-limits.

The identity recipe below is the baseline run.
"""


def run(model, tokenizer, calib_batches, device):
    return model
