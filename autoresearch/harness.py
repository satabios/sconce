"""
autoresearch/harness.py — READ-ONLY evaluation harness. Do not modify.

Analog of autoresearch's prepare.py: fixed constants, data, evaluation, and the
run protocol. The experiment loop only ever edits autoresearch/experiment.py
(and, where needed, the sconce/ library it calls).

Run:  python autoresearch/harness.py > run.log 2>&1

Protocol per run:
  1. Load base model + tokenizer (AR_MODEL, default Qwen/Qwen2-0.5B;
     "__tiny__" builds a small random Qwen2 for plumbing/smoke tests).
  2. Compute/load cached baseline: val_ppl + param count of the dense model.
  3. Call experiment.run(model, tokenizer, calib_batches, device) -> model.
  4. Enforce constraints:  params_ratio <= PARAMS_RATIO_TARGET,
                           prune_seconds <= PRUNE_BUDGET_S.
  5. HARD GATE: save_pretrained -> from_pretrained round-trip must succeed.
  6. Score: val_ppl of the RELOADED model on WikiText-2 test (fixed window).
"""
import importlib
import json
import math
import os
import resource
import sys
import tempfile
import time

import torch

# ---------------------------------------------------------------------------
# Fixed constants — the contract of the experiment. Never change mid-run-tag.
# ---------------------------------------------------------------------------
MODEL_NAME          = os.environ.get("AR_MODEL", "Qwen/Qwen2-0.5B")
SEQ_LEN             = 512
EVAL_TOKENS         = 32_768      # WikiText-2 test tokens scored per eval
CALIB_TOKENS        = 65_536      # WikiText-2 train tokens for calibration
CALIB_BATCH_SEQS    = 4           # sequences per calibration batch
PARAMS_RATIO_TARGET = 0.75        # pruned/dense total params must be <= this
PRUNE_BUDGET_S      = 900         # wall-clock cap for experiment.run()
CACHE_DIR           = os.path.expanduser("~/.cache/autoresearch-sconce")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model / data
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if MODEL_NAME == "__tiny__":
        from transformers import Qwen2Config, Qwen2ForCausalLM
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        cfg = Qwen2Config(
            hidden_size=128, num_hidden_layers=4, num_attention_heads=8,
            num_key_value_heads=2, intermediate_size=256,
            vocab_size=tok.vocab_size + len(tok.get_added_vocab()),
            max_position_embeddings=1024, tie_word_embeddings=True,
        )
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(cfg)
    else:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, tok


def _wikitext_ids(tokenizer, split: str, max_tokens: int) -> torch.Tensor:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"wt2_{split}_{max_tokens}_{tokenizer.name_or_path.replace('/', '_')}.pt")
    if os.path.exists(cache):
        return torch.load(cache, weights_only=True)
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0][:max_tokens]
    torch.save(ids, cache)
    return ids


def get_calibration_batches(tokenizer):
    """List of {"input_ids": [CALIB_BATCH_SEQS, SEQ_LEN]} dicts on CPU."""
    ids = _wikitext_ids(tokenizer, "train", CALIB_TOKENS)
    seqs = [ids[i:i + SEQ_LEN] for i in range(0, len(ids) - SEQ_LEN, SEQ_LEN)]
    batches = []
    for i in range(0, len(seqs) - CALIB_BATCH_SEQS + 1, CALIB_BATCH_SEQS):
        batches.append({"input_ids": torch.stack(seqs[i:i + CALIB_BATCH_SEQS])})
    return batches


# ---------------------------------------------------------------------------
# Evaluation — ground truth metric. Fixed.
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ppl(model, tokenizer, device) -> float:
    ids = _wikitext_ids(tokenizer, "test", EVAL_TOKENS)
    model.eval().to(device)
    nlls = []
    for begin in range(0, ids.size(0) - 1, SEQ_LEN):
        end = min(begin + SEQ_LEN, ids.size(0) - 1)
        chunk = ids[begin:end + 1].unsqueeze(0).to(device)
        out = model(chunk, labels=chunk.clone())
        nlls.append(out.loss.float().cpu())
    return math.exp(torch.stack(nlls).mean().item())


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def peak_mem_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024**3 if sys.platform == "darwin" else rss / 1024**2 / 1024


# ---------------------------------------------------------------------------
# Round-trip gate
# ---------------------------------------------------------------------------

def roundtrip(model, tokenizer):
    """save_pretrained -> from_pretrained. Returns (reloaded_model, error_str)."""
    from transformers import AutoModelForCausalLM
    with tempfile.TemporaryDirectory() as d:
        try:
            model.cpu().save_pretrained(d)
            tokenizer.save_pretrained(d)
            reloaded = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32)
            return reloaded, None
        except Exception as exc:  # noqa: BLE001 — report any reload failure
            return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Baseline cache
# ---------------------------------------------------------------------------

def get_baseline(model, tokenizer, device) -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = MODEL_NAME.replace("/", "_")
    cache = os.path.join(CACHE_DIR, f"baseline_{key}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    print("[harness] computing baseline (first run on this machine)...", flush=True)
    base = {"val_ppl": evaluate_ppl(model, tokenizer, device), "params": count_params(model)}
    with open(cache, "w") as f:
        json.dump(base, f)
    return base


# ---------------------------------------------------------------------------
# Main protocol
# ---------------------------------------------------------------------------

def main() -> None:
    device = get_device()
    print(f"[harness] model={MODEL_NAME} device={device.type}", flush=True)

    model, tokenizer = load_model_and_tokenizer()
    baseline = get_baseline(model, tokenizer, device)
    calib = get_calibration_batches(tokenizer)
    model.cpu()

    import experiment  # autoresearch/experiment.py — the file the loop edits
    importlib.reload(experiment)

    t0 = time.time()
    model = experiment.run(model, tokenizer, calib, device)
    prune_seconds = time.time() - t0

    params = count_params(model)
    params_ratio = params / baseline["params"]

    status, reason = "valid", ""
    if prune_seconds > PRUNE_BUDGET_S:
        status, reason = "invalid", f"prune_seconds {prune_seconds:.0f} > {PRUNE_BUDGET_S}"

    is_baseline_run = params_ratio > 0.999
    if not is_baseline_run and params_ratio > PARAMS_RATIO_TARGET and status == "valid":
        status, reason = "invalid", f"params_ratio {params_ratio:.3f} > {PARAMS_RATIO_TARGET}"

    reloaded, rt_err = roundtrip(model, tokenizer)
    if rt_err is not None:
        status, reason = "invalid", f"roundtrip failed: {rt_err[:200]}"
        val_ppl = float("nan")
    else:
        val_ppl = evaluate_ppl(reloaded, tokenizer, device)

    print("---")
    print(f"val_ppl:          {val_ppl:.6f}")
    print(f"baseline_ppl:     {baseline['val_ppl']:.6f}")
    print(f"ppl_ratio:        {val_ppl / baseline['val_ppl']:.4f}")
    print(f"params_M:         {params / 1e6:.2f}")
    print(f"params_ratio:     {params_ratio:.4f}")
    print(f"prune_seconds:    {prune_seconds:.1f}")
    print(f"peak_mem_gb:      {peak_mem_gb():.1f}")
    print(f"roundtrip:        {'ok' if rt_err is None else 'FAIL'}")
    print(f"status:           {status}{(' (' + reason + ')') if reason else ''}")


if __name__ == "__main__":
    main()
