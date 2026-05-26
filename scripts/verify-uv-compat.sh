#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${SCONCE_COMPAT_ENV:-/tmp/sconce-uv-compat}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/tmp/conda-pkgs}"
CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/tmp/conda-envs}"

cd "$(dirname "$0")/.."

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  echo "Creating temporary conda env: $ENV_PREFIX"
  env \
    CONDA_NO_PLUGINS=true \
    CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS" \
    CONDA_ENVS_PATH="$CONDA_ENVS_PATH" \
    CONDA_SOLVER=classic \
    conda create -p "$ENV_PREFIX" python=3.12 -y
fi

echo "Installing sconce with PyTorch/HF experiment dependencies..."
env UV_CACHE_DIR="$UV_CACHE_DIR" \
  uv pip install --python "$ENV_PREFIX/bin/python" -e '.[experiments,dev]'

echo "Running version/import smoke checks..."
"$ENV_PREFIX/bin/python" - <<'PY'
import json
import sys

import torch
import torchaudio
import torchvision
import transformers
from transformers import AutoModelForCausalLM, AutoModelForImageClassification, AutoTokenizer

import sconce

print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "torchvision": torchvision.__version__,
    "torchaudio": torchaudio.__version__,
    "transformers": transformers.__version__,
    "sconce_exports": len(sconce.__all__),
}, indent=2))

print(
    "hf imports ok:",
    AutoModelForCausalLM.__name__,
    AutoModelForImageClassification.__name__,
    AutoTokenizer.__name__,
)
PY

echo "Running core smoke checks..."
"$ENV_PREFIX/bin/python" - <<'PY'
import importlib.util

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sconce.perf import performance
from sconce.pruner import TransformerPruner

spec = importlib.util.spec_from_file_location("quanter_smoke", "sconce/quanter.py")
quanter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quanter)


class Harness(quanter.quantization):
    def __init__(self):
        self.model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        self.dataloader = {"train": loader, "test": loader}
        self.device = torch.device("cpu")
        self.qat_config = "x86"
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
        for inputs, targets in self.dataloader["train"]:
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(inputs), targets)
            loss.backward()
            self.optimizer.step()


q_model, prepared = Harness().qat()
print("qat smoke ok", type(q_model).__name__, type(prepared).__name__)


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 4
        self.head_dim = 2
        self.qkv = nn.Linear(8, 24)
        self.proj = nn.Linear(8, 8)


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attn()
        self.mlp = Mlp()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])


model = Model()
tp = TransformerPruner(model, device=torch.device("cpu"))
assert len(tp.find_layers()) == 2
tp.prune({0: {"attention": 0.5, "ffn": 0.5}})
assert model.blocks[0].attn.qkv.out_features == 12
assert model.blocks[0].attn.proj.in_features == 4
assert model.blocks[0].mlp.fc1.out_features == 8
assert model.blocks[0].mlp.fc2.in_features == 8
print("transformer pruning smoke ok")

p = performance()
p.snn = False
lat = p.measure_latency(nn.Linear(4, 2), torch.randn(1, 4), n_warmup=1, n_test=2)
assert lat >= 0
print("latency smoke ok")
PY

echo "Running import and syntax checks..."
"$ENV_PREFIX/bin/python" -c "import importlib.util; spec=importlib.util.spec_from_file_location('vision', 'scripts/run_experiment.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('vision runner import ok')"
"$ENV_PREFIX/bin/python" -c "import importlib.util; spec=importlib.util.spec_from_file_location('llm', 'scripts/run_llm_experiment.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('llm runner import ok')"
"$ENV_PREFIX/bin/python" -m compileall -q sconce scripts

echo "Running pytest. Exit code 5 is expected until tests/ exists."
set +e
"$ENV_PREFIX/bin/python" -m pytest
pytest_rc=$?
set -e
if [[ "$pytest_rc" -ne 0 && "$pytest_rc" -ne 5 ]]; then
  exit "$pytest_rc"
fi

echo "Compatibility verification complete."
