#!/usr/bin/env bash
# One-time setup inside WSL2 Ubuntu 24.04 (run from the repo root inside the Linux
# file system, e.g. ~/slideqr-seg). The apt step needs your sudo password, so it is
# printed rather than run:
#   sudo apt update && sudo apt install -y python3-venv python3-pip libdmtx0b fonts-dejavu
set -euo pipefail

for pkg in python3-venv libdmtx0b fonts-dejavu; do
  dpkg -s "$pkg" >/dev/null 2>&1 || { echo "missing $pkg: sudo apt install -y python3-venv python3-pip libdmtx0b fonts-dejavu"; exit 1; }
done
case "$(pwd)" in /mnt/*) echo "run this from the Linux file system (e.g. ~/slideqr-seg), not /mnt/c"; exit 1;; esac

python3 -m venv .venv
. .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -q -r requirements-train.txt

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not visible inside WSL"
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0),
      "sm_%d%d" % torch.cuda.get_device_capability(0), "archs", torch.cuda.get_arch_list())
x = torch.randn(1024, 1024, device="cuda")
print("matmul ok:", bool((x @ x).isfinite().all()))
EOF
echo "setup done"
