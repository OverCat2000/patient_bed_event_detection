#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if [[ "${1:-}" == "--gpu" ]]; then
  pip install torch torchvision
else
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
echo "Done. Activate with: source .venv/bin/activate"
