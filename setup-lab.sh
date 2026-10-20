#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Lab bootstrap script (for the Prayogshala / Linux lab).
#
# The lab hands out a fresh virtual machine each session, so tools installed
# in a previous session (make, kubectl) are gone when you log back in - even
# though the files in your home/workspace folder persist. This script simply
# re-installs those two tools so the project's `make` commands work again.
#
# Docker, Python and Minikube are assumed to be pre-installed by the lab.
#
# Usage (from the project folder):
#     bash setup-lab.sh
#     make all
# ---------------------------------------------------------------------------
set -e

echo "== Lab bootstrap: making sure make and kubectl are installed =="

# --- make ---
if command -v make >/dev/null 2>&1; then
  echo "make    : already installed"
else
  echo "make    : installing..."
  if command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y make
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y make
  else
    echo "  Could not find dnf or apt-get - please install make manually." >&2
  fi
fi

# --- kubectl ---
if command -v kubectl >/dev/null 2>&1; then
  echo "kubectl : already installed"
else
  echo "kubectl : installing..."
  KVER="$(curl -L -s https://dl.k8s.io/release/stable.txt)"
  curl -LO "https://dl.k8s.io/release/${KVER}/bin/linux/amd64/kubectl"
  chmod +x kubectl
  sudo mv kubectl /usr/local/bin/
fi

echo ""
echo "== Installed versions =="
command -v make    >/dev/null 2>&1 && make --version | head -1
command -v kubectl >/dev/null 2>&1 && kubectl version --client 2>/dev/null | head -1
command -v docker  >/dev/null 2>&1 && docker --version          || echo "docker  : NOT found"
command -v python3 >/dev/null 2>&1 && python3 --version          || echo "python3 : NOT found"
command -v minikube>/dev/null 2>&1 && minikube version | head -1 || echo "minikube: NOT found"

echo ""
echo "Bootstrap complete. Next: run 'make all'"
