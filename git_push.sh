#!/usr/bin/env bash
set -euo pipefail

# Optional proxy (uncomment if needed)
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7891

# One-click add/commit/push for AIOpsChallenge_Hybrid
# Usage: ./git_push.sh

msg="update"
repo_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_dir"

git add .
git commit -m "$msg"
git push origin main
