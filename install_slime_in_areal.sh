#!/usr/bin/env bash
set -euo pipefail

# Configure paths and optional proxy settings, then use Slime's standard build.
export SLIME_DIR="${SLIME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
export BASE_DIR="${BASE_DIR:-$SLIME_DIR/../slime-runtime}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$BASE_DIR/micromamba}"

if [[ -n "${SLIME_PROXY_URL:-}" ]]; then
    export http_proxy="$SLIME_PROXY_URL"
    export https_proxy="$SLIME_PROXY_URL"
    export HTTP_PROXY="$SLIME_PROXY_URL"
    export HTTPS_PROXY="$SLIME_PROXY_URL"
fi
export no_proxy="${no_proxy:-${NO_PROXY:-localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16}}"
export NO_PROXY="$no_proxy"

mkdir -p "$BASE_DIR" "$MAMBA_ROOT_PREFIX"
cd "$SLIME_DIR"
bash build_conda.sh
