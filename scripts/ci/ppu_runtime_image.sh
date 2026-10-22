#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Resolve a runner-preloaded CUDA-compatible PPU runtime image to an immutable ID.
set -Eeuo pipefail

override=${1:-}
configured=${PPU_CI_RUNTIME_IMAGE:-}
known_image_id=sha256:8e4b3adfef4f6051cd30516144ed32c25b700f96f861b43e233447b1c90ceb89
legacy_runtime=reg.docker.alibaba-inc.com/ai-tmp/llm@sha256:24872709b1a84ec0e91260eeeccf6ae7ba52a6c70fb90f55a7aae1c1df503bcb

if [[ -n "${override}" ]]; then
    candidates=("${override}")
elif [[ -n "${configured}" ]]; then
    candidates=("${configured}")
else
    candidates=("${known_image_id}" "${legacy_runtime}")
fi

for candidate in "${candidates[@]}"; do
    if image_id=$(docker image inspect "${candidate}" --format '{{.Id}}' 2>/dev/null); then
        [[ "${image_id}" =~ ^sha256:[0-9a-f]{64}$ ]] || {
            echo "Docker did not resolve an immutable runtime image ID for ${candidate}" >&2
            exit 2
        }
        printf 'Resolved preloaded PPU runtime image %s as %s\n' \
            "${candidate}" "${image_id}" >&2
        printf '%s\n' "${image_id}"
        exit 0
    fi
done

echo "No allowed CUDA-compatible PPU runtime image is preloaded on this runner." >&2
echo "Requested candidates: ${candidates[*]}" >&2
echo "Available Docker images:" >&2
docker image ls --digests --no-trunc >&2 || true
exit 2
