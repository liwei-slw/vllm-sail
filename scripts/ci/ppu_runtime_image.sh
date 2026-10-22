#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Resolve a runner-preloaded CUDA-compatible PPU runtime image to an immutable ID.
set -Eeuo pipefail

override=${1:-}
configured=${PPU_CI_RUNTIME_IMAGE:-}
known_image_id=sha256:55454f6df5e6a63f3d53741962615225cac4b562505f1584c0166eaf2bacb168
known_runtime=pkg.flytiger-eco.com/docker_release/llm@sha256:889a6504ad16ec5766c5f7bccceb2881276899dbb4e2ab965f312a47e8cefac8

if [[ -n "${override}" ]]; then
    candidates=("${override}")
elif [[ -n "${configured}" ]]; then
    candidates=("${configured}")
else
    candidates=("${known_image_id}" "${known_runtime}")
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
