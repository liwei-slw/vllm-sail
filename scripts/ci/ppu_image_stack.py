# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

PROTECTED_PACKAGES = (
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "flash-attn",
    "flash-attn-3",
    "pla",
    "deep-gemm",
    "acext",
    "tokenspeed-mla",
    "tokenspeed-triton",
)


def tree_digest(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"missing or empty image component: {root}")
    digest = hashlib.sha256()
    count = 0

    def visit(directory, ancestors):
        nonlocal count
        resolved = directory.resolve()
        if resolved in ancestors:
            return
        ancestors = ancestors | {resolved}
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(root)
            if path.name == "__pycache__" or path.suffix in (".pyc", ".pyo"):
                continue
            if path.is_symlink():
                digest.update(f"link:{relative}:{path.readlink()}\0".encode())
            if path.is_dir():
                visit(path, ancestors)
            elif path.is_file():
                digest.update(f"file:{relative}:{path.stat().st_size}\0".encode())
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                count += 1

    visit(root, set())
    if not count:
        raise ValueError(f"missing or empty image component: {root}")
    return digest.hexdigest()


def snapshot(sdk, torch_root, torch_version, packages):
    return {
        "stack_source": "image",
        "fingerprint_kind": "installed-tree-sha256",
        "sdk_path": str(Path(sdk).resolve()),
        "sdk_sha256": tree_digest(sdk),
        "torch_path": str(Path(torch_root).resolve()),
        "torch_sha256": tree_digest(torch_root),
        "torch_version": torch_version,
        "packages": dict(packages),
    }


def verify_unchanged(before, after):
    if before != after:
        changed = sorted(
            key
            for key in before.keys() | after.keys()
            if before.get(key) != after.get(key)
        )
        raise ValueError(f"image stack changed: {', '.join(changed)}")


def package_versions():
    versions = {}
    for name in PROTECTED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def constraints_text(versions):
    return "".join(
        f"{name}==={versions[name]}\n" if name in versions else f"{name}<0\n"
        for name in PROTECTED_PACKAGES
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("capture", "verify"))
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--constraints", type=Path)
    args = parser.parse_args()

    import torch

    observed = snapshot(
        args.sdk,
        Path(torch.__file__).parent,
        str(torch.__version__),
        package_versions(),
    )
    if args.command == "capture":
        if args.constraints is None:
            parser.error("capture requires --constraints")
        args.output.write_text(json.dumps(observed, sort_keys=True, indent=2) + "\n")
        args.constraints.write_text(constraints_text(observed["packages"]))
        for key in (
            "stack_source",
            "fingerprint_kind",
            "sdk_sha256",
            "torch_sha256",
            "torch_version",
        ):
            print(f"{key}={observed[key]}")
    else:
        verify_unchanged(json.loads(args.output.read_text()), observed)
        print("stack_unchanged=true")


if __name__ == "__main__":
    main()
