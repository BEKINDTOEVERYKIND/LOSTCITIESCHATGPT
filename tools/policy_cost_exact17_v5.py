#!/usr/bin/env python3
"""Portable exact-17 exporter for the policy-cost-v5 dataset source.

The exclusion evidence binds the source that defines the information-view
orbit, while the platform-specific runtime ELF is sealed later by the build
identity and transport manifests.  This keeps the checked-in companion
portable across the locked GCC and Clang build lanes.
"""
from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path


_PATH = Path(__file__).with_name("policy_cost_exact17.py")
_SPEC = importlib.util.spec_from_file_location("policy_cost_exact17_v1_impl", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load exact-17 implementation from {_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_IMPL)
_IMPL.DATASET_SOURCE = "tools/policy_cost_dataset_v5.c"
_IMPL.JSON_SCHEMA = "lc-policy-cost-exclusions-evidence-v2"
_BUILD_OUTPUTS = _IMPL.build_outputs

ExclusionError = _IMPL.ExclusionError
TEXT_SCHEMA = _IMPL.TEXT_SCHEMA
JSON_SCHEMA = _IMPL.JSON_SCHEMA
_canonical = _IMPL._canonical
_publish_pair = _IMPL._publish_pair


def build_outputs(root: Path, hash_probe: Path):
    """Return portable evidence; bind the runtime ELF in the build identity."""

    text, evidence = _BUILD_OUTPUTS(root, hash_probe)
    native = evidence["bindings"]["native_hash_probe"]
    del native["binary_sha256"]
    native["runtime_binary_binding"] = (
        "dynamically_sealed_in_build_identity_and_transport"
    )
    digest_payload = dict(evidence)
    del digest_payload["canonical_payload_sha256"]
    evidence["canonical_payload_sha256"] = hashlib.sha256(
        _canonical(digest_payload)
    ).hexdigest()
    return text, evidence


_IMPL.build_outputs = build_outputs


if __name__ == "__main__":
    _IMPL.main()
