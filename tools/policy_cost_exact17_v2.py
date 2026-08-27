#!/usr/bin/env python3
"""Versioned exact-17 exporter for the policy-cost-v2 dataset source."""
from __future__ import annotations

import importlib.util
from pathlib import Path


_PATH = Path(__file__).with_name("policy_cost_exact17.py")
_SPEC = importlib.util.spec_from_file_location("policy_cost_exact17_v1_impl", _PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load exact-17 implementation from {_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_IMPL)
_IMPL.DATASET_SOURCE = "tools/policy_cost_dataset_v2.c"

ExclusionError = _IMPL.ExclusionError
TEXT_SCHEMA = _IMPL.TEXT_SCHEMA
JSON_SCHEMA = _IMPL.JSON_SCHEMA
build_outputs = _IMPL.build_outputs
_canonical = _IMPL._canonical
_publish_pair = _IMPL._publish_pair


if __name__ == "__main__":
    _IMPL.main()
