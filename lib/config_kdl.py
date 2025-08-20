#!/usr/bin/env python3
"""
KDL Configuration Loader (ckdl only)

Parses KDL files using ckdl and extracts configuration maps for
the three CLI tools: recognizer, synthesizer, integrated.

Docs: https://kdl.dev ; Parser: https://github.com/tjol/ckdl
"""

from __future__ import annotations

from typing import Any, Iterable, Optional
import os
import re
import ckdl  # type: ignore


def _kdl_value_to_python(value: Any) -> Any:
    """Convert ckdl value objects to native Python types where appropriate."""
    if hasattr(value, "value"):
        try:
            return value.value
        except Exception:
            pass
    return value


def _collect_node_kv(nodes: Iterable[Any]) -> dict[str, Any]:
    """Collect key/value pairs from a list of KDL nodes.

    Rules:
    - For a node like: `key <value>` -> use first argument as value
    - For props on a node: `section key=value` -> include those pairs
    - If both args and props exist, props win for duplicate keys
    - Child nodes are ignored by default for flat configs
    """
    result: dict[str, Any] = {}
    for node in nodes:
        # argument-style: key 123 or key "text"
        node_args = getattr(node, "arguments", None)
        if node_args is None:
            node_args = getattr(node, "args", None)
        if isinstance(node_args, (list, tuple)) and len(node_args) >= 1:
            key = getattr(node, "name", "")
            result[key] = _kdl_value_to_python(node_args[0])
        # property-style: key=a other=b
        prop_map = getattr(node, "properties", None)
        if prop_map is None:
            prop_map = getattr(node, "props", None)
        if isinstance(prop_map, dict):
            for prop_name, prop_val in prop_map.items():
                result[prop_name] = _kdl_value_to_python(prop_val)
    return result


def _find_child(container: Any, name: str) -> Optional[Any]:
    """Find first child node by name."""
    children = getattr(container, "children", None)
    if isinstance(children, (list, tuple)):
        for n in children:
            if getattr(n, "name", None) == name:
                return n
    # Fallback for other parsers/older structures
    nodes = getattr(container, "nodes", None)
    if isinstance(nodes, (list, tuple)):
        for n in nodes:
            if getattr(n, "name", None) == name:
                return n
    return None


def _get_section_nodes(doc: Any, section_name: str) -> Iterable[Any]:
    """Return child nodes for a specific top-level section node if present.

    Supports:
    - a top-level node named exactly `section_name` whose children are kv nodes
    - a generic `config` top-level node with child `section_name { ... }`
    - falling back to treating the top-level as the section itself if it has
      direct kv nodes matching our keys
    """
    section_node = _find_child(doc, section_name)
    if section_node is not None and getattr(section_node, "children", None):
        return section_node.children
    # If not found, return empty to signal missing section
    return []


def load_kdl_config(path: str, section: str) -> dict[str, Any]:
    """Load a KDL file and extract a flat dict for the given section.

    section: one of "recognizer", "synthesizer", "integrated"
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    doc = ckdl.parse(source)  # type: ignore[attr-defined]
    nodes = list(_get_section_nodes(doc, section))
    if not nodes:
        raise ValueError(f"Missing section: {section}")
    return _collect_node_kv(nodes)


def list_present_sections(path: str) -> list[str]:
    # Deprecated: retained for compatibility, now a thin wrapper over parse_config_doc
    doc = parse_config_doc(path)
    candidates = ["recognizer", "synthesizer", "integrated"]
    present: list[str] = []
    for name in candidates:
        if _find_child(doc, name) is not None:
            present.append(name)
    return present


def apply_config_over_args(
    args: Any,
    config: dict[str, Any],
    flag_presence_lookup: Optional[dict[str, bool]] = None,
    key_map: Optional[dict[str, str]] = None,
):
    """Apply a flat config dict into an argparse.Namespace in a precedence-aware way.

    Precedence: explicit CLI flags > KDL config > argparse defaults

    - args: argparse.Namespace to mutate
    - config: flat key/value map loaded from KDL
    - flag_presence_lookup: mapping of argparse dest name to whether that option
      was provided on the CLI (derived from sys.argv)
    - key_map: optional mapping from KDL key name to argparse attr name
    """
    presence = flag_presence_lookup or {}
    mapping = key_map or {}

    def was_provided(dest: str) -> bool:
        return bool(presence.get(dest, False))

    # Normalize keys and apply
    for k, v in config.items():
        # Map alternative key names commonly used in KDL files
        normalized_key = _NORMALIZE_KEY_MAP.get(k, k)

        dest = mapping.get(normalized_key, normalized_key)

        # Special handling: auto_synthesize maps to no_speak inverse
        if dest == "auto_synthesize":
            if not was_provided("no_speak") and hasattr(args, "no_speak"):
                setattr(args, "no_speak", not bool(v))
            continue

        if was_provided(dest):
            continue
        if hasattr(args, dest):
            setattr(args, dest, v)

    return args


# Common key normalization used across loaders and appliers
_NORMALIZE_KEY_MAP: dict[str, str] = {
    "speaker": "speaker_id",
    "speaker-id": "speaker_id",
    "compute-type": "compute_type",
    "silence-threshold": "silence_threshold",
    "chunk-duration": "chunk_duration",
    "sample-rate": "sample_rate",
    "auto-synthesize": "auto_synthesize",
}


def _normalize_keys(d: dict[str, Any]) -> dict[str, Any]:
    if not d:
        return {}
    out: dict[str, Any] = {}
    for k, v in d.items():
        out[_NORMALIZE_KEY_MAP.get(k, k)] = v
    return out


def compose_integrated_config(path: str) -> dict[str, Any]:
    """Compose integrated config by merging recognizer + synthesizer + integrated overrides.

    Precedence: integrated > recognizer/synthesizer
    """
    # Load available sections; missing sections yield empty dicts
    try:
        rec = _normalize_keys(load_kdl_config(path, "recognizer"))
    except Exception:
        rec = {}
    try:
        syn = _normalize_keys(load_kdl_config(path, "synthesizer"))
    except Exception:
        syn = {}
    try:
        integ = _normalize_keys(load_kdl_config(path, "integrated"))
    except Exception:
        integ = {}

    combined: dict[str, Any] = {}
    combined.update(rec)
    combined.update(syn)
    combined.update(integ)
    return combined


def get_explicit_mode(path: str) -> Optional[str]:
    # Deprecated: use get_explicit_mode_from_doc with parse_config_doc
    doc = parse_config_doc(path)
    return get_explicit_mode_from_doc(doc)


def parse_config_doc(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    return ckdl.parse(source)  # type: ignore[attr-defined]


def get_explicit_mode_from_doc(doc: Any) -> Optional[str]:
    # Prefer a top-level node named 'mode'
    mode_node = _find_child(doc, "mode")
    if mode_node is None:
        return None
    # ckdl exposes positional args under .args
    args = getattr(mode_node, "args", None)
    if isinstance(args, list) and args:
        val = getattr(args[0], "value", args[0])
        if isinstance(val, str):
            low = val.strip().lower()
            if low in {"recognizer", "synthesizer", "integrated"}:
                return low
    return None


def load_section_from_doc(doc: Any, section: str) -> dict[str, Any]:
    nodes = list(_get_section_nodes(doc, section))
    if not nodes:
        return {}
    return _collect_node_kv(nodes)


def compose_integrated_from_doc(doc: Any) -> dict[str, Any]:
    rec = _normalize_keys(load_section_from_doc(doc, "recognizer"))
    syn = _normalize_keys(load_section_from_doc(doc, "synthesizer"))
    integ = _normalize_keys(load_section_from_doc(doc, "integrated"))
    combined: dict[str, Any] = {}
    combined.update(rec)
    combined.update(syn)
    combined.update(integ)
    return combined
