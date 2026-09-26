#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Select products from a Globato bundle before Fetchez recipe expansion."""

from __future__ import annotations

import copy
import importlib.resources
from collections.abc import Sequence
from typing import Any

from fetchez.registry import BundleRegistry, PresetRegistry
from fetchez.utils import parse_arg_to_list


def _parse_products(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = parse_arg_to_list(value, str)
    elif isinstance(value, Sequence):
        values = [str(v) for v in value]
    else:
        values = [str(value)]
    return list(dict.fromkeys(v.strip().lower() for v in values if str(v).strip()))


def _child_products(module: dict[str, Any]) -> list[str]:
    return _parse_products(module.get("args", {}).get("products"))


def _load_globato_bundle_resources() -> None:
    """Register package-owned Globato bundles when entry points are unavailable.

    Normal Fetchez entry-point discovery remains authoritative. This fallback is
    only reached for a bundle name that was not registered after ``load_all()``;
    it makes exact Globato source checkouts behave like installed Globato wheels.
    """

    try:
        bundle_root = importlib.resources.files("globato.modules.bundles")
        for file_path in bundle_root.iterdir():
            if file_path.name.endswith((".yaml", ".yml")):
                BundleRegistry._register_yaml(
                    "globato",
                    file_path.read_text(encoding="utf-8"),
                    str(file_path),
                )
    except (FileNotFoundError, ImportError, ModuleNotFoundError, TypeError):
        # Generic BundleRegistry expansion will report the normal unknown-bundle
        # error if neither installed discovery nor package resources can find it.
        return


def _get_bundle(target: str) -> dict[str, Any] | None:
    bundle = BundleRegistry.get_yaml(target)
    if bundle is not None:
        return bundle

    _load_globato_bundle_resources()
    return BundleRegistry.get_yaml(target)


def _normalize_bundle_item(item: Any) -> Any:
    """Recover bundle identity lost by generic source-string compilation.

    Fetchez ``compile_sources()`` only recognizes a bundle when the complete
    source string exactly matches a registered bundle key. Parameterized public
    syntax such as ``glob-tnm:products=1m/1_3as`` therefore arrives here as a
    normal-looking module dictionary, and a bare bundle does the same in an
    exact-source runtime where installed entry-point metadata is unavailable.

    If that parsed module name resolves to a real bundle, restore the bundle key
    before parameterized or generic bundle expansion. Unknown module names stay
    untouched and retain the normal Recipe validation/error path.
    """

    if not isinstance(item, dict) or item.get("bundle") or not item.get("module"):
        return item

    target = str(item["module"])
    # Do not reinterpret names of unrelated Fetchez bundles as modules here.
    # This recovery is only needed for Globato's opt-in public TNM shorthand.
    if target != "glob-tnm" or _get_bundle(target) is None:
        return item

    normalized = copy.deepcopy(item)
    normalized["bundle"] = normalized.pop("module")
    return normalized


def expand_parameterized_bundles(modules: list[Any]) -> list[Any]:
    """Expand product-selectable Globato bundles before generic expansion.

    Bundles without a top-level ``products`` declaration pass through unchanged.
    Supported wrapper arguments are deliberately small: ``products`` and
    ``weight``. Wrapper hooks are applied to every selected child using the same
    preset merge helper used by normal bundle inheritance.
    """

    BundleRegistry.load_all()
    PresetRegistry.load_all()
    expanded: list[Any] = []

    for item in modules:
        item = _normalize_bundle_item(item)
        if not isinstance(item, dict) or not item.get("bundle"):
            expanded.append(item)
            continue

        target = item["bundle"]
        bundle = _get_bundle(target)
        args = copy.deepcopy(item.get("args", {}))
        selected_arg = args.get("products")

        if not bundle or selected_arg is None or not bundle.get("products"):
            expanded.append(item)
            continue

        unknown_args = set(args).difference({"products", "weight"})
        if unknown_args:
            raise ValueError(
                f"Unknown parameterized bundle argument(s) for {target}: "
                + ", ".join(sorted(unknown_args))
            )

        canonical = [str(v).strip().lower() for v in bundle.get("products", [])]
        requested = _parse_products(selected_arg)
        if requested == ["all"] or not requested:
            requested = canonical

        unknown = sorted(set(requested).difference(canonical))
        if unknown:
            raise ValueError(
                f"Unknown product(s) for {target}: {', '.join(unknown)}; "
                f"supported: {'/'.join(canonical)}"
            )

        requested_set = set(requested)
        parent_weight = float(args.get("weight", 1.0))
        parent_hooks = copy.deepcopy(item.get("hooks", []))

        # Expand the unparameterized bundle using existing generic machinery,
        # then select child modules by their own products= identity. This keeps
        # nested-bundle behavior and default hook configuration centralized in
        # BundleRegistry rather than recreating it here.
        children = BundleRegistry.expand_modules([{"bundle": target}])
        matched: set[str] = set()
        for child in children:
            child = copy.deepcopy(child)
            child_products = _child_products(child)
            if not child_products:
                # Non-product helper modules in a parameterized bundle are kept.
                expanded.append(child)
                continue

            keep = [p for p in canonical if p in requested_set and p in child_products]
            if not keep:
                continue
            matched.update(keep)

            # A child can represent one or several canonical products. Narrow a
            # multi-product child to the selected intersection while preserving
            # canonical order.
            child.setdefault("args", {})["products"] = "/".join(keep)
            child["args"]["weight"] = (
                float(child["args"].get("weight", 1.0)) * parent_weight
            )
            if parent_hooks:
                child["hooks"] = PresetRegistry.expand_hooks(
                    child.get("hooks", []), parent_hooks
                )
            expanded.append(child)

        missing = [p for p in requested if p not in matched]
        if missing:
            raise ValueError(
                f"Bundle {target} declares product(s) without matching child modules: "
                + ", ".join(missing)
            )

    return expanded
