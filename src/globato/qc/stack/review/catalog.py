"""Map finished QC products to the standalone Raster Viz generic JSON contract."""

from pathlib import Path
import hashlib
import json


def publish(path, catalog):
    """Validate with Globato's shared catalog contract, then publish atomically.

    The validator uses only the standard library; no viewer rendering runs here.
    Keep QC-specific receipt fields alongside the common display contract.
    """
    from globato.viz.raster_viz.schema import validate_document
    from ..io import write

    validate_document(catalog)
    write(path, catalog)


def document(stages, *, title, method, preferred_entry=None):
    """Retain legacy stage metadata and publish only displayable raster products.

    RGB hillshade downloads and PNG figures remain in the HTML/legacy list;
    elevation products contain actual elevations for meaningful 3-D previews.
    """
    products = []
    for stage in stages:
        role = stage["role"]
        if Path(stage["path"]).suffix.lower() not in {".tif", ".tiff"} or role.endswith(
            "_hillshade"
        ):
            continue
        coverage = role == "entry_coverage"
        removed = role == "entry_removed_observations"
        key = (
            role
            + "_"
            + hashlib.sha256(
                json.dumps(
                    [stage.get("entry_id"), stage["label"]], ensure_ascii=True
                ).encode()
            ).hexdigest()[:12]
        )
        metadata = {k: v for k, v in stage.items() if k not in {"path", "label"}}
        metadata["value_units"] = (
            "presence" if coverage else "observations" if removed else "m"
        )
        if not (coverage or removed):
            metadata["vertical_datum"] = "See raster CRS and recipe"
        products.append(
            dict(
                key=key,
                path=stage["path"],
                label=stage["label"],
                stage=role,
                group="coverage"
                if coverage
                else "removed_observations"
                if removed
                else "elevation",
                style="categorical"
                if coverage
                else "scalar"
                if removed
                else "elevation",
                metadata=metadata,
            )
        )
    defaults = {}
    pairs = (
        ["entry_before_measured", "entry_after_measured"]
        if preferred_entry
        else ["before_measured", "after_measured"]
    )
    selected = []
    for role in pairs:
        product = next(
            (
                p
                for p in products
                if p["stage"] == role
                and (
                    preferred_entry is None
                    or p["metadata"].get("entry_id") == preferred_entry
                )
            ),
            None,
        )
        if product:
            selected.append(product["key"])
    if len(selected) == 2:
        defaults["default"] = selected
    return dict(
        schema_version=1,
        title=title,
        running=False,
        products=products,
        defaults=defaults,
        schema="stack_qc_viz/1",
        complete=True,
        method=method,
        stages=stages,
    )
