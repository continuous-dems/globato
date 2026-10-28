#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inspect and safely manage Globato side-stack FusionState caches.

Register ``sidestack_group`` on the Globato Click CLI, e.g.::

    from .sidestack import sidestack_group
    main.add_command(sidestack_group)

Only files matching the side-stack suffix *and* carrying the appropriate
FusionState metadata are eligible for removal. Generated QC-side-stacks and
assembled MultiStacks are deliberately excluded.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import rasterio

SUFFIX = "_sidestack.tif"


@dataclass
class SideStackRecord:
    path: str
    size_bytes: int
    age_days: float
    valid: bool
    reason: str | None
    source_id: str | None
    survey_id: str | None
    digest: str | None
    source: dict | None
    processing: list | None
    grid: dict | None
    fusion_version: str | None


def _scan(directory: Path, recursive: bool = True):
    if not directory.is_dir():
        raise click.ClickException(f"Directory not found: {directory}")
    iterator = (
        directory.rglob(f"*{SUFFIX}") if recursive else directory.glob(f"*{SUFFIX}")
    )
    # Never follow a symlink or accidentally delete something outside the tree.
    return sorted((p for p in iterator if p.is_file() and not p.is_symlink()), key=str)


def _record(path: Path, now: float | None = None) -> SideStackRecord:
    stat = path.stat()
    reason = None
    tags = {}
    identity = None
    try:
        with rasterio.open(path) as src:
            tags = src.tags()
            if tags.get("GLOBATO_FUSION_ROLE") in {
                "qc-side-stack",
                "collection-global",
                "multi-stack",
            }:
                reason = "not an original source side-stack"
            elif tags.get("GLOBATO_DATATYPE") != "FUSION_STATE":
                reason = "not tagged FUSION_STATE"
            elif not tags.get("GLOBATO_SIDESTACK_HASH"):
                reason = "missing side-stack cache hash"
            elif src.count != 7:
                reason = f"expected 7 bands, got {src.count}"
            elif src.width <= 0 or src.height <= 0:
                reason = "invalid raster dimensions"
            raw = tags.get("GLOBATO_SIDESTACK_IDENTITY")
            if raw:
                try:
                    identity = json.loads(raw)
                    if not isinstance(identity, dict):
                        raise ValueError("not an object")
                except (ValueError, TypeError) as exc:
                    reason = reason or f"invalid identity metadata: {exc}"
    except (OSError, rasterio.errors.RasterioError, ValueError) as exc:
        reason = f"cannot open raster: {exc}"

    identity = identity or {}
    return SideStackRecord(
        path=str(path.resolve()),
        size_bytes=stat.st_size,
        age_days=max(
            0.0, ((time.time() if now is None else now) - stat.st_mtime) / 86400
        ),
        valid=reason is None,
        reason=reason,
        source_id=tags.get("GLOBATO_SOURCE_ID"),
        survey_id=tags.get("GLOBATO_SURVEY_ID"),
        digest=tags.get("GLOBATO_SIDESTACK_HASH"),
        source=identity.get("source"),
        processing=identity.get("processing"),
        grid=identity.get("grid"),
        fusion_version=tags.get("GLOBATO_FUSION_VERSION"),
    )


def _records(directory, recursive, survey, older_than, include_invalid=False):
    found = [_record(p) for p in _scan(Path(directory).expanduser(), recursive)]
    selected = [
        r
        for r in found
        if (include_invalid or r.valid)
        and (survey is None or r.survey_id == survey)
        and (older_than is None or r.age_days >= older_than)
    ]
    return found, selected


def _human_size(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024


@click.group(name="sidestack")
def sidestack_group():
    """Inspect or clear per-source side-stack FusionState TIFF caches."""


@sidestack_group.command("info")
@click.option(
    "-D",
    "--dir",
    "directory",
    default=".",
    show_default=True,
    type=click.Path(file_okay=False),
)
@click.option("--recursive/--no-recursive", default=True, show_default=True)
@click.option("--survey", help="Exact GLOBATO_SURVEY_ID filter.")
@click.option(
    "--older-than",
    type=click.FloatRange(min=0),
    help="Include files at least this many days old.",
)
@click.option(
    "--details", is_flag=True, help="Display identity and processing metadata."
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable records.")
def info(directory, recursive, survey, older_than, details, as_json):
    """List side-stack caches, their source identity, survey and disk usage."""
    found, selected = _records(
        directory, recursive, survey, older_than, include_invalid=True
    )
    if as_json:
        click.echo(
            json.dumps(
                {"total_found": len(found), "records": [asdict(r) for r in selected]},
                indent=2,
            )
        )
        return
    if not found:
        click.echo("No side-stack files found.")
        return
    for r in selected:
        status = "OK" if r.valid else f"INVALID: {r.reason}"
        identity = r.survey_id or "(no survey)"
        source = r.source_id or (r.source or {}).get("basename") or "(no source tag)"
        click.echo(
            f"{status:>10}  {_human_size(r.size_bytes):>10}  {r.age_days:6.1f}d  survey={identity}  source={source}"
        )
        click.echo(f"    {r.path}")
        if details:
            click.echo(
                json.dumps(
                    {
                        "source": r.source,
                        "grid": r.grid,
                        "processing": r.processing,
                        "digest": r.digest,
                        "fusion_version": r.fusion_version,
                    },
                    indent=2,
                    default=str,
                )
            )
    click.echo(
        f"Found: {len(found)}  Listed: {len(selected)}  Size listed: {_human_size(sum(r.size_bytes for r in selected))}"
    )


@sidestack_group.command("clear")
@click.option(
    "-D",
    "--dir",
    "directory",
    default=".",
    show_default=True,
    type=click.Path(file_okay=False),
)
@click.option("--recursive/--no-recursive", default=True, show_default=True)
@click.option("--survey", help="Only remove caches with this exact survey ID.")
@click.option(
    "--older-than",
    type=click.FloatRange(min=0),
    help="Only remove caches at least this many days old.",
)
@click.option(
    "--execute",
    is_flag=True,
    help="Actually unlink matched files (default: preview only).",
)
@click.option(
    "-y", "--yes", is_flag=True, help="Skip confirmation when used with --execute."
)
def clear(directory, recursive, survey, older_than, execute, yes):
    """Preview or remove verified side-stack cache files (not QC outputs)."""
    found, selected = _records(directory, recursive, survey, older_than)
    invalid = len(found) - len([r for r in found if r.valid])
    if not selected:
        click.echo(
            f"No eligible side-stacks found ({invalid} invalid/unverified skipped)."
        )
        return
    for r in selected:
        click.echo(f"  {_human_size(r.size_bytes):>10}  {r.path}")
    click.echo(
        f"Selected {len(selected)} verified side-stacks ({_human_size(sum(r.size_bytes for r in selected))}); "
        f"{invalid} invalid/unverified skipped."
    )
    if not execute:
        click.secho("Dry run only. Pass --execute to delete these files.", fg="yellow")
        return
    if not yes and not click.confirm(
        f"Permanently delete {len(selected)} side-stack TIFFs?", default=False
    ):
        click.echo("Cancelled.")
        return
    removed = 0
    for r in selected:
        path = Path(r.path)
        # Recheck metadata and identity at deletion time; never follow symlinks.
        if path.is_symlink() or not path.is_file():
            click.echo(f"Skipped changed/missing file: {path}")
            continue
        latest = _record(path)
        if not latest.valid or latest.digest != r.digest:
            click.echo(f"Skipped changed/unverified file: {path}")
            continue
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            click.secho(f"Could not delete {path}: {exc}", fg="red", err=True)
    click.secho(f"Deleted {removed} side-stack files.", fg="green")


if __name__ == "__main__":
    sidestack_group()
