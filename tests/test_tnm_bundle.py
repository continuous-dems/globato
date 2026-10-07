from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
BUNDLES = ROOT / "src" / "globato" / "modules" / "bundles"


def _load(name):
    return yaml.safe_load((BUNDLES / name).read_text())


def _hooks(module):
    return [hook["name"] for hook in module.get("hooks", [])]


def test_glob_tnm_uses_current_products_interface_and_policy_owned_weights():
    bundle = _load("glob_tnm.yaml")
    modules = bundle["modules"]
    assert [m["args"]["products"] for m in modules] == [
        "s1m",
        "1m",
        "1_9as",
        "5m",
        "1_3as",
        "1_as",
        "2_as",
    ]
    assert all("weight" not in m["args"] for m in modules)
    assert all("datasets" not in m["args"] for m in modules)


def test_generic_merged_footprint_hooks_are_reused():
    modules = {m["args"]["products"]: m for m in _load("glob_tnm.yaml")["modules"]}
    assert "remote_raster_footprint" in _hooks(modules["s1m"])
    assert "remote_raster_footprint" in _hooks(modules["5m"])
    assert "remote_archive_footprint" in _hooks(modules["1_9as"])
    assert "tnm-wesm-coverage" in _hooks(modules["1m"])


def test_standalone_tnm_has_no_coastline_policy():
    text = (BUNDLES / "glob_tnm.yaml").read_text().lower()
    assert "coastline" not in text.replace("contains no coastline policy", "")
    assert "point_raster_mask" not in text


def test_every_tnm_tier_uses_policy_and_grid_claim_filter():
    for module in _load("glob_tnm.yaml")["modules"]:
        names = _hooks(module)
        assert "tnm-policy" in names
        assert "claim-grid-filter" in names
