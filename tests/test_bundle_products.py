import copy

import pytest

from globato.bundle_products import expand_parameterized_bundles


def _install_fake_bundle(monkeypatch):
    bundle = {
        "name": "fake-products",
        "products": ["fine", "medium", "coarse"],
        "modules": [
            {"module": "tnm", "args": {"products": "fine", "weight": 5.0}, "hooks": []},
            {
                "module": "tnm",
                "args": {"products": "medium", "weight": 3.0},
                "hooks": [],
            },
            {
                "module": "tnm",
                "args": {"products": "coarse", "weight": 1.0},
                "hooks": [],
            },
        ],
    }
    import globato.bundle_products as bp

    monkeypatch.setattr(bp.BundleRegistry, "load_all", classmethod(lambda cls: None))
    monkeypatch.setattr(bp.PresetRegistry, "load_all", classmethod(lambda cls: None))
    monkeypatch.setattr(
        bp.BundleRegistry,
        "get_yaml",
        classmethod(
            lambda cls, name: copy.deepcopy(bundle) if name == "fake-products" else None
        ),
    )

    def expand(cls, raw, parent_weight=1.0):
        assert raw == [{"bundle": "fake-products"}]
        return copy.deepcopy(bundle["modules"])

    monkeypatch.setattr(bp.BundleRegistry, "expand_modules", classmethod(expand))
    monkeypatch.setattr(
        bp.PresetRegistry,
        "expand_hooks",
        classmethod(
            lambda cls, hooks, parent_hooks=None: list(hooks) + list(parent_hooks or [])
        ),
    )


def test_product_subset_uses_bundle_canonical_order(monkeypatch):
    _install_fake_bundle(monkeypatch)
    modules = expand_parameterized_bundles(
        [{"bundle": "fake-products", "args": {"products": "coarse/fine"}}]
    )
    assert [m["args"]["products"] for m in modules] == ["fine", "coarse"]
    assert [m["args"]["weight"] for m in modules] == [5.0, 1.0]


def test_product_selection_deduplicates_and_supports_all(monkeypatch):
    _install_fake_bundle(monkeypatch)
    modules = expand_parameterized_bundles(
        [{"bundle": "fake-products", "args": {"products": "all"}}]
    )
    assert [m["args"]["products"] for m in modules] == ["fine", "medium", "coarse"]


def test_parameterized_bundle_weight_and_hooks_propagate(monkeypatch):
    _install_fake_bundle(monkeypatch)
    modules = expand_parameterized_bundles(
        [
            {
                "bundle": "fake-products",
                "args": {"products": "medium", "weight": 2},
                "hooks": [
                    {"name": "stream_reproject", "args": {"dst_srs": "EPSG:4269"}}
                ],
            }
        ]
    )
    assert modules[0]["args"]["weight"] == 6.0
    assert modules[0]["hooks"][-1]["name"] == "stream_reproject"


def test_unknown_product_fails(monkeypatch):
    _install_fake_bundle(monkeypatch)
    with pytest.raises(ValueError, match="Unknown product"):
        expand_parameterized_bundles(
            [{"bundle": "fake-products", "args": {"products": "bogus"}}]
        )


def test_bundle_without_products_is_untouched(monkeypatch):
    import globato.bundle_products as bp

    monkeypatch.setattr(bp.BundleRegistry, "load_all", classmethod(lambda cls: None))
    monkeypatch.setattr(bp.PresetRegistry, "load_all", classmethod(lambda cls: None))
    monkeypatch.setattr(
        bp.BundleRegistry,
        "get_yaml",
        classmethod(lambda cls, name: {"name": "plain", "modules": []}),
    )
    item = {"bundle": "plain", "args": {"products": "x"}}
    assert expand_parameterized_bundles([item]) == [item]
