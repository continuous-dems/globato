from importlib import import_module

from fetchez.registry import HookRegistry


def test_new_hook_classes_register_with_fetchez_plugin_registry():
    HookRegistry.get_registry(clear_registry=True)
    for module_name in (
        "fetchez.hooks.spatial_claim",
        "globato.hooks.filters.claim_grid",
        "globato.hooks.filters.tnm_policy",
        "globato.hooks.filters.tnm_wesm",
    ):
        HookRegistry._register_from_module(import_module(module_name))
    for name in (
        "spatial-claim",
        "claim-grid-filter",
        "tnm-policy",
        "tnm-wesm-coverage",
    ):
        assert HookRegistry.get_class(name) is not None, name
