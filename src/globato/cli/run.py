"""Globato recipe command, including module-attached Stack QC preparation.

Reuse stock Fetchez CLI options, recipe loading, modifiers and execution. Only
Globato chooses StackQCRecipe; no Fetchez command or process globals are patched.
"""

from pathlib import Path
from copy import deepcopy

import click
from fetchez.cli.run import run_recipe as fetchez_run
from fetchez.recipe import load_recipe_config
from fetchez.registry import RecipeRegistry
from fetchez.utils import parse_hook_string
from globato.qc.stack.pipeline import StackQCRecipe as Recipe


@click.command("run", cls=type(fetchez_run), params=fetchez_run.params)
def run_recipe(
    name, region, region_srs, outdir, shared_cache, modifier, schema, refresh, fail_fast
):
    """Execute a Globato recipe from a YAML file or the recipe registry."""
    RecipeRegistry.load_all()
    try:
        if Path(name).is_file():
            config = load_recipe_config(name)
        else:
            metadata = RecipeRegistry.get_yaml(name)
            config = metadata.get("config") if metadata else None
        if not config:
            raise ValueError(
                f"Recipe '{name}' was not found as a file or registered recipe"
            )
        # Registry configurations may be reused by subsequent invocations.
        # CLI overrides must not mutate the cached shared recipe definition.
        config = deepcopy(config)
        if region:
            config["region"] = region
            config["region_srs"] = region_srs
            for hook in config.get("global_hooks", []):
                if "region" in hook.get("args", {}):
                    hook["args"]["region"] = region
        if modifier:
            config["modifiers"] = [parse_hook_string(item) for item in modifier]
        if schema:
            config["schemas"] = list(schema)
        recipe = Recipe.from_dict(config)
        for _ in recipe.run(
            outdir=outdir,
            shared_cache=shared_cache,
            refresh=refresh,
            ignore_failures=not fail_fast,
        ):
            pass
    except Exception as exc:
        # A failed build must never report success to a shell/tee workflow.
        raise click.ClickException(str(exc)) from exc
    click.secho(f"Successfully executed {name} recipe!", fg="green")
