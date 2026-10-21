"""GlobSDB builds its training DEM through a fetchez Recipe."""

import fetchez.recipe
from fetchez.spatial import Region

from globato.modules.glob_sdb import GlobSDB


def test_run_builds_the_training_dem_with_a_fetchez_recipe(tmp_path, monkeypatch):
    """run() hands fetchez's Recipe a micro-recipe for the training source."""
    configs = []

    class FakeRecipe:
        def run(self):
            pass  # writes no training DEM, so run() stops after the recipe

    def from_file(config):
        configs.append(config)
        return FakeRecipe()

    monkeypatch.setattr(fetchez.recipe.Recipe, "from_file", staticmethod(from_file))
    module = GlobSDB(
        src_region=Region(w=-80.0, e=-79.0, s=25.0, n=26.0),
        outdir=str(tmp_path),
        train_source="mbdb",
    )

    assert module.run() is module
    [config] = configs
    assert config["region"] == [-80.0, -79.0, 25.0, 26.0]
    assert config["modules"] == [{"module": "mbdb"}]
