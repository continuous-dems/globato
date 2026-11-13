"""Native collection aliases are resolved before coastline evidence and stacking."""

import copy
import shutil
from pathlib import Path

import yaml
from test_global_refinement import global_recipe, invoke


def test_url_alias_has_one_contribution_and_preserves_native_processing_lineage(
    tmp_path,
):
    recipe, _, config = global_recipe(tmp_path)
    producer = next(
        m
        for m in config["modules"]
        if any(
            h.get("name") == "coast_mask_source"
            and h.get("args", {}).get("evidence")
            and h["args"].get("remove") == "water"
            for h in m["hooks"]
        )
    )
    alias = copy.deepcopy(producer)
    path = tmp_path / "same-source-another-url.tif"
    shutil.copyfile(producer["args"]["path"], path)
    alias["args"]["path"] = str(path)
    config["modules"].append(alias)
    recipe.write_text(yaml.safe_dump(config, sort_keys=False))
    result = invoke(tmp_path, recipe)
    assert len(result["sources"]) == 5
    assert (
        len(
            {
                s["native_source_identity"]["native_dataset_id"]
                for s in result["sources"]
            }
        )
        == 5
    )
    aliases = [a for row in result["sources"] for a in row["source_aliases"]]
    assert len(aliases) == 1
    assert aliases[0]["comparison"] == "exact_seven_band_state"
    assert {Path(a["source_file"]).name for a in aliases} <= {
        Path(producer["args"]["path"]).name,
        path.name,
    }
