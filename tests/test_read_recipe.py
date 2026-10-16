"""Regression tests for Globato's Recipe-backed streaming API.

Place in globato/tests/test_api_read_recipe.py (or the equivalent test folder).
These tests use the public read() entry point and capture Recipe.compile(), so
no network access, HDF5 reader installation, or data downloads are needed.
"""

from types import SimpleNamespace


from fetchez.recipe import Recipe
from globato.api import read
from globato.streams.base import GlobatoStream, GlobatoRasterStream


def _capture_compilations(monkeypatch):
    """Capture the prepared configuration *when the stream is consumed*."""
    seen = []

    def compile_(self, context=None, *, ignore_failures=False):
        seen.append((self.config, context, ignore_failures))
        return SimpleNamespace(
            modules=[],
            global_hooks=[],
            execute=lambda: None,
        )

    monkeypatch.setattr(Recipe, "compile", compile_)
    return seen


def test_ivert_file_read_preserves_legacy_data_type_and_pipe(monkeypatch, tmp_path):
    """IVERT's local ATL03 path and read(data_type=...) remain supported."""
    seen = _capture_compilations(monkeypatch)
    h5_file = tmp_path / "ATL03_001.h5"
    h5_file.touch()
    cache_dir = tmp_path / "icesat_cache"

    # Mirrors the IVERT shape without requiring an actual ICESat-2 reader.
    stream = read(
        str(h5_file),
        data_type="icesat-atl03",
        region=[-120, -119, 33, 34],
    ).pipe(
        "stream-init",
        classes="1/2/3/6/7/40/41/42",
        vertical_datum="mllw",
        reject_failed_qa=True,
        append_atl24=True,
        atl_version="006",
        cache_dir=str(cache_dir),
        use_external_masks=True,
    )

    assert isinstance(stream, GlobatoStream)
    assert seen == []  # Construction and pipe() must remain lazy.
    assert list(stream) == []
    assert len(seen) == 1

    config, context, ignore_failures = seen[0]
    assert config["modules"] == [
        {
            "module": str(h5_file),
            "args": {"data_type": "icesat-atl03"},
        }
    ]
    assert ignore_failures is False
    assert context.region is not None

    hooks = config["global_hooks"]
    assert len(hooks) == 1
    assert hooks[0]["name"] == "stream-init"
    assert hooks[0]["args"] == {
        "classes": "1/2/3/6/7/40/41/42",
        "vertical_datum": "mllw",
        "reject_failed_qa": True,
        "append_atl24": True,
        "atl_version": "006",
        "cache_dir": str(cache_dir),
        "use_external_masks": True,
    }
    # data_type belongs on the module, not on stream-init or set-entry.
    assert all(h.get("name") != "set-entry" for h in hooks)


def test_read_keyword_overrides_source_args_without_mutation(monkeypatch):
    """Legacy kwargs take precedence over an explicit module definition."""
    seen = _capture_compilations(monkeypatch)
    source = {
        "module": "example-module",
        "args": {"data_type": "old", "products": "1m"},
        "hooks": [{"name": "audit"}],
    }
    stream = read(source, data_type="icesat-atl03", extra_flag=True)
    assert source == {
        "module": "example-module",
        "args": {"data_type": "old", "products": "1m"},
        "hooks": [{"name": "audit"}],
    }

    assert list(stream) == []
    config, _, _ = seen[0]
    assert config["modules"] == [
        {
            "module": "example-module",
            "args": {
                "data_type": "icesat-atl03",
                "products": "1m",
                "extra_flag": True,
            },
            "hooks": [{"name": "audit"}],
        }
    ]
    assert source["args"]["data_type"] == "old"


def test_mixed_sources_receive_kwargs_and_compile_each_iteration(monkeypatch, tmp_path):
    """Overrides apply to each source, without mutating repeated runs."""
    seen = _capture_compilations(monkeypatch)
    path = tmp_path / "local.xyz"
    path.write_text("0 0 0\n", encoding="utf-8")
    source_defs = [str(path), {"module": "example-module", "args": {"year": 2020}}]
    stream = read(source_defs, data_type="xyz")

    assert list(stream) == []
    assert list(stream) == []
    assert len(seen) == 2
    expected = [
        {"module": str(path), "args": {"data_type": "xyz"}},
        {"module": "example-module", "args": {"year": 2020, "data_type": "xyz"}},
    ]
    for config, _, _ in seen:
        assert config["modules"] == expected
    assert source_defs == [
        str(path),
        {"module": "example-module", "args": {"year": 2020}},
    ]


def test_explicit_local_source_and_shorthand_share_recipe_path(monkeypatch, tmp_path):
    """Both source spellings reach Recipe; Recipe owns local_fs resolution."""
    seen = _capture_compilations(monkeypatch)
    h5_file = tmp_path / "ATL03.h5"
    h5_file.touch()

    assert list(read(str(h5_file), data_type="icesat-atl03")) == []
    assert (
        list(
            read(
                {
                    "module": str(h5_file),
                    "args": {"data_type": "icesat-atl03"},
                }
            )
        )
        == []
    )

    assert seen[0][0]["modules"] == seen[1][0]["modules"]


def test_target_srs_and_ignore_failures_preserved(monkeypatch):
    seen = _capture_compilations(monkeypatch)
    stream = read(
        "example-module",
        target_srs="EPSG:26911",
        ignore_failures=True,
    )
    assert stream.target_srs == "EPSG:26911"
    assert stream.stream_type == "point"
    assert list(stream) == []
    config, _, flag = seen[0]
    assert flag is True
    assert config["global_hooks"] == [
        {
            "name": "stream-reproject",
            "args": {"dst_srs": "EPSG:26911"},
        }
    ]


def test_globato_stream_raster_conversion_retains_recipe(monkeypatch):
    seen = _capture_compilations(monkeypatch)
    stream = read("example-module").to_raster("1s", want_sums=True)
    assert isinstance(stream, GlobatoRasterStream)
    assert stream.stream_type == "raster"
    assert list(stream) == []
    config, _, _ = seen[0]
    assert config["global_hooks"] == [
        {
            "name": "points2pixels",
            "args": {"x_inc": "1s", "y_inc": "1s", "want_sums": True},
        }
    ]

    stream = stream.to_points()
    assert isinstance(stream, GlobatoStream)
    assert list(stream) == []
    config, _, _ = seen[1]
    assert [hook["name"] for hook in config["global_hooks"]] == [
        "points2pixels",
        "pixels2points",
    ]
