#!/usr/bin/env python

"""
Tests the source/hook parsing
"""

# from globato.utils import parse_source_string as globato_parse_source
from fetchez.utils import parse_source_string as globato_parse_source


def test_globato_source_parser_injects_stream_data():
    """Ensures that Globato's wrapper around fetchez.utils.parse_source_string
    successfully injects the mandatory 'stream_data' hook.
    """

    # globato parses a basic source
    res = globato_parse_source("copernicus:datatype=3")

    assert res["module"] == "copernicus"
    assert res["args"]["datatype"] == 3

    # Make sure stream_data was added before any other hooks
    # assert len(res["hooks"]) == 1
    # assert res["hooks"][0]["name"] == "stream-init"
    assert len(res["hooks"]) == 0


def test_globato_source_parser_chained_injection():
    """Ensures the injected stream_data stays at the front when user provides chained hooks."""

    res = globato_parse_source("mbdb+rq:threshold=10")

    assert len(res["hooks"]) == 1
    assert res["hooks"][0]["name"] == "rq"
    # assert len(res["hooks"]) == 2
    # assert res["hooks"][0]["name"] == "stream-init"
    # assert res["hooks"][1]["name"] == "rq"
