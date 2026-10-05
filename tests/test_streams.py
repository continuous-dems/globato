import numpy as np

from globato.streams.readers.base import BaseGlobatoReader


class MockGlobatoReader(BaseGlobatoReader):
    """A dummy reader that only yields X, Y, Z."""

    name = "mock_reader"
    meta_category = "point-stream"

    def _yield_raw_chunks(self):
        # Create a raw chunk missing all the required metadata columns
        chunk = np.zeros(10, dtype=[("x", "f8"), ("y", "f8"), ("z", "f4")])
        chunk["x"] = np.arange(10)
        chunk["z"] = 5.0
        yield chunk


def test_base_globato_reader_schema_enforcement():
    """Ensure BaseGlobatoReader automatically applies the strict Globato schema and weights."""

    # Initialize the reader with a custom weight and uncertainty
    reader = MockGlobatoReader("dummy.xyz", weight=3.5, uncertainty=0.1)

    # yield_chunks() should trigger the schema enforcer wrapper
    chunks = list(reader.yield_chunks())
    assert len(chunks) == 1

    data = chunks[0]

    # Verify all required columns were injected
    expected_cols = ["x", "y", "z", "w", "u", "classification", "confidence"]
    for col in expected_cols:
        assert col in data.dtype.names

    # Verify the values were applied correctly
    assert np.all(data["w"] == 3.5)
    assert np.all(data["u"] == 0.1)
    assert np.all(data["classification"] == 0)
    assert np.all(data["confidence"] == 1)
