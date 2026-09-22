import logging
import pytest


@pytest.fixture
def cap_globato(caplog):
    """Fixture to capture logs from the 'globato' logger despite propagate=False."""
    logger = logging.getLogger("globato")

    # Save original propagation state
    original_propagate = logger.propagate

    # Enable propagation so caplog can intercept the logs
    logger.propagate = True

    yield caplog

    # Restore original state after the test completes
    logger.propagate = original_propagate
