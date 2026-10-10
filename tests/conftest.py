import pytest

from tests.helpers import TEST_URL, Env


@pytest.fixture
def env(tmp_path):
    if not TEST_URL:
        pytest.skip("Defina TEST_DATABASE_URL (PostgreSQL) para las pruebas de integración")
    e = Env(tmp_path)
    yield e
    e.close()


@pytest.fixture(scope="session")
def reference_season(tmp_path_factory):
    """Una temporada completa (semilla 42) jugada de punta a punta; se reutiliza en varias pruebas."""
    if not TEST_URL:
        pytest.skip("Defina TEST_DATABASE_URL (PostgreSQL) para las pruebas de integración")
    e = Env(tmp_path_factory.mktemp("ref"))
    assert e.trainer().run() == 0
    e.sig = e.signature()
    yield e
    e.close()
