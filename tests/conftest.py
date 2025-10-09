import pytest
import matplotlib


@pytest.fixture(autouse=True, scope="session")
def set_matplotlib_backend():
    matplotlib.use("Agg")
