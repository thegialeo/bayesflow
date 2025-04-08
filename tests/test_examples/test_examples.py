
import pytest

from tests.utils import run_notebook


@pytest.mark.slow
def test_two_moons_starter(examples_path):
    run_notebook(examples_path / "Two_Moons_Starter.ipynb")
