import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run_notebook(path):
    with open(str(path)) as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)

    kernel = ExecutePreprocessor(timeout=600, kernel_name="python3")

    return kernel.preprocess(nb)
