import os
import torch

_HERE = os.path.realpath(os.path.dirname(__file__))


def _lib_path():
    path = os.path.join(os.path.join(_HERE, "lib"), "libcusss.so")

    if os.path.isfile(path):
        return path

    raise ImportError("Could not find libcusss shared library at " + path)

# Load the C++ operators
torch.classes.load_library(_lib_path())

# Import python wrapper
from .glue import SSS

# Expose the main classes
__all__ = [
    "SSS",
]
