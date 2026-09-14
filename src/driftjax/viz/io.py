"""Lightweight JSON result logging for driftjax example scripts."""

import inspect
import json
import os

import numpy as np


def _json_default(o):
    # numpy / jax scalars and arrays -> plain Python types
    try:
        return np.asarray(o).tolist()
    except Exception:
        return str(o)


def save_results(results, name=None, directory=None):
    """Persist *results* to ``<name>_results.json`` next to the calling script.

    Call with no extra arguments from an example script; the output filename is
    derived from the calling ``.py`` file (e.g. ``ex1_np_junction_results.json``).
    ``results`` may contain numpy/jax arrays and scalars, which are converted to
    native Python types automatically.  Returns the written path.
    """
    caller = inspect.stack()[1].filename
    if directory is None:
        directory = os.path.dirname(os.path.abspath(caller))
    if name is None:
        name = os.path.splitext(os.path.basename(caller))[0]
    path = os.path.join(directory, f"{name}_results.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, default=_json_default)
    return path
