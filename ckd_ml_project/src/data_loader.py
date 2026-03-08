"""
data_loader.py
==============
Utilities for loading CKD datasets in ARFF (OpenML) format.

Supports both file paths (str / pathlib.Path) and file-like objects such as
Streamlit's UploadedFile, making it usable both from the CLI pipeline and the
web interface.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from scipy.io import arff

from src.config import DATA_DIR, get_logger

logger = get_logger(__name__)


def _strip_data_whitespace(content: str) -> str:
    """Normalise every data line in the @DATA section before parsing.

    Some ARFF files exported from OpenML contain embedded tab characters or
    single-quote wrappers (e.g. ``'\\tno'``, ``' yes'``) and values that span
    two physical lines.  scipy's ARFF parser does strict nominal-value
    matching and cannot handle any of these forms.

    This preprocessor:
    1. Joins lines that were wrapped mid-value (a ``'`` at end of a line
       followed by continuation on the next).
    2. Removes single-quote characters and embedded tab characters from
       each field.
    3. Strips leading/trailing whitespace from each field.
    """
    lines = content.splitlines()
    in_data = False
    result = []
    buf: str | None = None  # accumulates a wrapped data row
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "@data":
            in_data = True
            result.append(line)
            continue
        if not in_data:
            result.append(line)
            continue
        if not stripped or stripped.startswith("%"):
            result.append(line)
            continue
        # Accumulate wrapped lines (a data value split across lines)
        if buf is not None:
            buf = buf + " " + stripped
        else:
            buf = stripped
        # A complete data row has a fixed number of commas; keep joining until
        # single-quote count is even (all opened quotes are closed).
        if buf.count("'") % 2 != 0:
            continue  # still inside a quoted value that spans lines
        # Clean up each field: strip single-quotes, literal \t escape sequences,
        # actual tab characters, and surrounding whitespace
        fields = [
            f.replace("'", "").replace("\\t", "").replace("\t", "").strip()
            for f in buf.split(",")
        ]
        result.append(",".join(fields))
        buf = None
    if buf is not None:
        # Flush any remaining partial row
        fields = [
            f.replace("'", "").replace("\\t", "").replace("\t", "").strip()
            for f in buf.split(",")
        ]
        result.append(",".join(fields))
    return "\n".join(result)


def load_arff(filepath: Union[str, Path, io.BytesIO]) -> pd.DataFrame:
    """
    Load an ARFF dataset and convert it to a pandas DataFrame.

    Byte-string columns produced by the scipy ARFF parser are automatically
    decoded to plain Python strings, and leading/trailing whitespace is
    stripped.

    Parameters
    ----------
    filepath : str | Path | file-like object
        Path to a ``.arff`` file on disk, or a file-like object (e.g. a
        Streamlit ``UploadedFile`` or ``io.BytesIO``).

    Returns
    -------
    pd.DataFrame
        Raw dataset with all original columns intact.

    Raises
    ------
    FileNotFoundError
        If *filepath* is a path string/Path that does not exist.
    Exception
        Propagates any parsing errors from the scipy ARFF parser.
    """
    logger.info("Loading ARFF dataset from: %s", filepath)
    try:
        if isinstance(filepath, (str, Path)):
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"ARFF file not found: {path}")
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        else:
            # File-like object (Streamlit UploadedFile / BytesIO)
            content = filepath.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")

        # Strip whitespace around comma-separated values in data lines so that
        # nominal attributes like ' yes' match their declared values ('yes').
        content = _strip_data_whitespace(content)
        data, _meta = arff.loadarff(io.StringIO(content))

        df = pd.DataFrame(data)

        # Decode byte strings emitted by the scipy ARFF parser
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8").strip() if isinstance(x, bytes) else x
            )

        logger.info("Dataset loaded successfully — shape: %s", df.shape)
        return df

    except Exception as exc:
        logger.error("Failed to load ARFF file: %s", exc)
        raise


def load_sample_data() -> pd.DataFrame:
    """
    Load the first ``.arff`` file found in the project ``data/`` directory.

    This is a convenience helper for running the pipeline without explicitly
    specifying a file path.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        When ``data/`` contains no ``.arff`` files.
    """
    arff_files = sorted(DATA_DIR.glob("*.arff"))
    if not arff_files:
        raise FileNotFoundError(
            f"No .arff files found in '{DATA_DIR}'.\n"
            "Please place your CKD dataset (e.g. chronic_kidney_disease.arff) "
            "in the data/ directory, or pass the path explicitly via --data."
        )
    logger.info("Found ARFF file: %s", arff_files[0])
    return load_arff(arff_files[0])
