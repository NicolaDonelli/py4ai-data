"""Helpers functions for handling file systems paths."""

import os
from pathlib import Path
from typing import List, Union


def complete_path_split(path: str) -> List[str]:
    """Split path into single directory names.

    :param path: path as string
    :returns: list of directory names.
    """
    p, end = os.path.split(path)
    return complete_path_split(p) + [end] if p != "" else [end]


def create_dir_if_not_exists(directory: Union[str, Path]) -> Union[str, Path]:
    """Create the directory if it does not exist.

    :param directory: directory, either provided as string or as path
    :returns: directory pash, as a string or as path depending on the input type.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory
