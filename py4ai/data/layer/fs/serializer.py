"""Module for implementation of serializer objects for FileSystem persistence layers."""

from abc import ABC
from enum import Enum
from io import IOBase
from pathlib import Path
from typing import Generic

from py4ai.core.types import KE, E
from pydantic import BaseModel

from py4ai.data.layer.common.serialiazer import DataSerializer


class IndexedIO(BaseModel, Generic[KE]):
    """Domain object to represent data read from FileSystem."""

    name: KE
    buffer: IOBase

    class Config:
        """Specs for the pydantic model."""

        arbitrary_types_allowed = True


class FileSerializerMode(str, Enum):
    """Enum representing the type of reading/writing mode for files."""

    TEXT = ""
    BINARY = "b"


class FileSerializer(Generic[KE, E], DataSerializer[KE, str, E, IndexedIO], ABC):
    """DataSerializer to read raw data and convert it into IndexedIO."""

    mode: FileSerializerMode = FileSerializerMode.TEXT

    def __init__(self, path: Path, encoding: str = "utf-8"):
        """Return instance of DataSerializer.

        :param path: local folder to be used to construct filenames
        :param encoding: type of IO serialization (text, binary) to be used when writing files
        """
        self.path = path
        self.encoding = encoding if self.mode is FileSerializerMode.TEXT else None

    @classmethod
    def with_path(cls, path: Path) -> "FileSerializer":
        """Return an instance of the class, pointing to a given path.

        :param path: new path to be used for the serializer.
        :returns: FileSerializer instance with path set to the input path.
        """
        return cls(path)
