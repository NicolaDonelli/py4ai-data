"""Module with abstraction for accessing to data persisted in files and represented by TabularData."""

import os
import pickle
from io import BytesIO, StringIO
from pathlib import Path
from typing import Type

import pandas as pd
from pydantic import BaseModel

from py4ai.data.layer.fs.repository import FileSystemRepository
from py4ai.data.layer.fs.serializer import FileSerializer, FileSerializerMode, IndexedIO


class TabularData(BaseModel):
    """Domain object to represent tabular data."""

    name: str = ""
    data: pd.DataFrame

    class Config:
        """Specs for the pydantic model."""

        arbitrary_types_allowed = True

    def update(self, other: "TabularData") -> "TabularData":
        """Return TabularData object by concatenating two TabularData objects.

        :param other: second TabularData
        :returns: merged TabularData
        """
        return TabularData(
            name=f"{self.name}/{other.name}" if other.name != self.name else self.name,
            data=pd.concat([self.data, other.data], axis=0),
        )


class CsvSerializer(FileSerializer[str, TabularData]):
    """DataSerializer to be used for serializing/deserializing CSV files."""

    mode = FileSerializerMode.TEXT

    def __init__(self, path: Path, encoding: str = "utf-8", sep: str = ";"):
        """Return instance of DataSerializer.

        :param path: local folder to be used to construct filenames
        :param encoding: type of IO serialization (text, binary) to be used when writing files
        :param sep: separator used in the csv
        """
        super().__init__(path, encoding)
        self.sep = sep

    def get_key(self, entity: TabularData) -> str:
        """Extract key for given entity.

        :param entity: provided TabularData
        :returns: entity key
        """
        return entity.name

    def to_object_key(self, key: str) -> str:
        """Transform entity key into raw key, to be used for indexing in the persistence layer.

        :param key: entity key
        :returns: raw key
        """
        return os.path.join(self.path, key + ".csv")

    def to_entity(self, document: IndexedIO[str]) -> TabularData:
        """Deserialize raw content into domain object entity.

        :param document: raw content
        :returns: domain object entity
        """
        data = pd.read_csv(document.buffer, sep=self.sep, dtype=str)
        return TabularData(name=document.name, data=data.set_index(data.columns[0]))

    def to_object(self, entity: TabularData) -> IndexedIO[str]:
        """Serialize domain object entity into raw content.

        :param entity: domain object entity
        :returns: raw content
        """
        csv_buffer = StringIO()
        entity.data.to_csv(csv_buffer, sep=self.sep)
        csv_buffer.seek(0)
        return IndexedIO(name=self.get_key(entity), buffer=csv_buffer)


class PickleSerializer(CsvSerializer):
    """DataSerializer to be used for serializing/deserializing pickle files."""

    mode = FileSerializerMode.BINARY

    def to_object_key(self, key: str) -> str:
        """Transform entity key into raw key, to be used for indexing in the persistence layer.

        :param key: entity key
        :returns: raw key
        """
        return os.path.join(self.path, key + ".pkl")

    def to_entity(self, document: IndexedIO[str]) -> TabularData:
        """Deserialize raw content into domain object entity.

        :param document: raw content
        :returns: domain object entity
        """
        return TabularData(name=document.name, data=pd.read_pickle(document.buffer))

    def to_object(self, entity: TabularData) -> IndexedIO[str]:
        """Serialize domain object entity into raw content.

        :param entity: domain object entity
        :returns: raw content
        """
        buffer = BytesIO()
        buffer.write(pickle.dumps(entity.data, protocol=pickle.HIGHEST_PROTOCOL))
        buffer.seek(0)
        return IndexedIO(name=self.get_key(entity), buffer=buffer)


class LocalDatabase(FileSystemRepository[str, TabularData]):
    """Archiver used for persistent layers used to store tabular data files."""

    def __init__(
        self,
        path: Path,
        serializer: Type[FileSerializer[str, TabularData]] = CsvSerializer,
    ):
        """Return an instance of the class.

        :param path: path where to store the files.
        :param serializer: An instance of serializer to convert between raw and domain objects.
        """
        super(LocalDatabase, self).__init__(path, serializer(path))
