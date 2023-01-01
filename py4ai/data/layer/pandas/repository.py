"""Module with abstraction for accessing to data persistent in pickles, mimicking a ficticious database."""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional, Sequence, Tuple, TypeVar

import pandas as pd
from pandas.errors import EmptyDataError
from py4ai.core.types import KE, E, PathLike, Q

from py4ai.data.layer.common.criteria import SearchCriteria
from py4ai.data.layer.common.repository import (
    Paged,
    QueryOptions,
    Repository,
    SortingDirection,
)
from py4ai.data.layer.common.serialiazer import DataSerializer
from py4ai.data.layer.pandas.criteria import PandasFilter, PandasSearchCriteria

KD = TypeVar("KD", str, int, Tuple)


class PandasRepository(Repository[KE, KD, E, pd.Series, PandasFilter], ABC):
    """Archiver based on persistent layers based on tabular files, represented in memory by a pandas DataFrame."""

    @cached_property
    def serializer(self) -> DataSerializer[KE, KD, E, pd.Series]:
        """Return the serializer of the data repository.

        :returns: data serializer
        """
        return self._serializer

    @abstractmethod
    def _read(self) -> pd.DataFrame:
        """Read data from the file and return it as pandas Dataframe."""

    @abstractmethod
    def _write(self) -> None:
        """Write data (stored in a dataframe in memory) to the file."""

    def __init__(self, serializer: DataSerializer[KE, KD, E, pd.Series]) -> None:
        """
        Create an in-memory archiver based on structured data stored as a pandas DataFrame.

        :param serializer: An instance of :class:`serializer` that helps
            to retrieve/archive a pd.DataFrame row
        """
        self._serializer: DataSerializer[KE, KD, E, pd.Series] = serializer

    @property
    def data(self) -> pd.DataFrame:
        """
        Return tabular data stored in memory.

        :return: pd.DataFrame
        """
        try:
            return self._data
        except AttributeError:
            self._data: pd.DataFrame = self._read()
        return self.data

    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        """
        Set data property to given value.

        :param value: value to set
        """
        self._data = value

    def commit(self):
        """
        Persist data stored in memory in the file.

        :return: self
        """
        self._write()
        return self

    async def retrieve(self, key: KE) -> Optional[E]:
        """
        Retrieve row from a dataframe by id.

        :param key: row id
        :return: retrieved row parsed according to self.dao
        """
        document_key = self.serializer.to_object_key(key)
        try:
            return self.serializer.to_entity(self.data.loc[document_key])
        except KeyError:
            return None

    async def retrieve_by_criteria(
        self,
        criteria: SearchCriteria[PandasFilter],
        options: QueryOptions = QueryOptions(),
    ) -> Paged[E]:
        """
        Retrieve rows satisfying condition, sorted according to given ordering.

        :param criteria: condition to satisfy. If None returns all rows.
        :param options: ordering to respect. If None, no ordering is given.
        :return: iterator of (ordered) rows satisfying given condition
        """
        rows = self.data[criteria.query(self.data)]

        if len(self.data) == 0:
            return Paged(0, [], False)

        rows = (
            rows.sort_values(
                by=[sort[0] for sort in options.sorting_options],
                ascending=[
                    True if sort[1] == SortingDirection.ASC else False
                    for sort in options.sorting_options
                ],
            )
            if len(options.sorting_options) > 0
            else rows
        )

        size = rows.shape[0]
        end_page = (
            (options.page_start + options.page_size)
            if options.page_size != -1
            else None
        )
        elements = [
            self.serializer.to_entity(row)
            for row_id, row in rows.iloc[options.page_start : end_page].iterrows()
        ]

        return Paged(size, elements, options.page_start + options.page_size < size)

    async def create(self, obj: E) -> E:
        """
        Insert an object of type Document/pd.DataFrame/pd.Series in a pd.DataFrame.

        :param obj: An instance of :class:`cgnal.data.model.text.Document, pd.DataFrame or pd.Series`
        :return: self i.e. an instance of ``PandasArchiver`` with updated self.data object
        """
        return (await self.save([obj]))[0]

    def __create_series_from_entity(self, entity: E) -> pd.Series:
        serie = self.serializer.to_object(entity)
        serie.name = self.serializer.to_object_key(self.serializer.get_key(entity))
        return serie

    async def save(self, entities: Sequence[E]) -> Sequence[E]:
        """Insert many objects of type Document/pd.DataFrame/pd.Series in a pd.DataFrame.

        :param entities: List of objects to be inserted.
        :returns: the entities inserted in the persistence layer.

        """
        new = pd.DataFrame(
            [self.__create_series_from_entity(entity) for entity in entities]
        )
        self.data = pd.concat(
            [self.data.loc[list(set(self.data.index).difference(new.index))], new]
        ).sort_index()
        self.commit()
        return entities

    async def list(self, options: QueryOptions = QueryOptions()) -> Paged[E]:
        """Return a full list of entities stored in the persistence layer.

        :param options: query options to be used when retrieving data
        :returns: Paged object for retrieved list of entities
        """
        return await self.retrieve_by_criteria(PandasSearchCriteria.empty(), options)

    async def delete(self, key: KE) -> bool:
        """Delete the entry in the persisence layer associated to the provided entity key.

        :param key: key identifying the entity.
        :returns: boolean value indicating whether the deletion has completed successfully.
        """
        document_key = self.serializer.to_object_key(key)
        self.data = self.data.drop(document_key)
        self.commit()
        return True

    async def delete_by_criteria(self, criteria: SearchCriteria[Q]) -> bool:
        """Delete all entries matching a given query.

        :param criteria: query to be used for deleting entries.
        :returns: boolean value indicating whether the deletion has completed successfully.
        """
        rows = self.data[criteria.query(self.data)]
        self.data = self.data.drop(rows.index)
        self.commit()
        return True


class CsvRepository(PandasRepository[KE, KD, E]):
    """Repository to be used with tabular files stored as csv files."""

    def __init__(
        self,
        filename: PathLike,
        serializer: DataSerializer[KE, KD, E, pd.Series],
        sep: str = ";",
    ) -> None:
        """
        Create an in-memory archiver based on structured data stored in the filesystem as a CSV.

        :param filename: str, path object or file like object. Any valid string path to a csv file.
        :param serializer: An instance of serializer to convert between raw and domain objects.
        :param sep: str, default ';'. Delimiter to use
        """
        super(CsvRepository, self).__init__(serializer)

        self.filename = filename
        self.sep = sep

    def _write(self) -> None:
        """Write object to a csv file."""
        self.data.to_csv(self.filename, sep=self.sep)

    def _read(self) -> pd.DataFrame:
        """
        Read csv file into a pandas DataFrame.

        :return: pandas Dataframe
        """
        try:
            output = pd.read_csv(self.filename, sep=self.sep, index_col=0)
        except (FileNotFoundError, EmptyDataError):
            output = pd.DataFrame()
        return output
