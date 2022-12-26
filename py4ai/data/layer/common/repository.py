"""Module containing the abstractions and implementations for repository classes."""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Generic, List, Optional, Sequence, Tuple

from py4ai.core.logging import WithLogging
from py4ai.core.types import KD, KE, D, E, Q

from py4ai.data.layer.common.criteria import SearchCriteria
from py4ai.data.layer.common.serialiazer import DataSerializer


class Paged(Generic[E], WithLogging):
    """Class representing a Paged query result."""

    items: List[E]
    size: int
    more_pages: bool

    def __init__(self, size: int, items: List[E], more_pages: bool):
        """Instantiate a paged list of elements class.

        :param size: number of results
        :param items: list of returned objects
        :param more_pages: flag to notify whether there are more pages or not
        """
        self.items = items
        self.size = size
        self.more_pages = more_pages


class SortingDirection(IntEnum):
    """Enum class representing the direction for sorting results in queries."""

    DES = -1
    ASC = 1


class QueryOptions(WithLogging):
    """Class for providing query options."""

    def __init__(
        self,
        page_start: int = 0,
        page_size: int = -1,
        sorting_options: List[Tuple[str, SortingDirection]] = [],
    ):
        """
        Implement the options to be used in a query call in the repository abstraction.

        :param page_start: integer setting the current page
        :param page_size: integer setting the size of paging (default value is -1 - all result are returned in one page)
        :param sorting_options: a list of options for ordering results
        """
        self.page_start: int = page_start
        self.page_size: int = page_size
        self.sorting_options: List[Tuple[str, SortingDirection]] = sorting_options

    def copy(
        self,
        page_start: Optional[int] = 0,
        page_size: Optional[int] = -1,
        sorting_options: Optional[List[Tuple[str, SortingDirection]]] = None,
    ) -> "QueryOptions":
        """Copy the object, overriding provided properties.

        :param page_start: integer setting the current page
        :param page_size: integer setting the size of paging (default value is -1 - all result are returned in one page)
        :param sorting_options: a list of options for ordering results
        :returns: new object, with overridden properties.
        """
        return QueryOptions(
            page_start if page_start is not None else self.page_start,
            page_size if page_size is not None else self.page_size,
            sorting_options if sorting_options is not None else self.sorting_options,
        )


class Repository(Generic[KE, KD, E, D, Q], WithLogging, ABC):
    """Abstract class representing the base Repository."""

    @property
    @abstractmethod
    def serializer(self) -> DataSerializer[KE, KD, E, D]:
        """Return the data serializer used in the repository."""
        ...

    @abstractmethod
    async def create(self, entity: E) -> E:
        """Create the entity in the underlying persistence layer.

        :param entity: Entity to be created
        :returns: same entity provided as input, after creation. If creation fails, an error should be returned.
        """
        ...

    async def retrieve(self, key: KE) -> Optional[E]:
        """Return an entry corresponding to a determined Entity key. If no match is found, returns None.

        :param key: Entity Key to be used for retrieving the entity.
        """
        ...

    @abstractmethod
    async def retrieve_by_criteria(
        self, criteria: SearchCriteria[Q], options: QueryOptions = QueryOptions()
    ) -> Paged[E]:
        """Return a list of entities, matching the query provided.

        :param criteria: query to be used for selecting items
        :param options: query options to be used when retrieving data
        :returns: Paged object for retrieved list of entities
        """
        ...

    @abstractmethod
    async def list(self, options: QueryOptions = QueryOptions()) -> Paged[E]:
        """Return a full list of entities stored in the persistence layer.

        :param options: query options to be used when retrieving data
        :returns: Paged object for retrieved list of entities
        """
        ...

    @abstractmethod
    async def save(self, entities: Sequence[E]) -> Sequence[E]:
        """Create the entries in the persistence layer associated to a list of entities.

        :param entities: list of entities to be created.
        :returns: list of entities that have been successfully created in the persistence layer
        """
        ...

    @abstractmethod
    async def delete(self, key: KE) -> bool:
        """Delete the entry in the persistence layer associated to the provided entity key.

        :param key: key identifying the entity.
        :returns: boolean value indicating whether the deletion has completed successfully.
        """
        ...

    @abstractmethod
    async def delete_by_criteria(self, criteria: SearchCriteria[Q]) -> bool:
        """Delete all entries matching a given query.

        :param criteria: query to be used for deleting entries.
        :returns: boolean value indicating whether the deletion has completed successfully.
        """
        ...
