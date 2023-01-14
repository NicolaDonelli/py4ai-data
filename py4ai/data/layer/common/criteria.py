"""Module containing basic classes and abstractions for creating persistence layers query."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from py4ai.core.logging import WithLogging
from py4ai.core.types import Q

TSearchCriteria = TypeVar("TSearchCriteria", bound="SearchCriteria")  # type: ignore


class BaseCriteria(WithLogging, Generic[Q], ABC):
    """Basic query class."""

    @property
    @abstractmethod
    def query(self) -> Q:
        """Return the underlying query."""
        ...


class SearchCriteria(BaseCriteria[Q], Generic[Q]):
    """Base query extended with logical operations."""

    @abstractmethod
    def __or__(self: TSearchCriteria, other: TSearchCriteria) -> TSearchCriteria:
        """Return query resulting from OR operation between queries.

        :param other: the other query to be used in the OR operation
        :returns: resulting query
        """
        ...

    @abstractmethod
    def __and__(self: TSearchCriteria, other: TSearchCriteria) -> TSearchCriteria:
        """Return query resulting from AND operation between queries.

        :param other: the other query to be used in the AND operation
        :returns: resulting query
        """
        ...
