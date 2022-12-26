"""Module containing basic classes and abstractions for creating persistence layers query."""

from abc import ABC, abstractmethod
from typing import Generic

from py4ai.core.types import Q


class BaseCriteria(Generic[Q], ABC):
    """Basic query class."""

    @property
    @abstractmethod
    def query(self) -> Q:
        """Return the underlying query."""
        ...


class SearchCriteria(BaseCriteria, Generic[Q]):
    """Base query extended with logical operations."""

    @abstractmethod
    def __or__(self, other: "SearchCriteria[Q]") -> "SearchCriteria[Q]":
        """Return query resulting from OR operation between queries.

        :param other: the other query to be used in the OR operation
        :returns: resulting query
        """
        ...

    @abstractmethod
    def __and__(self, other: "SearchCriteria[Q]") -> "SearchCriteria[Q]":
        """Return query resulting from AND operation between queries.

        :param other: the other query to be used in the AND operation
        :returns: resulting query
        """
        ...
