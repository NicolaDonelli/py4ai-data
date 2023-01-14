"""Module for implementing the criteria to be used for in-memory Pandas persistence layers."""

from typing import Callable

import pandas as pd
from py4ai.core.utils.decorators import same_type

from py4ai.data.layer.common.criteria import SearchCriteria

PandasFilter = Callable[[pd.DataFrame], pd.Series]


class PandasSearchCriteria(SearchCriteria[PandasFilter]):
    """SearchCriteria to be used for in-memory Pandas persistence layers."""

    def __init__(self, condition: PandasFilter):
        """Instantiate a Pandas SearchCriteria, using the provided condition.

        :param condition: underlying condition representing the Pandas query.
        """
        self.condition = condition

    @property
    def query(self) -> PandasFilter:
        """Return the underlying query based on a PandasFilter condition.

        :returns: underlying query
        """
        return self.condition

    @same_type
    def __or__(self, other: SearchCriteria[PandasFilter]) -> "PandasSearchCriteria":
        """Return query resulting from OR operation between queries.

        :param other: the other query to be used in the OR operation
        :returns: resulting query
        """
        return PandasSearchCriteria(lambda df: self.query(df) | other.query(df))

    @same_type
    def __and__(self, other: SearchCriteria[PandasFilter]) -> "PandasSearchCriteria":
        """Return query resulting from AND operation between queries.

        :param other: the other query to be used in the AND operation
        :returns: resulting query
        """
        return PandasSearchCriteria(lambda df: self.query(df) & other.query(df))

    @staticmethod
    def empty() -> "PandasSearchCriteria":
        """Return an empty query, i.e. a query including all items.

        :returns: empty query
        """
        return PandasSearchCriteria(lambda df: df.apply(lambda x: True, axis=1))
