"""Module for SQL Alchemy query implementations."""

from typing import Optional, Union

from py4ai.core.utils.decorators import same_type
from sqlalchemy import and_, or_
from sqlalchemy.sql.elements import BinaryExpression, BooleanClauseList

from py4ai.data.layer.common.criteria import SearchCriteria

SqlAlchemyQuery = Union[BinaryExpression, BooleanClauseList, bool]


class SqlAlchemySearchCriteria(SearchCriteria[SqlAlchemyQuery]):
    """Base Query Implementation for SQL Alchemy."""

    def __init__(self, query: Optional[SqlAlchemyQuery] = None):
        """Return instance of SQL Alchemy Query.

        :param query: SqlAlchemyQuery using either binary expressions or boolean clause lists
        """
        self.__query__ = query

    @property
    def query(self) -> BinaryExpression:
        """Return the underlying query as SQL Alchemy object.

        :returns: underlying query as SQL Alchemy object
        """
        return self.__query__ if self.__query__ is not None else 1 == 1

    @same_type
    def __or__(self, other: "SqlAlchemySearchCriteria") -> "SqlAlchemySearchCriteria":
        """Return query resulting from OR operation between queries.

        :param other: the other query to be used in the OR operation
        :returns: resulting query
        """
        return SqlAlchemySearchCriteria(or_(self.query, other.query))

    @same_type
    def __and__(self, other: "SqlAlchemySearchCriteria") -> "SqlAlchemySearchCriteria":
        """Return query resulting from AND operation between queries.

        :param other: the other query to be used in the AND operation
        :returns: resulting query
        """
        return SqlAlchemySearchCriteria(and_(self.query, other.query))

    @staticmethod
    def empty() -> "SqlAlchemySearchCriteria":
        """Return an empty query, i.e. a query including all items.

        :returns: empty query
        """
        return SqlAlchemySearchCriteria()
