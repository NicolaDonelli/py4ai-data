"""Module for Criteria abstraction for MongoDB persistence layer."""

from typing import Any, Dict, Optional

from py4ai.core.utils.decorators import same_type

from py4ai.data.layer.common.criteria import SearchCriteria


class MongoSearchCriteria(SearchCriteria[Dict[str, Any]]):
    """General Criteria to be used in MongoDB repositories."""

    def __init__(self, query: Optional[Dict[str, Any]] = None):
        """Instantiate a new MongoDB Criteria object.

        :param query: Optional[Dict] representing the MongoDB query
        """
        self.__query__ = query

    @property
    def query(self) -> Dict[str, Any]:
        """Return the query.

        :return: Dict with the MongoDB query
        """
        return self.__query__ if self.__query__ is not None else dict()

    @same_type
    def __or__(self, other: SearchCriteria) -> "MongoSearchCriteria":
        """Return a new Criteria resulting from the OR condition between two queries.

        :param other: MongoSearchCriteria to be used in the OR condition
        :return: MongoSearchCriteria resulting from operation.
        """
        queries = [self.query, other.query]
        return MongoSearchCriteria({"$or": queries})

    @same_type
    def __and__(self, other: SearchCriteria) -> "MongoSearchCriteria":
        """Return a new Criteria resulting from the AND condition between two queries.

        :param other: SearchCriteria to be used in the AND condition
        :return: MongoSearchCriteria resulting from operation.
        """
        queries = [self.query, other.query]
        return MongoSearchCriteria({"$and": queries})

    @staticmethod
    def empty() -> "MongoSearchCriteria":
        """Return an empty query, i.e. that retrieve all records.

        :return: MongoSearchCriteria with empty query
        """
        return MongoSearchCriteria()
