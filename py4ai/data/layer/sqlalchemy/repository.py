"""Module for SQL Alchemy repository implementations."""

from typing import Any, Dict, Generic, Optional, Sequence, Union

from py4ai.core.logging import WithLogging
from sqlalchemy import func, select
from sqlalchemy.exc import MultipleResultsFound, NoResultFound
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.sql.elements import BinaryExpression, BooleanClauseList

from py4ai.data.layer.common.criteria import SearchCriteria
from py4ai.data.layer.common.repository import (
    KD,
    KE,
    E,
    Paged,
    QueryOptions,
    Repository,
    SortingDirection,
)
from py4ai.data.layer.common.serialiazer import DataSerializer
from py4ai.data.layer.sqlalchemy.criteria import SqlAlchemySearchCriteria
from py4ai.data.layer.sqlalchemy.serializer import SqlAlchemySerializer


class SqlAlchemyRepository(
    Repository[KE, KD, E, Dict[str, Any], Union[BinaryExpression, BooleanClauseList]],
    WithLogging,
    Generic[KE, KD, E],
):
    """Repository implementation for SQL Alchemy persistence layers."""

    def __init__(
        self, engine: AsyncEngine, serializer: SqlAlchemySerializer[KE, KD, E]
    ):
        """Return a instance of the SQL Alchemy Repository.

        :param engine: SQL Alchemy Async Engine to be used for db connections
        :param serializer: Data serializer for serializing/deserializing raw data
        """
        self.engine = engine
        self._serializer = serializer
        self.table = serializer.table

    @property
    def serializer(self) -> DataSerializer[KE, KD, E, Dict[str, Any]]:
        """Return the data serializer used in the repository.

        :returns: data serializer
        """
        return self._serializer

    async def retrieve(self, key: KE) -> Optional[E]:
        """Return an entry corresponding to a determined Entity key. If no match is found, returns None.

        :param key: Entity Key to be used for retrieving the entity.
        :returns: Entity associated with the provided key. If no match is found, None is returned.
        """
        try:
            async with self.engine.begin() as connection:
                condition = self.table.c.id == self.serializer.to_object_key(key)
                result = await connection.execute(self.table.select().where(condition))
                raw_entity = result.one()
                return self.serializer.to_entity(raw_entity)
        except NoResultFound as e:
            self.logger.info(
                f"Failed to fetch entity with key={str(key)} from repository {str(self)}: {e}"
            )
            return None
        except MultipleResultsFound as e:
            self.logger.info(
                f"Multiple fetch entity with key={str(key)} from repository {str(self)}: {e}"
            )
            return None
        except Exception as e:
            self.logger.error(f"Error occur in retrieving {key}: {e}")
            return None

    async def retrieve_by_criteria(
        self,
        criteria: SearchCriteria[Union[BinaryExpression, BooleanClauseList]],
        options: QueryOptions = QueryOptions(),
    ) -> Paged[E]:
        """Return a list of entities, matching the query provided.

        :param criteria: query to be used for selecting items
        :param options: query options to be used when retrieving data
        :returns: Paged object for retrieved list of entities
        """
        async with self.engine.begin() as connection:
            number_results = await connection.execute(
                select([func.count()]).select_from(self.table).where(criteria.query)
            )
            number = number_results.first()[0]

            query = self.table.select().where(criteria.query)
            query = (
                query
                if options.page_size < 0
                else query.limit(options.page_size).offset(
                    options.page_start * options.page_size
                )
            )

            for column_name, order in options.sorting_options:
                query = query.order_by(
                    self.table.c[column_name]
                    if order == SortingDirection.ASC
                    else self.table.c[column_name].desc()
                )

            cursor = await connection.execute(query)

            results = cursor.fetchall()  # noqa: E131
            has_more_pages = (
                False
                if options.page_size < 0
                else options.page_size + options.page_start * options.page_size < number
            )

            return Paged(
                number,
                [self.serializer.to_entity(doc) for doc in results],
                has_more_pages,
            )

    async def create(self, entity: E) -> E:
        """Create the entity in the underlying persistence layer.

        :param entity: Entity to be created
        :returns: same entity provided as input, after creation. If creation fails, an error should be returned.
        :raises ValueError: if the object is not inserted correctly
        """
        async with self.engine.begin() as connection:
            doc = self.serializer.to_object(entity)
            result = await connection.execute(self.table.insert(doc))
            if result.rowcount > 0:
                return self.serializer.to_entity(doc)
            else:
                raise ValueError

    async def list(self, options: QueryOptions = QueryOptions()) -> Paged[E]:
        """Return a full list of entities stored in the persistence layer.

        :param options: query options to be used when retrieving data
        :returns: Paged object for retrieved list of entities
        """
        return await self.retrieve_by_criteria(SqlAlchemySearchCriteria(True), options)

    async def save(self, entities: Sequence[E]) -> Sequence[E]:
        """Create the entries in the persistence layer associated to a list of entities.

        :param entities: list of entities to be created.
        :returns: list of entities that have been successfully created in the persistence layer
        :raises ValueError:  if any object is not inserted correctly
        """
        async with self.engine.begin() as connection:
            docs = [self.serializer.to_object(e) for e in entities]
            result = await connection.execute(self.table.insert(), docs)
            if result.rowcount == len(entities):
                return entities
            else:
                raise ValueError

    async def delete(self, key: KE) -> bool:
        """Delete the entry in the persisence layer associated to the provided entity key.

        :param key: key identifying the entity.
        :returns: boolean value indicating whether the deletion has completed successfully.
        """
        async with self.engine.begin() as connection:
            condition = self.table.c.id == self.serializer.to_object_key(key)
            result = await connection.execute(self.table.delete(condition))
            return bool(result.rowcount > 0)

    async def delete_by_criteria(
        self, criteria: SearchCriteria[Union[BinaryExpression, BooleanClauseList]]
    ) -> bool:
        """Delete all entries matching a given query.

        :param criteria: query to be used for deleting entries.
        :returns: boolean value indicating whether the deletion has completed successfully.
        """
        async with self.engine.begin() as connection:
            result = await connection.execute(self.table.delete(criteria.query))
            return bool(result.rowcount > 0)
