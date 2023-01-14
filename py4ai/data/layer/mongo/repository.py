"""Module for Repository pattern for MongoDB persistence layers."""

from functools import cached_property
from typing import Any, Dict, Generic, Optional, Sequence

from motor.motor_tornado import MotorClientSession, MotorCollection
from pymongo import ReplaceOne
from pymongo.results import BulkWriteResult, DeleteResult, InsertOneResult

from py4ai.data.layer.common.criteria import SearchCriteria
from py4ai.data.layer.common.repository import (
    KD,
    KE,
    E,
    Paged,
    QueryOptions,
    Repository,
)
from py4ai.data.layer.common.serialiazer import DataSerializer
from py4ai.data.layer.mongo.criteria import MongoSearchCriteria


class MongoRepository(
    Repository[KE, KD, E, Dict[Any, Any], Dict[str, Any]], Generic[KE, KD, E]
):
    """Class implementing MongoDB repository."""

    def __init__(
        self,
        collection: MotorCollection,
        serializer: DataSerializer[KE, KD, E, Dict[Any, Any]],
        session: Optional[MotorClientSession] = None,
    ):
        """Return a MongoDB Repository Implementation.

         The current implementation uses the Motor async framework.

        :param collection: MongoDB collection
        :param serializer: Serializer to be used to serialize/deserialize MongoDB documents into domain objects
        :param session: MongoDB session to be used for unit-of-work operations
        """
        self.collection = collection
        self._serializer = serializer
        self.session = session

    @cached_property
    def serializer(self) -> DataSerializer[KE, KD, E, Dict[Any, Any]]:
        """Return the serializer.

        :return: DataSerializer for serializing/deserializing MongoDB documents into domain objects
        """
        return self._serializer

    async def retrieve(self, key: KE) -> Optional[E]:
        """Return an entry corresponding to a determined Entity key. If no match is found, returns None.

        :param key: Entity Key to be used for retrieving the entity.
        :returns: Entity associated with the provided key. If no match is found, None is returned.
        """
        try:
            raw_entity = await self.collection.find_one(
                {"_id": self.serializer.to_object_key(key)}
            )
            if raw_entity is None:
                self.logger.info(
                    f"Failed to fetch entity with key={str(key)} from repository {str(self)}"
                )
                return None
            return self.serializer.to_entity(raw_entity)
        except Exception as e:
            self.logger.error(f"Error occur in retrieving {key}: {e}")
            return None

    async def retrieve_by_criteria(
        self,
        criteria: SearchCriteria[Dict[str, Any]],
        options: QueryOptions = QueryOptions(),
    ) -> Paged[E]:
        """Return a list of entities, matching the query provided.

        :param criteria: query to be used for selecting items
        :param options: query options to be used when retrieving data
        :returns: Paged object for retrieved list of entities
        """
        number = await self.collection.count_documents(criteria.query)

        query = self.collection.find(criteria.query)
        cursor = (
            query
            if options.page_size < 0
            else query.limit(options.page_size).skip(
                options.page_start * options.page_size
            )
        )

        if len(options.sorting_options) > 0:
            cursor = cursor.sort(options.sorting_options)

        results = await cursor.to_list(length=None)  # noqa: E131

        has_more_pages = (
            False
            if options.page_size < 0
            else options.page_size + options.page_start * options.page_size < number
        )

        return Paged(
            number, [self.serializer.to_entity(doc) for doc in results], has_more_pages
        )

    async def list(self, options: QueryOptions = QueryOptions()) -> Paged[E]:
        """Return a full list of entities stored in the persistence layer.

        :param options: query options to be used when retrieving data
        :returns: Paged object for retrieved list of entities
        """
        return await self.retrieve_by_criteria(MongoSearchCriteria.empty(), options)

    async def create(self, entity: E) -> E:
        """Create the entity in the underlying persistence layer.

        :param entity: Entity to be created
        :returns: same entity provided as input, after creation. If creation fails, an error should be returned.
        """
        doc = self.serializer.to_object(entity)
        result: InsertOneResult = await self.collection.insert_one(
            doc, session=self.session
        )
        doc["_id"] = result.inserted_id
        return self.serializer.to_entity(doc)

    async def save(self, entities: Sequence[E]) -> Sequence[E]:
        """Create the entries in the persistence layer associated to a list of entities.

        :param entities: list of entities to be created.
        :returns: list of entities that have been successfully created in the persistence layer
        """
        updates = [
            (
                self.serializer.to_object_key(self.serializer.get_key(entity)),
                self.serializer.to_object(entity),
            )
            for entity in entities
        ]

        _: BulkWriteResult = await self.collection.bulk_write(
            [ReplaceOne({"_id": docId}, doc, upsert=True) for docId, doc in updates],
            ordered=False,
            session=self.session,
        )

        return entities

    async def delete(self, key: KE) -> bool:
        """Delete the entry in the persisence layer associated to the provided entity key.

        :param key: key identifying the entity.
        :returns: boolean value indicating whether the deletion has completed successfully.
        """
        result: DeleteResult = await self.collection.delete_one(
            {"_id": self.serializer.to_object_key(key)}, session=self.session
        )
        return True if (result.deleted_count > 0) else False

    async def delete_by_criteria(
        self, criteria: SearchCriteria[Dict[str, Any]]
    ) -> bool:
        """Delete all entries matching a given query.

        :param criteria: query to be used for deleting entries.
        :returns: boolean value indicating whether the deletion has completed successfully.
        """
        result: DeleteResult = await self.collection.delete_many(
            criteria.query, session=self.session
        )
        return True if (result.deleted_count > 0) else False
