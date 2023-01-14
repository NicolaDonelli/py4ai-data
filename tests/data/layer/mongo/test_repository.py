import os
from typing import Any, Dict, Optional, Union

import pandas as pd
from bson import ObjectId
from motor.motor_tornado import MotorClientSession, MotorCollection
from py4ai.core.tests.core import TestCase
from py4ai.core.utils.executors import AsyncExecutor

from py4ai.data.layer.common.repository import QueryOptions, SortingDirection
from py4ai.data.layer.mongo.criteria import MongoSearchCriteria
from py4ai.data.layer.mongo.repository import MongoRepository
from py4ai.data.layer.mongo.serializer import MongoModel, create_mongo_id
from tests import DATA_FOLDER
from tests.data.layer.base import (
    CriteriaFactory,
    Entity,
    EntityDataSerializer,
    EntityRepository,
)
from tests.data.layer.mongo import init_mongo


class EntityMongoORM(Entity, metaclass=MongoModel):
    pass


class MongoCriteriaFactory(CriteriaFactory[Dict[str, Any]]):
    model = EntityMongoORM

    def by_cai(self, cai: int) -> MongoSearchCriteria:
        return MongoSearchCriteria({str(self.model.cai): cai})

    def from_birth_year(self, birth_year: int) -> MongoSearchCriteria:
        return MongoSearchCriteria({str(self.model.birth_year): {"$gte": birth_year}})

    def by_birth_year(self, birth_year: int) -> MongoSearchCriteria:
        return MongoSearchCriteria({str(self.model.birth_year): birth_year})


class MongoEntitySerializer(
    EntityDataSerializer[ObjectId, Dict[str, Union[int, ObjectId]]]
):
    def to_object(self, entity: Entity) -> Dict[str, Union[int, ObjectId]]:
        doc = entity.dict()
        doc["_id"] = self.to_object_key(self.get_key(entity))
        return doc

    def to_entity(self, document: Dict[str, Union[int, ObjectId]]) -> Entity:
        return Entity(**document)

    def to_object_key(self, key: int) -> ObjectId:
        return create_mongo_id(str(key))


class MongoEntityRepository(
    MongoRepository[int, ObjectId, Entity],
    EntityRepository[ObjectId, Dict[str, Union[int, ObjectId]], Dict[str, Any]],
):
    criteria = MongoCriteriaFactory()

    def __init__(
        self,
        collection: MotorCollection,
        session: Optional[MotorClientSession] = None,
    ):
        super().__init__(collection, MongoEntitySerializer(), session)


class TestRepository(TestCase):
    _async = AsyncExecutor()

    DB_NAME = "db"
    COLLECTION = "entities"

    data = pd.read_csv(os.path.join(DATA_FOLDER, "donors.dummy.csv"), sep=";")
    entities = [Entity(**row) for _, row in data.iterrows()]

    collection = _async.execute(init_mongo(DB_NAME, COLLECTION))

    repo = MongoEntityRepository(collection)

    def test_001_insert_dummies(self) -> None:
        entities = self._async.execute(self.repo.save(self.entities))

        self.assertEqual(len(entities), len(self.entities))

        self.assertEqual(
            self._async.execute(self.repo.collection.count_documents({})),
            len(self.entities),
        )

    def test_002_query_options_sorted_desc(self) -> None:
        options = QueryOptions(
            0, 1, sorting_options=[("birth_year", SortingDirection.DES)]
        )

        entities = self._async.execute(self.repo.list(options))

        self.assertEqual(len(entities.items), 1)

        entity = entities.items[0]
        self.assertIsInstance(entity, Entity)
        self.assertEqual(entity.birth_year, 1989)

    def test_003_query_options_sorted_asc(self) -> None:
        options = QueryOptions(
            0, 1, sorting_options=[("birth_year", SortingDirection.ASC)]
        )

        entities = self._async.execute(self.repo.list(options))

        self.assertEqual(len(entities.items), 1)

        entity = entities.items[0]
        self.assertIsInstance(entity, Entity)
        self.assertEqual(entity.birth_year, 1985)

    def test_004_retrieve_by_id(self) -> None:
        self.assertIsNone(self._async.execute(self.repo.retrieve(00000)))

        self.assertIsNotNone(self._async.execute(self.repo.retrieve(1234)))

    def test_005_create_and_delete_entity(self) -> None:
        new_entity = Entity(cai=9999, birth_year=2000)

        self.assertIsNone(self._async.execute(self.repo.retrieve(9999)))

        _ = self._async.execute(self.repo.create(new_entity))

        self.assertIsNotNone(self._async.execute(self.repo.retrieve(9999)))

        self.assertTrue(self._async.execute(self.repo.delete(9999)))

        self.assertIsNone(self._async.execute(self.repo.retrieve(9999)))

    def test_006_retrieve_by_criteria(self) -> None:
        criteria = self.repo.criteria.by_birth_year(1985)

        entities = self._async.execute(self.repo.retrieve_by_criteria(criteria))

        self.assertEqual(len(entities.items), 1)

        criteria2 = self.repo.criteria.by_birth_year(1989)

        empty = self._async.execute(
            self.repo.retrieve_by_criteria(criteria & criteria2)
        )

        self.assertEqual(len(empty.items), 0)

        all_entities = self._async.execute(
            self.repo.retrieve_by_criteria(criteria | criteria2)
        )

        self.assertEqual(len(all_entities.items), 2)

    def test_007_delete_by_criteria(self) -> None:
        criteria = self.repo.criteria.by_birth_year(1985)

        self.assertTrue(self._async.execute(self.repo.delete_by_criteria(criteria)))

        left_over = self._async.execute(self.repo.list())

        self.assertEqual(len(left_over.items), 1)
