import os
from typing import Any, Dict

import pandas as pd
from py4ai.core.tests.core import TestCase
from py4ai.core.utils.executors import AsyncExecutor
from py4ai.core.utils.fs import create_dir_if_not_exists
from sqlalchemy import Column, Integer, MetaData, String, Table, func, select
from sqlalchemy.engine.cursor import CursorResult
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.sql import Select

from py4ai.data.layer.common.repository import QueryOptions, SortingDirection
from py4ai.data.layer.sqlalchemy.criteria import (
    SqlAlchemyQuery,
    SqlAlchemySearchCriteria,
)
from py4ai.data.layer.sqlalchemy.repository import SqlAlchemyRepository
from py4ai.data.layer.sqlalchemy.serializer import SqlAlchemySerializer
from tests import DATA_FOLDER, TMP_FOLDER
from tests.data.layer.base import (
    CriteriaFactory,
    Entity,
    EntityDataSerializer,
    EntityRepository,
)


class SqlCriteriaFactory(CriteriaFactory[SqlAlchemyQuery]):
    def __init__(self, table: Table):
        self.table = table

    def by_cai(self, cai: int) -> SqlAlchemySearchCriteria:
        return SqlAlchemySearchCriteria(self.table.c.cai == cai)

    def from_birth_year(self, birth_year: int) -> SqlAlchemySearchCriteria:
        return SqlAlchemySearchCriteria(self.table.c.birth_year >= birth_year)

    def by_birth_year(self, birth_year: int) -> SqlAlchemySearchCriteria:
        return SqlAlchemySearchCriteria(self.table.c.birth_year == birth_year)


class SqlEntitySerializer(
    SqlAlchemySerializer[int, int, Entity], EntityDataSerializer[int, Dict[str, Any]]
):
    def __init__(self, table: Table):
        super(SqlEntitySerializer, self).__init__(table)

    def to_object(self, entity: Entity) -> Dict[str, int]:
        doc = entity.dict()
        doc["id"] = self.to_object_key(self.get_key(entity))
        return doc

    def to_entity(self, document: Dict[str, int]) -> Entity:
        return Entity(**document)

    def to_object_key(self, key: int) -> int:
        return key


class SqlEntityRepository(
    SqlAlchemyRepository[int, int, Entity],
    EntityRepository[int, Dict[str, Any], SqlAlchemyQuery],
):
    def __init__(
        self, engine: AsyncEngine, serializer: SqlAlchemySerializer[int, int, Entity]
    ) -> None:
        super().__init__(engine, serializer)
        self.criteria = SqlCriteriaFactory(self.table)


class TestRepository(TestCase):
    _async = AsyncExecutor()

    DB_NAME = "db"
    COLLECTION = "entities"

    data = pd.read_csv(os.path.join(DATA_FOLDER, "donors.dummy.csv"), sep=";")
    entities = [Entity(**row) for _, row in data.iterrows()]

    engine = create_async_engine(f"sqlite+aiosqlite:///{TMP_FOLDER}/db")

    meta = MetaData()

    table = Table(
        "table",
        meta,
        Column("id", String(24), primary_key=True),
        Column("cai", Integer, nullable=False),
        Column("birth_year", Integer),
    )

    repo = SqlEntityRepository(engine, SqlEntitySerializer(table))

    @classmethod
    async def init_models(cls) -> None:
        async with cls.engine.begin() as conn:
            await conn.run_sync(cls.meta.drop_all)
            await conn.run_sync(cls.meta.create_all)

    async def execute_query(self, query: Select) -> CursorResult:
        async with self.engine.begin() as conn:
            return await conn.execute(query)

    def count_elements(self) -> int:
        q = select([func.count()]).select_from(self.table)
        return self._async.execute(self.execute_query(q)).all()[0][0]

    @classmethod
    def setUpClass(cls) -> None:
        create_dir_if_not_exists(TMP_FOLDER)
        cls._async.execute(cls.init_models())

    def test_001_insert_dummies(self) -> None:
        entities = self._async.execute(self.repo.save(self.entities))

        self.assertEqual(len(entities), self.count_elements())

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
