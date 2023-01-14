import os
from shutil import copyfile, rmtree

import pandas as pd
from py4ai.core.tests.core import TestCase
from py4ai.core.utils.executors import AsyncExecutor
from py4ai.core.utils.fs import create_dir_if_not_exists
from pydantic import BaseModel

from py4ai.data.layer.common.repository import QueryOptions, SortingDirection
from py4ai.data.layer.common.serialiazer import DataSerializer
from py4ai.data.layer.pandas.criteria import PandasSearchCriteria
from py4ai.data.layer.pandas.repository import CsvRepository
from tests import DATA_FOLDER, TMP_FOLDER


class DummyEntity(BaseModel):
    cai: int
    birth_year: int


class DummySerializer(DataSerializer[int, int, DummyEntity, pd.Series]):
    def to_object(self, entity: DummyEntity) -> pd.Series:
        return pd.Series(entity.dict())

    def to_entity(self, document: pd.Series) -> DummyEntity:
        return DummyEntity(**document)

    def to_object_key(self, key: int) -> int:
        return key

    def get_key(self, entity: DummyEntity) -> int:
        return entity.cai


class TestRepository(TestCase):
    _async = AsyncExecutor()

    @classmethod
    def setUpClass(cls) -> None:
        create_dir_if_not_exists(TMP_FOLDER)

    @classmethod
    def tearDownClass(cls) -> None:
        rmtree(TMP_FOLDER)

    def test_retrieve_list(self) -> None:
        filename = os.path.join(DATA_FOLDER, "donors.dummy.csv")

        repo = CsvRepository(filename, DummySerializer())

        entities = self._async.execute(repo.list())

        self.assertEqual(len(entities.items), 2)

    def test_query_options_sorted_desc(self) -> None:
        filename = os.path.join(DATA_FOLDER, "donors.dummy.csv")

        repo = CsvRepository(filename, DummySerializer())

        options = QueryOptions(
            0, 1, sorting_options=[("birth_year", SortingDirection.DES)]
        )

        entities = self._async.execute(repo.list(options))

        self.assertEqual(len(entities.items), 1)

        entity = entities.items[0]
        self.assertIsInstance(entity, DummyEntity)
        self.assertEqual(entity.birth_year, 1989)

    def test_query_options_sorted_asc(self) -> None:
        filename = os.path.join(DATA_FOLDER, "donors.dummy.csv")

        repo = CsvRepository(filename, DummySerializer())

        options = QueryOptions(
            0, 1, sorting_options=[("birth_year", SortingDirection.ASC)]
        )

        entities = self._async.execute(repo.list(options))

        self.assertEqual(len(entities.items), 1)

        entity = entities.items[0]
        self.assertIsInstance(entity, DummyEntity)
        self.assertEqual(entity.birth_year, 1985)

    def test_retrieve_by_id(self) -> None:
        filename = os.path.join(DATA_FOLDER, "donors.dummy.csv")

        repo = CsvRepository(filename, DummySerializer())

        self.assertIsNone(self._async.execute(repo.retrieve(00000)))

        self.assertIsNotNone(self._async.execute(repo.retrieve(1234)))

    def test_create_and_delete_entity(self) -> None:
        copyfile(
            os.path.join(DATA_FOLDER, "donors.dummy.csv"),
            os.path.join(TMP_FOLDER, "donors.dummy.csv"),
        )

        filename = os.path.join(TMP_FOLDER, "donors.dummy.csv")

        repo = CsvRepository(filename, DummySerializer())

        self.assertIsNone(self._async.execute(repo.retrieve(9999)))

        new_entity = DummyEntity(cai=9999, birth_year=2000)

        _ = self._async.execute(repo.create(new_entity))

        self.assertIsNotNone(self._async.execute(repo.retrieve(9999)))

        self.assertTrue(self._async.execute(repo.delete(9999)))

        self.assertIsNone(self._async.execute(repo.retrieve(9999)))

        os.remove(filename)

    def test_retrieve_by_criteria(self) -> None:
        filename = os.path.join(DATA_FOLDER, "donors.dummy.csv")

        repo = CsvRepository(filename, DummySerializer())

        criteria = PandasSearchCriteria(lambda df: df["birth_year"] == 1985)

        entities = self._async.execute(repo.retrieve_by_criteria(criteria))

        self.assertEqual(len(entities.items), 1)

        criteria2 = PandasSearchCriteria(lambda df: df["birth_year"] == 1989)

        empty = self._async.execute(repo.retrieve_by_criteria(criteria & criteria2))

        self.assertEqual(len(empty.items), 0)

        all = self._async.execute(repo.retrieve_by_criteria(criteria | criteria2))

        self.assertEqual(len(all.items), 2)

    def test_delete_by_criteria(self) -> None:
        copyfile(
            os.path.join(DATA_FOLDER, "donors.dummy.csv"),
            os.path.join(TMP_FOLDER, "donors.dummy.csv"),
        )

        filename = os.path.join(TMP_FOLDER, "donors.dummy.csv")

        repo = CsvRepository(filename, DummySerializer())

        criteria = PandasSearchCriteria(lambda df: df["birth_year"] == 1985)

        self.assertTrue(self._async.execute(repo.delete_by_criteria(criteria)))

        left_over = self._async.execute(repo.list())

        self.assertEqual(len(left_over.items), 1)

        os.remove(filename)

    def test_from_not_existing(self) -> None:
        filename = os.path.join(TMP_FOLDER, "not-existing.file.csv")

        repo = CsvRepository(filename, DummySerializer())

        entities = self._async.execute(repo.list())

        self.assertEqual(len(entities.items), 0)

    def test_from_empty(self) -> None:
        filename = os.path.join(TMP_FOLDER, "empty.file.csv")
        with open(filename, "w") as fid:
            fid.write("")

        repo = CsvRepository(filename, DummySerializer())

        entities = self._async.execute(repo.list())

        self.assertEqual(len(entities.items), 0)
