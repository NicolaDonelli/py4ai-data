import os
import pickle
import shutil
from glob import glob
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from py4ai.core.tests.core import TestCase
from py4ai.core.utils.executors import AsyncExecutor

from py4ai.data.layer.common.repository import QueryOptions, SortingDirection
from py4ai.data.layer.fs.criteria import (
    FileSystemCriteriaFactory,
    FileSystemSearchCriteria,
)
from py4ai.data.layer.fs.repository import FileSystemRepository
from py4ai.data.layer.fs.serializer import FileSerializer, FileSerializerMode, IndexedIO
from tests import DATA_FOLDER, TMP_FOLDER
from tests.data.layer.base import CriteriaFactory, Entity, EntityRepository


class PickleEntitySerializer(FileSerializer[int, Entity]):
    mode = FileSerializerMode.BINARY

    def get_key(self, entity: Entity) -> int:
        return entity.cai

    def to_object_key(self, key: int) -> str:
        return os.path.join(self.path, str(key) + ".pkl")

    def to_entity(self, document: IndexedIO) -> Entity:
        return pickle.load(document.buffer)

    def to_object(self, entity: Entity) -> IndexedIO:
        buffer = BytesIO()
        buffer.write(pickle.dumps(entity, protocol=pickle.HIGHEST_PROTOCOL))
        buffer.seek(0)
        return IndexedIO(name=self.get_key(entity), buffer=buffer)


class FileSystemEntityCriteriaFactory(
    CriteriaFactory[List[int]], FileSystemCriteriaFactory[int, Entity]
):
    def get_index_fields(self, entity: Entity) -> Dict[str, Any]:
        return {"cai": entity.cai, "birth_year": entity.birth_year}

    def by_cai(self, cai: int) -> FileSystemSearchCriteria:
        return self.filter_path_by_condition(lambda data: data["cai"] == cai)

    def from_birth_year(self, birth_year: int) -> FileSystemSearchCriteria:
        return self.filter_path_by_condition(
            lambda data: data["birth_year"] >= birth_year
        )

    def by_birth_year(self, birth_year: int) -> FileSystemSearchCriteria:
        return self.filter_path_by_condition(
            lambda data: data["birth_year"] == birth_year
        )


class PickleEntityRepository(
    FileSystemRepository[int, Entity],
    EntityRepository[str, Dict[str, Any], List[int]],
):
    criteria: FileSystemEntityCriteriaFactory

    def __init__(self, path: Path, serializer: PickleEntitySerializer):
        super().__init__(path, serializer)
        self.criteria = FileSystemEntityCriteriaFactory(path)


class TestRepository(TestCase):
    _async = AsyncExecutor()

    data = pd.read_csv(os.path.join(DATA_FOLDER, "donors.dummy.csv"), sep=";")
    entities = [Entity(**row) for _, row in data.iterrows()]

    folder = Path(os.path.join(TMP_FOLDER, "fs-repo"))

    repo = PickleEntityRepository(folder, PickleEntitySerializer(folder))

    def count_elements(self) -> int:
        return len(
            set(glob(os.path.join(self.folder, "*"), recursive=True)).difference(
                {os.path.join(self.folder, self.repo.criteria.index_file)}
            )
        )

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.folder)

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

    def test_004_retrieve_by_id(self):
        self.assertIsNone(self._async.execute(self.repo.retrieve(00000)))

        self.assertIsNotNone(self._async.execute(self.repo.retrieve(1234)))

    def test_005_create_and_delete_entity(self):
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

    def test_007_delete_by_criteria(self):
        criteria = self.repo.criteria.by_birth_year(1985)

        self.assertTrue(self._async.execute(self.repo.delete_by_criteria(criteria)))

        left_over = self._async.execute(self.repo.list())

        self.assertEqual(len(left_over.items), 1)
