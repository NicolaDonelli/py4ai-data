from abc import ABC, abstractmethod
from typing import Generic

from py4ai.core.types import KD, D, Q
from pydantic import BaseModel

from py4ai.data.layer.common.criteria import SearchCriteria
from py4ai.data.layer.common.repository import Repository
from py4ai.data.layer.common.serialiazer import DataSerializer


class Entity(BaseModel):
    cai: int
    birth_year: int


class EntityDataSerializer(DataSerializer[int, KD, Entity, D], ABC):
    def get_key(self, entity: Entity) -> int:
        return entity.cai


class CriteriaFactory(Generic[Q], ABC):
    @abstractmethod
    def by_cai(self, cai: int) -> SearchCriteria[Q]:
        ...

    @abstractmethod
    def from_birth_year(self, birth_year: int) -> SearchCriteria[Q]:
        ...

    @abstractmethod
    def by_birth_year(self, birth_year: int) -> SearchCriteria[Q]:
        ...


class EntityRepository(Repository[int, KD, Entity, D, Q], Generic[KD, D, Q], ABC):
    criteria: CriteriaFactory[Q]
    # serializer: EntityDataSerializer[KD, D]
