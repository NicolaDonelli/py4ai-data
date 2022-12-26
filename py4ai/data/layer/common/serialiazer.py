"""Module containing implementations and abstractions for data serializers."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

KE = TypeVar("KE")
KD = TypeVar("KD")
E = TypeVar("E")
D = TypeVar("D")


class DataSerializer(ABC, Generic[KE, KD, E, D]):
    """Base DataSerializer."""

    @abstractmethod
    def get_key(self, entity: E) -> KE:
        """Extract key for given entity.

        :param entity: provided entity
        :returns: entity key
        """
        ...

    @abstractmethod
    def to_object_key(self, key: KE) -> KD:
        """Transform entity key into raw key, to be used for indexing in the persistence layer.

        :param key: entity key
        :returns: raw key
        """
        ...

    @abstractmethod
    def to_entity(self, document: D) -> E:
        """Deserialize raw content into domain object entity.

        :param document: raw content
        :returns: domain object entity
        """
        ...

    @abstractmethod
    def to_object(self, entity: E) -> D:
        """Serialize domain object entity into raw content.

        :param entity: domain object entity
        :returns: raw content
        """
        ...
