"""Module for Repository pattern for FileSystem persistence layers."""

import asyncio
import os
from pathlib import Path
from typing import Generic, List, Optional, Sequence, cast

from py4ai.data.layer.common.repository import (
    Paged,
    QueryOptions,
    Repository,
    SearchCriteria,
)
from py4ai.data.layer.fs import create_dir_if_not_exists
from py4ai.data.layer.fs.criteria import (
    FileSystemCriteriaFactory,
    FileSystemSearchCriteria,
)
from py4ai.data.layer.fs.serializer import KE, E, FileSerializer, IndexedIO


class FileSystemRepository(Repository[KE, str, E, IndexedIO, List[KE]], Generic[KE, E]):
    """Class implementing MongoDB repository."""

    criteria: FileSystemCriteriaFactory

    def __init__(self, path: Path, serializer: FileSerializer[KE, E]):
        """Return a FileSystem Repository Implementation.

         The current implementation uses the Motor async framework.

        :param path: location where objects are stored
        :param serializer: Serializer to be used to serialize/deserialize FileSystem raw objects into domain objects
        """
        self.path = create_dir_if_not_exists(path)
        self._serializer = serializer.with_path(path)

    @property
    def serializer(self) -> FileSerializer[KE, E]:
        """Return the serializer.

        :return: DataSerializer for serializing/deserializing FileSystem documents into domain objects
        """
        return self._serializer

    async def create(self, entity: E) -> E:
        """Create the entity in the underlying persistence layer.

        :param entity: Entity to be created
        :returns: same entity provided as input, after creation. If creation fails, an error should be returned.
        """
        ibuffer = self.serializer.to_object(entity)
        path_to_file = self.serializer.to_object_key(ibuffer.name)
        _ = create_dir_if_not_exists(os.path.dirname(path_to_file))
        with open(
            path_to_file, "w" + self.serializer.mode, encoding=self.serializer.encoding
        ) as fid:
            fid.write(ibuffer.buffer.read())
        self.criteria.update_index(self.serializer.get_key(entity), entity)
        return entity

    async def retrieve(self, key: KE) -> Optional[E]:
        """Return an entry corresponding to a determined Entity key. If no match is found, returns None.

        :param key: Entity Key to be used for retrieving the entity.
        :returns: Entity associated with the provided key. If no match is found, None is returned.
        """
        file_name = self.serializer.to_object_key(key)
        if os.path.exists(file_name):
            with open(
                file_name, "r" + self.serializer.mode, encoding=self.serializer.encoding
            ) as fid:
                x: E = self.serializer.to_entity(IndexedIO(name=key, buffer=fid))
                return x
        else:
            return None

    async def retrieve_by_criteria(
        self, criteria: SearchCriteria[List[KE]], options: QueryOptions = QueryOptions()
    ) -> Paged[E]:
        """Return a list of entities, matching the query provided.

        :param criteria: query to be used for selecting items
        :param options: query options to be used when retrieving data
        :returns: Paged object for retrieved list of entities
        :raises ValueError: when multiple sorting options are provided. Note that FileSystem repositories only
            support one sorting option.
        """
        if len(options.sorting_options) > 0:
            if len(options.sorting_options) > 1:
                raise ValueError(
                    "Multiple sorting options are not allowed for FileSystemRepository"
                )
            criteria = self.criteria.sort_by(
                cast(FileSystemSearchCriteria, criteria), options.sorting_options[0]
            )

        query = criteria.query

        selection = (
            query
            if options.page_size < 0
            else query[
                options.page_start : options.page_start + options.page_size
            ]  # noqa: ignore
        )
        all_elements = await asyncio.gather(*[self.retrieve(key) for key in selection])
        elements = [element for element in all_elements if element is not None]
        size = len(elements)
        return Paged(size, elements, options.page_start + size < len(selection))

    # async def update(self, entity: E) -> Optional[E]:
    #    return await self.create(entity)

    async def save(self, entities: Sequence[E]) -> Sequence[E]:
        """Create the entries in the persistence layer associated to a list of entities.

        :param entities: list of entities to be created.
        :returns: list of entities that have been successfully created in the persistence layer
        """
        elements: List[Optional[E]] = await asyncio.gather(
            *[self.create(entity) for entity in entities]
        )
        return [element for element in elements if element is not None]

    async def delete(self, key: KE) -> bool:
        """Delete the entry in the persisence layer associated to the provided entity key.

        :param key: key identifying the entity.
        :returns: boolean value indicating whether the deletion has completed successfully.
        """
        try:
            os.remove(self.serializer.to_object_key(key))
            return True
        except FileNotFoundError:
            return False

    async def delete_by_criteria(self, criteria: SearchCriteria[List[KE]]) -> bool:
        """Delete all entries matching a given query.

        :param criteria: query to be used for deleting entries.
        :returns: boolean value indicating whether the deletion has completed successfully.
        """
        return all(await asyncio.gather(*[self.delete(key) for key in criteria.query]))

    async def list(self, options: QueryOptions = QueryOptions()) -> Paged[E]:
        """Return a full list of entities stored in the persistence layer.

        :param options: query options to be used when retrieving data
        :returns: Paged object for retrieved list of entities
        """
        return await self.retrieve_by_criteria(self.criteria.all(), options)
