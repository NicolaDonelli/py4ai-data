"""Module containing implementations and abstractions for query to be used in FileSystem persistence layers."""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Iterable, List, Tuple

from py4ai.core.utils.decorators import same_type
from py4ai.core.utils.dict import union

from py4ai.data.layer.common.criteria import SearchCriteria
from py4ai.data.layer.common.repository import KE, E, SortingDirection


class FileSystemSearchCriteria(SearchCriteria[List[KE]]):
    """Base class for representing a FileSystem query."""

    def __init__(self, iterable: Iterable[KE]):
        """Instantiate class based on list of filenames.

        :param iterable: list of filenames
        """
        self.__elements__ = [arg for arg in iterable]

    @property
    def query(self) -> List[KE]:
        """Return the underlying query based on a PandasFilter condition.

        :returns: underlying query
        """
        return self.__elements__

    @same_type
    def __or__(self, other: SearchCriteria) -> "FileSystemSearchCriteria":
        """Return query resulting from OR operation between queries.

        :param other: the other query to be used in the AND operation
        :returns: resulting query
        """
        return FileSystemSearchCriteria(set(self.query).union(other.query))

    @same_type
    def __and__(self, other: SearchCriteria) -> "FileSystemSearchCriteria":
        """Return query resulting from AND operation between queries.

        :param other: the other query to be used in the AND operation
        :returns: resulting query
        """
        return FileSystemSearchCriteria(set(self.query).intersection(other.query))


class FileSystemCriteriaFactory(Generic[KE, E]):
    """Base class to be used for extending CriteriaFactory for FileSystems."""

    def __init__(self, path: Path, index_file: Path = Path("indices.json")):
        """Instantiate the class.

        :param path: path where to store the objects
        :param index_file: file where to store the indexing of the files, to be used to querying and fast-retrieving.
        """
        self.path = path
        self.index_file = index_file

    @property
    def index(self) -> Dict[KE, Dict]:
        """Return a hash-property dictionary, to be used to filter objects and retrieve the corresponding hash.

        :returns: dictionary with entity hash, properties as (key, value) pairs.
        """
        try:
            with open(os.path.join(self.path, self.index_file), "r") as fid:
                return json.load(fid)
        except FileNotFoundError:
            return {}

    def get_index_fields(self, entity: E) -> Dict[str, Any]:
        """Extract the indexed field from an entity.

        :param entity: entity
        :returns: indices fields
        """
        return {}

    def update_index(self, key: KE, entity: E):
        """Update indices file in the file-system, with the entity key and the corresponding indexed fields.

        :param key: entity key
        :param entity: entity whose indices are to be computed
        """
        data = self.get_index_fields(entity)
        index = self.index
        with open(os.path.join(self.path, self.index_file), "w") as fid:
            json.dump(union(index, {key: data}), fid)

    def filter_path_by_condition(
        self, condition: Callable[[Dict], bool]
    ) -> FileSystemSearchCriteria:
        """Return query with filtered elements based on a given condition.

        :param condition: filtering function to be applied at indexed fields.
        :returns: resulting query
        """
        keys = [key for key, data in self.index.items() if condition(data)]
        return FileSystemSearchCriteria(keys)

    @staticmethod
    def format_name(name: Path, path: Path):
        """Reformat full path to transform it to a file-system key.

        :param name: full path.
        :param path: base path to be used to strip the full path.
        :returns: file-system key.
        """
        return os.path.splitext(name.relative_to(path))[0]

    def all(self) -> FileSystemSearchCriteria:
        """Return empty query.

        :returns: empty query.
        """
        return FileSystemSearchCriteria(self.index.keys())

    def sort_by(
        self,
        criteria: FileSystemSearchCriteria,
        sorting_option: Tuple[str, SortingDirection],
    ) -> FileSystemSearchCriteria:
        """Create a sorted query, based on a sorting option.

        :param criteria: file-system query.
        :param sorting_option: sorting options to be used.
        :returns: sorted query.
        """
        index = self.index
        name, order = sorting_option
        return FileSystemSearchCriteria(
            sorted(
                criteria.query,
                key=lambda key: index[key][name],
                reverse=False if order == SortingDirection.ASC else True,
            )
        )
