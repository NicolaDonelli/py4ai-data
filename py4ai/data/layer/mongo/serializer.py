"""Module for implementation of serializer objects for Mongo persistence layers."""

from hashlib import md5

from bson import ObjectId
from pydantic.main import ModelMetaclass


def create_mongo_id(key: str) -> ObjectId:
    """Create a MongoDB hash compatible key from a general string.

    :param key: input string to be converted to a Mongo compatible hash
    :returns: MongoDB compatible ObjectId
    """
    return ObjectId(md5(str(key).encode("utf-8")).hexdigest()[:24])


class MongoModel(ModelMetaclass):
    """Class to convert pydantic model into MongoDB schema objects."""

    def __init__(cls, name, bases, dct):  # type: ignore
        """Create a MongoDB schema object.

        :param name: name of the class
        :param bases: Bases for the class
        :param dct: extra arguments
        """
        super().__init__(name, bases, dct)

        path = dct.get("__mongo_path__", "")

        prefix = f"{path}." if len(path) > 0 else ""

        for k, v in cls.__fields__.items():
            field_name = f"{prefix}{v.name or v.alias}"

            if isinstance(v.type_, ModelMetaclass):
                setattr(
                    cls,
                    k,
                    MongoModel(
                        v.type_.__name__,
                        (v.type_,),
                        {
                            "__module__": v.type_.__module__,
                            "__mongo_path__": field_name,
                        },
                    ),
                )
            else:
                setattr(cls, k, field_name)

    def __str__(self) -> str:
        """Return a string representation of the class.

        :returns: string representation
        """
        return str(self.__dict__["__mongo_path__"])


# from typing import Any
# from py4ai.core.data.layer.common.serialiazer import DataSerializer
#
# class RawMongoSerializer(DataSerializer[ObjectId, ObjectId, dict, dict]):
#
#     dollar_sign = "$"
#
#     def __init__(self, special_character: str = "§"):
#         self.special_character = special_character
#
#     def get_key(self, entity: dict) -> ObjectId:
#         return entity["_id"]
#
#     def to_document_key(self, key: ObjectId) -> ObjectId:
#         return key
#
#     def to_document(self, dict_in: dict) -> dict:
#         def transform_values(obj: Any) -> Any:
#             if isinstance(obj, list):
#                 return [transform_values(v) for v in obj]
#             elif isinstance(obj, dict):
#                 return self.to_document(obj)
#             else:
#                 return obj
#
#         return {
#             self.special_character if k == self.dollar_sign else k: transform_values(v)
#             for k, v in dict_in.items()
#         }
#
#     def to_model(self, dict_in: dict) -> dict:
#         def transform_values(obj: Any) -> Any:
#             if isinstance(obj, list):
#                 return [transform_values(v) for v in obj]
#             elif isinstance(obj, dict):
#                 return self.to_model(obj)
#             else:
#                 return obj
#
#         return {
#             self.dollar_sign if k == self.special_character else k: transform_values(v)
#             for k, v in dict_in.items()
#         }
