import os
import random
from typing import Any

from mongomock import MongoClient
from py4ai.core.utils.fs import create_dir_if_not_exists

test_path = os.path.dirname(os.path.abspath(__file__))

RESOURCE_FOLDER = os.path.join(test_path, "resources")

DATA_FOLDER = os.path.join(RESOURCE_FOLDER, "data")
TMP_FOLDER = str(
    create_dir_if_not_exists(os.path.join("/tmp", "%032x" % random.getrandbits(128)))
)

os.environ["TMP_LOG_FOLDER"] = str(
    create_dir_if_not_exists(os.path.join(TMP_FOLDER, "logs"))
)

DB_NAME = "db"

client: Any = MongoClient()

db = client[DB_NAME]


def clean_tmp_folder() -> None:
    os.rmdir(TMP_FOLDER)


def unset_TMP_FOLDER() -> None:
    del os.environ["TMP_LOG_FOLDER"]
