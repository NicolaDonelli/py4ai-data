"""
Test criteria files
"""

from typing import Any, Callable

import numpy as np
import pandas as pd
from py4ai.core.tests.core import TestCase

from py4ai.data.layer.common.criteria import SearchCriteria
from py4ai.data.layer.pandas.criteria import PandasSearchCriteria


class DummyCriteria(SearchCriteria[dict]):
    def query(self) -> dict:
        return {}

    def __or__(self, other: SearchCriteria) -> "DummyCriteria":
        return DummyCriteria()

    def __and__(self, other: SearchCriteria) -> "DummyCriteria":
        return DummyCriteria()


class TestConfig(TestCase):
    df = pd.DataFrame({"a": np.arange(0, 10)})

    @staticmethod
    def filter_by_key_value(key: str, value: Any) -> PandasSearchCriteria:
        def func(df: pd.DataFrame) -> pd.Series:
            return df[key].apply(lambda x: x == value)

        return PandasSearchCriteria(func)

    @staticmethod
    def filter_by_func(key: str, func: Callable[[Any], bool]) -> PandasSearchCriteria:
        def _func(df: pd.DataFrame) -> pd.Series:
            return df[key].apply(func)

        return PandasSearchCriteria(_func)

    @classmethod
    def perform_query(cls, query: PandasSearchCriteria) -> pd.DataFrame:
        return cls.df[query.query(cls.df)]

    def test_query(self) -> None:
        query = self.filter_by_key_value("a", 5)
        result = self.perform_query(query)
        self.assertEqual(1, len(result))

        query_2 = self.filter_by_func("a", lambda x: x < 5)
        result = self.perform_query(query_2)
        self.assertEqual(5, len(result))

    def test_query_or(self) -> None:
        query = self.filter_by_key_value("a", 5) | self.filter_by_func(
            "a", lambda x: x < 5
        )
        result = self.perform_query(query)
        self.assertEqual(6, len(result))

    def test_query_or_invalid(self) -> None:
        self.assertRaises(
            TypeError, lambda: self.filter_by_key_value("a", 5) | DummyCriteria()
        )

    def test_query_and_invalid(self) -> None:
        self.assertRaises(
            TypeError, lambda: self.filter_by_key_value("a", 5) & DummyCriteria()
        )

    def test_query_and(self) -> None:
        query = self.filter_by_key_value("a", 5) & self.filter_by_func(
            "a", lambda x: x < 5
        )
        result = self.perform_query(query)
        self.assertEqual(0, len(result))

        query = self.filter_by_key_value("a", 3) & self.filter_by_func(
            "a", lambda x: x < 5
        )
        result = self.perform_query(query)
        self.assertEqual(1, len(result))
