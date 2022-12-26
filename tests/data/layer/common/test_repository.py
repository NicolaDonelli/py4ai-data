"""
Test config files
"""

from py4ai.core.tests.core import TestCase

from py4ai.data.layer.common.repository import QueryOptions


class TestConfig(TestCase):
    query = QueryOptions(0, 10)

    def test_query_options_copy(self) -> None:
        self.assertEqual(5, self.query.copy(page_start=5).page_start)
