import unittest

from py4ai.core.logging import getDefaultLogger
from py4ai.core.tests.core import TestCase, logTest

from py4ai.data.model.text import CachedDocuments, Document, LazyDocuments

logger = getDefaultLogger()

n = 10


def createCorpus(n):
    for i in range(n):
        yield Document(str(i), {"text": "my text 1"})


class TestDocuments(TestCase):
    docs = (
        CachedDocuments(createCorpus(n))
        .map(lambda x: x.addProperty("tags", {"1": "1"}))
        .map(lambda x: x.addProperty("tags", {"2": "2"}))
    )

    @logTest
    def test_documents_parsing(self):
        filteredDocs = self.docs.filter(lambda x: int(x.uuid) % 2)
        self.assertIsInstance(filteredDocs, LazyDocuments)
        self.assertEqual(len(filteredDocs.to_cached()), n / 2)

    @logTest
    def test_documents_cached(self):
        filteredDocs = self.docs.filter(lambda x: int(x.uuid) % 2).to_cached()
        self.assertIsInstance(filteredDocs, CachedDocuments)


if __name__ == "__main__":
    unittest.main()
