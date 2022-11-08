from py4ai.data.model.ml import PandasDataset, PandasTimeIndexedDataset
from py4ai.data.datasets import get_unbalanced_dataset, get_weather_nyc_dataset
from py4ai.core.tests.core import TestCase, logTest


class TestLoadDatasets(TestCase):
    @logTest
    def test_unbalanced_dataset(self):
        self.assertIsInstance(get_unbalanced_dataset(), PandasDataset)

    @logTest
    def test_time_indexed_dataset(self):
        self.assertIsInstance(get_weather_nyc_dataset(), PandasTimeIndexedDataset)
