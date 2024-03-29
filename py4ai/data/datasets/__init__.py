"""Module containing samples datasets to be used for testing."""

import os
from typing import Any

import pandas as pd

from py4ai.data.model.ml import PandasDataset, PandasTimeIndexedDataset

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def get_weather_nyc_dataset(
    filename: str = os.path.join(DATA_FOLDER, "weather_nyc_short.csv")
) -> PandasTimeIndexedDataset[Any, float]:
    """Return the NYC Weather dataset.

    :param filename: name of the csv file where the Weather NYC data are stored.
    :return: PandasTimeIndexedDataset object with the NYC Weather dataset.
    """
    df = pd.read_csv(filename, index_col="Date")
    return PandasTimeIndexedDataset(features=df.drop("TempM", axis=1), labels=df.TempM)


def get_unbalanced_dataset(
    features_file: str = os.path.join(DATA_FOLDER, "unbalanced_features.p"),
    labels_file: str = os.path.join(DATA_FOLDER, "unbalanced_labels.p"),
) -> PandasDataset[Any, Any]:
    """Return unbalanced dataset.

    :param features_file: name of the pickle file where the features data are stored.
    :param labels_file: name of the pickle file where the labels data are stored.
    :return: PandasDataset object with unbalanced classes.
    """
    return PandasDataset.createObject(
        pd.read_pickle(features_file), pd.read_pickle(labels_file)
    )
