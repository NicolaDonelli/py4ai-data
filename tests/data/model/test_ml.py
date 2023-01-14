import os
import unittest
from shutil import rmtree
from typing import Any, Generator, Iterator, List, cast

import numpy as np
import pandas as pd
from py4ai.core.tests.core import TestCase, logTest
from py4ai.core.utils.fs import create_dir_if_not_exists

from py4ai.data.model.ml import (
    AllowedTypes,
    CachedDataset,
    IterGenerator,
    LazyDataset,
    MultiFeatureSample,
    PandasDataset,
    PandasTimeIndexedDataset,
    Sample,
    features_and_labels_to_dataset,
)
from tests import TMP_FOLDER

samples = [
    Sample(features=[100, 101], label=1),
    Sample(features=[102, 103], label=2),
    Sample(features=[104, 105], label=3),
    Sample(features=[106, 107], label=4),
    Sample(features=[108, 109], label=5),
    Sample(features=[110, 111], label=6),
    Sample(features=[112, 113], label=7),
    Sample(features=[114, 115], label=8),
    Sample(features=[116, 117], label=9),
]


def samples_gen() -> Iterator[Sample[List[int], int]]:
    for sample in samples:
        if not any([np.isnan(x).any() for x in sample.features]):
            yield sample


class TestFeaturesAndLabelsToDataset(TestCase):
    def test_features_and_labels_to_dataset(self) -> None:
        dataset = features_and_labels_to_dataset(
            pd.concat(
                [
                    pd.Series([1, 0, 2, 3], name="feat1"),
                    pd.Series([1, 2, 3, 4], name="feat2"),
                ],
                axis=1,
            ),
            pd.Series([0, 0, 0, 1], name="Label"),
        )

        dataset_no_labels = features_and_labels_to_dataset(
            pd.concat(
                [
                    pd.Series([1, 0, 2, 3], name="feat1"),
                    pd.Series([1, 2, 3, 4], name="feat2"),
                ],
                axis=1,
            ),
            None,
        )

        self.assertTrue(isinstance(dataset_no_labels, CachedDataset))
        self.assertTrue(isinstance(dataset, CachedDataset))
        self.assertTrue(
            (
                dataset.getFeaturesAs("pandas")
                == pd.concat(
                    [
                        pd.Series([1, 0, 2, 3], name="feat1"),
                        pd.Series([1, 2, 3, 4], name="feat2"),
                    ],
                    axis=1,
                )
            )
            .all()
            .all()
        )
        self.assertTrue(
            (
                dataset.getLabelsAs("pandas")
                == pd.DataFrame(pd.Series([0, 0, 0, 1], name="Label"))
            )
            .all()
            .all()
        )


class TestLazyDataset(TestCase):
    @logTest
    def test_withLookback_MultiFeatureSample(self) -> None:
        samples = [
            MultiFeatureSample(
                features=[np.array([100.0, 101.0]), np.array([np.NaN])], label=1.0
            ),
            MultiFeatureSample(
                features=[np.array([102.0, 103.0]), np.array([1.0])], label=2.0
            ),
            MultiFeatureSample(
                features=[np.array([104.0, 105.0]), np.array([2.0])], label=3.0
            ),
            MultiFeatureSample(
                features=[np.array([106.0, 107.0]), np.array([3.0])], label=4.0
            ),
            MultiFeatureSample(
                features=[np.array([108.0, 109.0]), np.array([4.0])], label=5.0
            ),
            MultiFeatureSample(
                features=[np.array([110.0, 111.0]), np.array([5.0])], label=6.0
            ),
            MultiFeatureSample(
                features=[np.array([112.0, 113.0]), np.array([6.0])], label=7.0
            ),
            MultiFeatureSample(
                features=[np.array([114.0, 115.0]), np.array([7.0])], label=8.0
            ),
            MultiFeatureSample(
                features=[np.array([116.0, 117.0]), np.array([8.0])], label=9.0
            ),
        ]

        def samples_gen() -> Iterator[MultiFeatureSample[float]]:
            for sample in samples:
                if not any([np.isnan(x).any() for x in sample.features]):
                    yield sample

        X1 = np.array(
            [
                [[102.0, 103.0], [104.0, 105.0], [106.0, 107.0]],
                [[104.0, 105.0], [106.0, 107.0], [108.0, 109.0]],
                [[106.0, 107.0], [108.0, 109.0], [110.0, 111.0]],
                [[108.0, 109.0], [110.0, 111.0], [112.0, 113.0]],
            ]
        )
        y1 = np.array(
            [
                [[1.0], [2.0], [3.0]],
                [[2.0], [3.0], [4.0]],
                [[3.0], [4.0], [5.0]],
                [[4.0], [5.0], [6.0]],
            ]
        )
        lab1 = np.array([4.0, 5.0, 6.0, 7.0])
        X2 = np.array(
            [
                [[110.0, 111.0], [112.0, 113.0], [114.0, 115.0]],
                [[112.0, 113.0], [114.0, 115.0], [116.0, 117.0]],
            ]
        )
        y2 = np.array([[[5.0], [6.0], [7.0]], [[6.0], [7.0], [8.0]]])
        lab2 = np.array([8.0, 9.0])

        lookback = 3
        batch_size = 4

        lazyDat: LazyDataset[List[np.ndarray[Any, np.dtype[Any]]], float] = LazyDataset(
            IterGenerator(samples_gen)
        )
        lookbackDat: LazyDataset[
            List[np.ndarray[Any, np.dtype[Any]]], float
        ] = lazyDat.withLookback(lookback)
        batch_gen = lookbackDat.batch(batch_size)

        batch1: CachedDataset[List[np.ndarray[Any, Any]], float] = next(batch_gen)
        batch2: CachedDataset[List[np.ndarray[Any, Any]], float] = next(batch_gen)

        tmp1 = batch1.getFeaturesAs("array")
        temp1X = np.array(list(map(lambda x: np.stack(x), tmp1[:, :, 0])))
        temp1y = np.array(list(map(lambda x: np.stack(x), tmp1[:, :, 1])))
        tmp1lab = batch1.getLabelsAs("array")

        res = [
            np.array_equal(temp1X, X1),
            np.array_equal(temp1y, y1),
            np.array_equal(tmp1lab, lab1),
        ]

        tmp2 = batch2.getFeaturesAs("array")
        temp2X = np.array(list(map(lambda x: np.stack(x), tmp2[:, :, 0])))
        temp2y = np.array(list(map(lambda x: np.stack(x), tmp2[:, :, 1])))
        tmp2lab = batch2.getLabelsAs("array")

        res = res + [
            np.array_equal(temp2X, X2),
            np.array_equal(temp2y, y2),
            np.array_equal(tmp2lab, lab2),
        ]

        self.assertTrue(all(res))

    @logTest
    def test_withLookback_ArrayFeatureSample(self) -> None:

        samples = [
            Sample(features=np.array([100, 101]), label=1),
            Sample(features=np.array([102, 103]), label=2),
            Sample(features=np.array([104, 105]), label=3),
            Sample(features=np.array([106, 107]), label=4),
            Sample(features=np.array([108, 109]), label=5),
            Sample(features=np.array([110, 111]), label=6),
            Sample(features=np.array([112, 113]), label=7),
            Sample(features=np.array([114, 115]), label=8),
            Sample(features=np.array([116, 117]), label=9),
        ]

        def samples_gen() -> Iterator[Sample[np.ndarray[Any, np.dtype[Any]], int]]:
            for sample in samples:
                if not any([np.isnan(x).any() for x in sample.features]):
                    yield sample

        X1 = np.array(
            [
                [[100, 101], [102, 103], [104, 105]],
                [[102, 103], [104, 105], [106, 107]],
                [[104, 105], [106, 107], [108, 109]],
                [[106, 107], [108, 109], [110, 111]],
            ]
        )
        lab1 = np.array([3, 4, 5, 6])
        X2 = np.array(
            [
                [[108, 109], [110, 111], [112, 113]],
                [[110, 111], [112, 113], [114, 115]],
                [[112, 113], [114, 115], [116, 117]],
            ]
        )
        lab2 = np.array([7, 8, 9])

        lookback = 3
        batch_size = 4

        lazyDat: LazyDataset[np.ndarray[Any, np.dtype[Any]], int] = LazyDataset(
            IterGenerator(samples_gen)
        )
        lookbackDat: LazyDataset[
            np.ndarray[Any, np.dtype[Any]], int
        ] = lazyDat.withLookback(lookback)
        batch_gen = lookbackDat.batch(batch_size)

        batch1: CachedDataset[np.ndarray[Any, Any], int] = next(batch_gen)
        batch2: CachedDataset[np.ndarray[Any, Any], int] = next(batch_gen)

        tmp1 = batch1.getFeaturesAs("array")
        tmp1lab = batch1.getLabelsAs("array")

        res = [np.array_equal(tmp1, X1), np.array_equal(tmp1lab, lab1)]

        tmp2 = batch2.getFeaturesAs("array")
        tmp2lab = batch2.getLabelsAs("array")

        res = res + [np.array_equal(tmp2, X2), np.array_equal(tmp2lab, lab2)]

        self.assertTrue(all(res))

    @logTest
    def test_withLookback_ListFeatureSample(self) -> None:

        samples = [
            Sample(features=[100, 101], label=1),
            Sample(features=[102, 103], label=2),
            Sample(features=[104, 105], label=3),
            Sample(features=[106, 107], label=4),
            Sample(features=[108, 109], label=5),
            Sample(features=[110, 111], label=6),
            Sample(features=[112, 113], label=7),
            Sample(features=[114, 115], label=8),
            Sample(features=[116, 117], label=9),
        ]

        def samples_gen() -> Iterator[Sample[List[int], int]]:
            for sample in samples:
                if not any([np.isnan(x).any() for x in sample.features]):
                    yield sample

        X1 = np.array(
            [
                [[100, 101], [102, 103], [104, 105]],
                [[102, 103], [104, 105], [106, 107]],
                [[104, 105], [106, 107], [108, 109]],
                [[106, 107], [108, 109], [110, 111]],
            ]
        )
        lab1 = np.array([3, 4, 5, 6])
        X2 = np.array(
            [
                [[108, 109], [110, 111], [112, 113]],
                [[110, 111], [112, 113], [114, 115]],
                [[112, 113], [114, 115], [116, 117]],
            ]
        )
        lab2 = np.array([7, 8, 9])

        lookback = 3
        batch_size = 4

        lazyDat: LazyDataset[List[int], int] = LazyDataset(IterGenerator(samples_gen))
        lookbackDat: LazyDataset[List[int], int] = lazyDat.withLookback(lookback)
        batch_gen = lookbackDat.batch(batch_size)

        batch1: CachedDataset[List[int], int] = next(batch_gen)
        batch2: CachedDataset[List[int], int] = next(batch_gen)

        tmp1 = batch1.getFeaturesAs("array")
        tmp1lab = batch1.getLabelsAs("array")

        res = [np.array_equal(tmp1, X1), np.array_equal(tmp1lab, lab1)]

        tmp2 = batch2.getFeaturesAs("array")
        tmp2lab = batch2.getLabelsAs("array")

        res = res + [np.array_equal(tmp2, X2), np.array_equal(tmp2lab, lab2)]

        self.assertTrue(all(res))

    @logTest
    def test_features_labels(self) -> None:
        lazyDat: LazyDataset[List[int], int] = LazyDataset(IterGenerator(samples_gen))
        self.assertTrue(isinstance(lazyDat.features(), Generator))
        self.assertTrue(isinstance(lazyDat.labels(), Generator))
        self.assertTrue(isinstance(lazyDat.getFeaturesAs("lazy"), Generator))
        self.assertTrue(isinstance(lazyDat.getLabelsAs("lazy"), Generator))
        self.assertEqual(next(lazyDat.getFeaturesAs("lazy")), samples[0].features)
        self.assertEqual(next(lazyDat.getLabelsAs("lazy")), samples[0].label)
        self.assertEqual(next(lazyDat.features()), samples[0].features)
        self.assertEqual(next(lazyDat.labels()), samples[0].label)

    @logTest
    def test_union(self) -> None:
        dataset: LazyDataset[List[int], int] = LazyDataset(IterGenerator(samples_gen))

        def samples_gen_1() -> Iterator[Sample[List[int], int]]:
            for sample in samples[:2]:
                if not any([np.isnan(x).any() for x in sample.features]):
                    yield sample

        dataset_1: LazyDataset[List[int], int] = LazyDataset(
            IterGenerator(samples_gen_1)
        )

        def samples_gen_2() -> Iterator[Sample[List[int], int]]:
            for sample in samples[2:]:
                if not any([np.isnan(x).any() for x in sample.features]):
                    yield sample

        dataset_2: LazyDataset[List[int], int] = LazyDataset(
            IterGenerator(samples_gen_2)
        )
        cached_2: CachedDataset[List[int], int] = CachedDataset(list(samples_gen_2()))

        self.assertEqual(
            dataset_1.union(dataset_2).getFeaturesAs("pandas"),
            dataset.getFeaturesAs("pandas"),
        )
        self.assertEqual(
            dataset_1.union(dataset_2).getLabelsAs("pandas"),
            dataset.getLabelsAs("pandas"),
        )
        self.assertIsInstance(dataset_1.union(cached_2), LazyDataset)
        self.assertEqual(
            dataset_1.union(cached_2).getFeaturesAs("pandas"),
            dataset.getFeaturesAs("pandas"),
        )
        self.assertEqual(
            dataset_1.union(cached_2).getLabelsAs("pandas"),
            dataset.getLabelsAs("pandas"),
        )


class TestCachedDataset(TestCase):
    lazyDat: LazyDataset[List[int], int]

    @classmethod
    def setUpClass(cls) -> None:
        cls.lazyDat = LazyDataset(IterGenerator(samples_gen))

    @logTest
    def test_to_df(self) -> None:
        self.assertTrue(isinstance(self.lazyDat.to_cached().to_df(), pd.DataFrame))
        self.assertTrue(
            (
                CachedDataset(self.lazyDat).to_df()["features"][0].values
                == [100, 102, 104, 106, 108, 110, 112, 114, 116]
            ).all()
        )
        self.assertTrue(
            (
                CachedDataset(self.lazyDat).to_df()["labels"][0].values
                == [1, 2, 3, 4, 5, 6, 7, 8, 9]
            ).all()
        )

    @logTest
    def test_asPandasDataset(self) -> None:
        self.assertTrue(
            isinstance(CachedDataset(self.lazyDat).asPandasDataset, PandasDataset)
        )
        self.assertTrue(
            (
                CachedDataset(self.lazyDat).asPandasDataset.features[0].values
                == [100, 102, 104, 106, 108, 110, 112, 114, 116]
            ).all()
        )
        self.assertTrue(
            (
                CachedDataset(self.lazyDat).asPandasDataset.labels[0].values
                == [1, 2, 3, 4, 5, 6, 7, 8, 9]
            ).all()
        )

    @logTest
    def test_union(self) -> None:
        dataset: CachedDataset[List[int], int] = CachedDataset(samples)
        dataset_1: CachedDataset[List[int], int] = CachedDataset(samples[:2])
        dataset_2: CachedDataset[List[int], int] = CachedDataset(samples[2:])

        def samples_gen_2() -> Iterator[Sample[List[int], int]]:
            for sample in samples[2:]:
                if not any([np.isnan(x).any() for x in sample.features]):
                    yield sample

        lazy_2: LazyDataset[List[int], int] = LazyDataset(IterGenerator(samples_gen_2))

        self.assertEqual(
            dataset_1.union(dataset_2).getFeaturesAs("pandas"),
            dataset.getFeaturesAs("pandas"),
        )
        self.assertEqual(
            dataset_1.union(dataset_2).getLabelsAs("pandas"),
            dataset.getLabelsAs("pandas"),
        )
        self.assertIsInstance(dataset_1.union(lazy_2), CachedDataset)
        self.assertEqual(
            dataset_1.union(lazy_2).getFeaturesAs("pandas"),
            dataset.getFeaturesAs("pandas"),
        )
        self.assertEqual(
            dataset_1.union(lazy_2).getLabelsAs("pandas"),
            dataset.getLabelsAs("pandas"),
        )


class TestPandasDataset(TestCase):
    dataset: PandasDataset[pd.DataFrame, pd.Series] = PandasDataset(
        features=pd.concat(
            [
                pd.Series([1, np.nan, 2, 3], name="feat1"),
                pd.Series([1, 2, 3, 4], name="feat2"),
            ],
            axis=1,
        ),
        labels=pd.Series([0, 0, 0, 1], name="Label"),
    )

    dataset_no_label: PandasDataset[Any, Any] = PandasDataset(
        features=pd.concat(
            [
                pd.Series([1, np.nan, 2, 3], name="feat1"),
                pd.Series([1, 2, 3, 4], name="feat2"),
            ],
            axis=1,
        )
    )

    @classmethod
    def setUpClass(cls) -> None:
        create_dir_if_not_exists(TMP_FOLDER)

    @classmethod
    def tearDownClass(cls) -> None:
        rmtree(TMP_FOLDER)

    @logTest
    def test_check_none(self) -> None:
        self.assertEqual(self.dataset._check_none(None), None)
        self.assertEqual(self.dataset._check_none("test"), "test")

    @logTest
    def test__len__(self) -> None:
        self.assertEqual(self.dataset.__len__(), 4)

    @logTest
    def test_items(self) -> None:
        self.assertTrue(isinstance(self.dataset.items, Iterator))
        self.assertEqual(next(self.dataset.items).features, {"feat1": 1.0, "feat2": 1})
        self.assertEqual(cast(pd.DataFrame, next(self.dataset.items).label)["Label"], 0)
        self.assertEqual(
            next(self.dataset_no_label.items).features, {"feat1": 1.0, "feat2": 1}
        )
        self.assertEqual(
            next(self.dataset_no_label.items).label,
            pd.Series(
                np.nan, index=[None], name=self.dataset_no_label.index[0], dtype=object
            ),
        )

    @logTest
    def test_dropna_none_labels(self) -> None:
        res = pd.concat(
            [pd.Series([1, 2, 3], name="feat1"), pd.Series([1, 3, 4], name="feat2")],
            axis=1,
        )

        self.assertTrue(
            (
                self.dataset.dropna(subset=["feat1"]).features.reset_index(drop=True)
                == res
            )
            .all()
            .all()
        )
        self.assertTrue(
            (
                self.dataset.dropna(feat__subset=["feat1"]).features.reset_index(
                    drop=True
                )
                == res
            )
            .all()
            .all()
        )
        self.assertTrue(
            (
                self.dataset.dropna(labs__subset=["Label"]).features.reset_index(
                    drop=True
                )
                == res
            )
            .all()
            .all()
        )

    @logTest
    def test_cached(self) -> None:
        self.assertTrue(self.dataset.cached)

    @logTest
    def test_features_labels(self) -> None:
        self.assertEqual(
            self.dataset.features,
            pd.concat(
                [
                    pd.Series([1, np.nan, 2, 3], name="feat1"),
                    pd.Series([1, 2, 3, 4], name="feat2"),
                ],
                axis=1,
            ),
        )
        self.assertTrue((self.dataset.labels["Label"] == pd.Series([0, 0, 0, 1])).all())

    @logTest
    def test_index(self) -> None:
        self.assertTrue((self.dataset.index == range(4)).all())

    @logTest
    def test_createObject(self) -> None:
        self.assertTrue(
            isinstance(
                PandasDataset.createObject(
                    features=pd.concat(
                        [
                            pd.Series([1, np.nan, 2, 3], name="feat1"),
                            pd.Series([1, 2, 3, 4], name="feat2"),
                        ],
                        axis=1,
                    ),
                    labels=None,
                ),
                PandasDataset,
            )
        )
        self.assertEqual(
            PandasDataset.createObject(
                features=pd.concat(
                    [
                        pd.Series([1, np.nan, 2, 3], name="feat1"),
                        pd.Series([1, 2, 3, 4], name="feat2"),
                    ],
                    axis=1,
                ),
                labels=None,
            ).features,
            self.dataset_no_label.features,
        )
        self.assertEqual(
            PandasDataset.createObject(
                features=pd.concat(
                    [
                        pd.Series([1, np.nan, 2, 3], name="feat1"),
                        pd.Series([1, 2, 3, 4], name="feat2"),
                    ],
                    axis=1,
                ),
                labels=None,
            ).labels,
            self.dataset_no_label.labels,
        )

    @logTest
    def test_take(self) -> None:
        self.assertTrue(isinstance(self.dataset.takeAsPandas(1), PandasDataset))
        self.assertEqual(
            self.dataset.takeAsPandas(1).features.feat2, pd.Series([1], name="feat2")
        )
        self.assertEqual(
            self.dataset.takeAsPandas(1).labels["Label"], pd.Series([0], name="Label")
        )

    @logTest
    def test_loc(self) -> None:
        self.assertEqual(self.dataset.loc([2]).features.loc[2]["feat1"], 2)
        self.assertEqual(self.dataset.loc([2]).features.loc[2]["feat2"], 3)
        self.assertEqual(self.dataset.loc([2]).labels.loc[2]["Label"], 0)
        self.assertEqual(
            self.dataset_no_label.loc([2]).labels,
            pd.DataFrame(
                np.nan,
                index=[self.dataset.features.index[2]],
                columns=[None],
                dtype=object,
            ),
        )

    @logTest
    def test_from_sequence(self) -> None:
        features_1 = pd.DataFrame(
            {"feat1": [1, 2, 3, 4], "feat2": [100, 200, 300, 400]}, index=[1, 2, 3, 4]
        )
        features_2 = pd.DataFrame(
            {"feat1": [9, 11, 13, 14], "feat2": [90, 110, 130, 140]},
            index=[10, 11, 12, 13],
        )
        features_3 = pd.DataFrame(
            {"feat1": [90, 10, 10, 1400], "feat2": [0.9, 0.11, 0.13, 0.14]},
            index=[15, 16, 17, 18],
        )
        labels_1 = pd.DataFrame({"target": [1, 0, 1, 1]}, index=[1, 2, 3, 4])
        labels_2 = pd.DataFrame({"target": [1, 1, 1, 0]}, index=[10, 11, 12, 13])
        labels_3 = pd.DataFrame({"target": [0, 1, 1, 0]}, index=[15, 16, 17, 18])
        dataset_1: PandasDataset[pd.DataFrame, pd.DataFrame] = PandasDataset(
            features_1, labels_1
        )
        dataset_2: PandasDataset[pd.DataFrame, pd.DataFrame] = PandasDataset(
            features_2, labels_2
        )
        dataset_3: PandasDataset[pd.DataFrame, pd.DataFrame] = PandasDataset(
            features_3, labels_3
        )
        dataset_merged = PandasDataset.from_sequence([dataset_1, dataset_2, dataset_3])
        self.assertEqual(
            pd.concat([features_1, features_2, features_3]), dataset_merged.features
        )
        self.assertEqual(
            pd.concat([labels_1, labels_2, labels_3]), dataset_merged.labels
        )

    @logTest
    def test_serialization(self) -> None:
        filename = os.path.join(TMP_FOLDER, "my_dataset.p")

        self.dataset.write(filename)

        newDataset: PandasDataset[pd.DataFrame, pd.Series] = PandasDataset.load(
            filename
        )

        self.assertTrue(isinstance(newDataset, PandasDataset))
        self.assertTrue(
            (self.dataset.features.fillna("NaN") == newDataset.features.fillna("NaN"))
            .all()
            .all()
        )

    @logTest
    def test_creation_from_samples(self) -> None:
        samples = [
            Sample(features=[100, 101], label=1, name=1),
            Sample(features=[102, 103], label=2, name=2),
            Sample(features=[104, 105], label=1, name=3),
            Sample(features=[106, 107], label=2, name=4),
            Sample(features=[108, 109], label=2, name=5),
            Sample(features=[110, 111], label=2, name=6),
            Sample(features=[112, 113], label=1, name=7),
            Sample(features=[114, 115], label=2, name=8),
            Sample(features=[116, 117], label=2, name=9),
        ]

        lazyDataset: LazyDataset[Any, int] = CachedDataset(samples).filter(
            lambda x: cast(int, x.label) <= 5
        )

        self.assertIsInstance(lazyDataset, LazyDataset)

        for format in ["pandas", "array", "dict"]:
            features1 = lazyDataset.getFeaturesAs(cast(AllowedTypes, format))
            labels1 = lazyDataset.getLabelsAs(cast(AllowedTypes, format))

            cached: CachedDataset[List[int], int] = lazyDataset.to_cached()

            features2 = cached.getFeaturesAs(cast(AllowedTypes, format))
            labels2 = cached.getLabelsAs(cast(AllowedTypes, format))

            self.assertEqual(features1, features2)
            self.assertEqual(labels1, labels2)

            pandasDataset = cached.asPandasDataset

            features3 = pandasDataset.getFeaturesAs(cast(AllowedTypes, format))
            labels3 = pandasDataset.getLabelsAs(cast(AllowedTypes, format))

            self.assertEqual(features1, features3)
            self.assertEqual(labels1, labels3)

    @logTest
    def test_union(self) -> None:
        union = self.dataset.union(
            PandasDataset(  # type: ignore
                features=pd.concat(
                    [
                        pd.Series([np.nan, 5, 6, 7], name="feat1"),
                        pd.Series([7, 8, 9, 10], name="feat2"),
                    ],
                    axis=1,
                ),
                labels=pd.Series([0, 0, 0, 1], name="Label"),
            )
        )

        self.assertIsInstance(union, PandasDataset)
        self.assertEqual(
            union.features.reset_index(drop=True),
            pd.concat(
                [
                    pd.Series([1, np.nan, 2, 3, np.nan, 5, 6, 7], name="feat1"),
                    pd.Series([1, 2, 3, 4, 7, 8, 9, 10], name="feat2"),
                ],
                axis=1,
            ),
        )
        self.assertEqual(
            union.labels.Label.reset_index(drop=True),
            pd.Series([0, 0, 0, 1, 0, 0, 0, 1], name="Label"),
        )
        self.assertRaisesRegex(
            TypeError,
            "other's type (.*) is different from self's type (.*)",
            self.dataset.union,
            CachedDataset(samples),
        )

    @logTest
    def test_intersection(self) -> None:
        other: PandasDataset[pd.DataFrame, pd.Series] = PandasDataset(
            features=pd.concat(
                [
                    pd.Series([1, 2, 3, 4], name="feat1"),
                    pd.Series([5, 6, 7, 8], name="feat2"),
                ],
                axis=1,
            ),
            labels=pd.Series([1, 1, 0, 0], name="Label", index=[0, 1, 4, 5]),
        )

        self.assertEqual(other.intersection().labels.index.to_list(), [0, 1])
        self.assertEqual(other.intersection().features.index.to_list(), [0, 1])

    @logTest
    def test_getFeaturesAs(self) -> None:
        self.assertTrue(isinstance(self.dataset.getFeaturesAs("array"), np.ndarray))
        self.assertTrue(isinstance(self.dataset.getFeaturesAs("pandas"), pd.DataFrame))
        self.assertTrue(isinstance(self.dataset.getFeaturesAs("dict"), dict))

    @logTest
    def test_getLabelsAs(self) -> None:
        self.assertTrue(isinstance(self.dataset.getLabelsAs("array"), np.ndarray))
        self.assertTrue(isinstance(self.dataset.getLabelsAs("pandas"), pd.DataFrame))
        self.assertTrue(isinstance(self.dataset.getLabelsAs("dict"), dict))


class PandasTimeIndexedDatasetTests(TestCase):
    dates = pd.date_range("2010-01-01", "2010-01-04")

    dateStr = [str(x) for x in dates]

    dataset: PandasTimeIndexedDataset[List[float], None] = PandasTimeIndexedDataset(
        features=pd.concat(
            [
                pd.Series([1, np.nan, 2, 3], index=dateStr, name="feat1"),
                pd.Series([1, 2, 3, 4], index=dateStr, name="feat2"),
            ],
            axis=1,
        )
    )

    @classmethod
    def setUpClass(cls) -> None:
        create_dir_if_not_exists(TMP_FOLDER)

    @classmethod
    def tearDownClass(cls) -> None:
        rmtree(TMP_FOLDER)

    @logTest
    def test_time_index(self) -> None:
        # duck-typing check
        days = [x.day for x in self.dataset.features.index]

        self.assertTrue(set(days), set(range(4)))

    @logTest
    def test_serialization(self) -> None:
        filename = os.path.join(TMP_FOLDER, "my_dataset.p")

        self.dataset.write(filename)

        newDataset: PandasTimeIndexedDataset[List[float], None] = type(
            self.dataset
        ).load(filename)

        self.assertTrue(isinstance(newDataset, PandasTimeIndexedDataset))
        self.assertTrue(
            (self.dataset.features.fillna("NaN") == newDataset.features.fillna("NaN"))
            .all()
            .all()
        )

    @logTest
    def test_createObject(self) -> None:
        NewDataset = self.dataset.createObject(
            features=pd.concat(
                [
                    pd.Series([1, 3], index=self.dateStr[0:2], name="feat1"),
                    pd.Series([1, 2], index=self.dateStr[0:2], name="feat2"),
                ],
                axis=1,
            ),
            labels=pd.Series([0, 0], index=self.dateStr[0:2], name="Label"),
        )

        self.assertTrue(isinstance(NewDataset, PandasTimeIndexedDataset))
        self.assertTrue(
            (
                NewDataset.features
                == pd.concat(
                    [
                        pd.Series(
                            [1, 3],
                            index=map(pd.to_datetime, self.dateStr[0:2]),
                            name="feat1",
                        ),
                        pd.Series(
                            [1, 2],
                            index=map(pd.to_datetime, self.dateStr[0:2]),
                            name="feat2",
                        ),
                    ],
                    axis=1,
                )
            )
            .all()
            .all()
        )
        self.assertTrue(
            (
                NewDataset.labels.values
                == pd.Series([0, 0], index=self.dateStr[0:2], name="Label").values
            ).all()
        )

    @logTest
    def test_loc(self) -> None:
        new_dataset = self.dataset.loc(
            [x for x in pd.date_range("2010-01-01", "2010-01-02")]
        )
        to_check: PandasTimeIndexedDataset[
            List[float], None
        ] = PandasTimeIndexedDataset(
            features=pd.DataFrame(self.dataset.features.iloc[:2])
        )
        self.assertIsInstance(new_dataset, PandasTimeIndexedDataset)
        self.assertEqual(new_dataset.features, to_check.features)


if __name__ == "__main__":
    unittest.main()
