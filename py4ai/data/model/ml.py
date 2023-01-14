"""Module for specifying data-models to be used in modelling."""

import sys
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from py4ai.core.types import T
from py4ai.core.utils.decorators import lazyproperty as lazy
from py4ai.core.utils.decorators import same_type
from py4ai.core.utils.pandas import loc
from typing_extensions import Literal

from py4ai.data.model.core import (
    CachedIterable,
    DillSerialization,
    IterableUtilsMixin,
    IterGenerator,
    LazyIterable,
    RegisterLazyCachedIterables,
)

if sys.version_info[0] < 3:
    from itertools import islice
    from itertools import izip as zip
else:
    from itertools import islice

TPandasDataset = TypeVar("TPandasDataset", bound="PandasDataset")  # type: ignore
TDatasetUtilsMixin = TypeVar("TDatasetUtilsMixin", bound="DatasetUtilsMixin")  # type: ignore

FeatType = TypeVar(
    "FeatType",
    bound=Union[List[Any], Tuple[Any], np.ndarray[Any, np.dtype[Any]], Dict[str, Any]],
)
LabType = TypeVar("LabType", int, float, None)
FeaturesType = Union[
    np.ndarray,
    pd.DataFrame,
    Dict[Union[str, int], FeatType],
    List[FeatType],
    Iterator[FeatType],
]
LabelsType = Union[
    None,
    np.ndarray,
    pd.DataFrame,
    Dict[Union[str, int], LabType],
    List[LabType],
    Iterator[LabType],
]
AllowedTypes = Literal["array", "pandas", "dict", "list", "lazy"]


def features_and_labels_to_dataset(
    X: Union[pd.DataFrame, pd.Series],
    y: Optional[Union[pd.DataFrame, pd.Series]] = None,
) -> "CachedDataset[Dict[Any, Any], int]":
    """
    Pack features and labels into a CachedDataset.

    :param X: features which can be a pandas dataframe or a pandas series object
    :param y: labels which can be a pandas dataframe or a pandas series object
    :return: an instance of :class:`py4ai.data.model.ml.CachedDataset`

    """
    if y is not None:
        df = pd.concat({"features": X, "labels": y}, axis=1)
        return CachedDataset(
            [
                Sample(
                    df["features"].loc[i].to_dict(), df["labels"].loc[i].to_dict(), i
                )
                for i in df.index
            ]
        )
    else:
        df = pd.concat({"features": X}, axis=1)
        return CachedDataset(
            [Sample(df["features"].loc[i].to_dict(), None, i) for i in df.index]
        )


class Sample(DillSerialization, Generic[FeatType, LabType]):
    """Base class for representing a sample/observation."""

    def __init__(
        self,
        features: FeatType,
        label: Optional[LabType] = None,
        name: Optional[Union[int, str, Any]] = None,
    ) -> None:
        """
        Return an object representing a single sample of a training or test set.

        :param features: features of the sample
        :param label: labels of the sample (optional)
        :param name: id of the sample (optional)
        """
        self.features: FeatType = features
        self.label: Optional[LabType] = label
        self.name: Optional[Union[str, int, Any]] = name


class MultiFeatureSample(Sample[List[np.ndarray[Any, Any]], LabType]):
    """Class representing an observation defined by a nested list of arrays."""

    @staticmethod
    def _check_features(features: List[np.ndarray[Any, Any]]) -> None:
        """
        Check that features is list of lists.

        :param features: list of lists
        :raises TypeError: if features is not a list or one of the feature content is not a numpy array
        """
        if not isinstance(features, list):
            raise TypeError("features must be a list")

        for f in features:
            if not isinstance(f, np.ndarray):
                raise TypeError("all features elements must be np.ndarrays")

    def __init__(
        self,
        features: List[np.ndarray[Any, Any]],
        label: Optional[LabType] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Object representing a single sample of a training or test set.

        :param features: features of the sample
        :param label: labels of the sample (optional)
        :param name: id of the sample (optional)

        """
        self._check_features(features)
        super(MultiFeatureSample, self).__init__(features, label, name)


class DatasetUtilsMixin(
    IterableUtilsMixin[
        Sample[FeatType, LabType],
        "LazyDataset[FeatType, LabType]",
        "CachedDataset[FeatType, LabType]",
    ],
    Generic[FeatType, LabType],
    ABC,
):
    """Base class for representing datasets as iterable over Samples."""

    @property
    def type(self) -> Type[Sample[FeatType, LabType]]:
        """
        Return the type of the objects in the Iterable.

        :return: type of the object of the iterable
        """
        return Sample[FeatType, LabType]

    @staticmethod
    def checkNames(x: Optional[Union[str, int, Any]]) -> Union[str, int]:
        """
        Check that feature names comply with format and cast them to either string or int.

        :param x: feature name
        :return: name as int or str
        :raises AttributeError: if x is none
        """
        if x is None:
            raise AttributeError("With type 'dict' all samples must have a name")
        else:
            return x if isinstance(x, int) else str(x)

    @overload
    def getFeaturesAs(self, type: Literal["array"]) -> np.ndarray[Any, Any]:
        ...

    @overload
    def getFeaturesAs(self, type: Literal["pandas"]) -> pd.DataFrame:
        ...

    @overload
    def getFeaturesAs(self, type: Literal["dict"]) -> Dict[Union[str, int], FeatType]:
        ...

    @overload
    def getFeaturesAs(self, type: Literal["list"]) -> List[FeatType]:
        ...

    @overload
    def getFeaturesAs(self, type: Literal["lazy"]) -> Iterator[FeatType]:
        ...

    def getFeaturesAs(self, type: AllowedTypes = "array") -> FeaturesType[FeatType]:
        """
        Return object of the specified type containing the feature space.

        :param type: type of return. Can be one of "pandas", "dict", "list" or "array
        :return: an object of the specified type containing the features
        :raises ValueError: if the provided type is not one of the allowed ones
        """
        if type == "array":
            return np.array([sample.features for sample in self])
        elif type == "dict":
            return {self.checkNames(sample.name): sample.features for sample in self}
        elif type == "list":
            return [sample.features for sample in self]
        elif type == "lazy":
            return (sample.features for sample in self)
        elif type == "pandas":
            try:
                features: Union[
                    Dict[Union[str, int], FeatType], List[FeatType]
                ] = self.getFeaturesAs("dict")
                try:
                    return pd.DataFrame(features).T
                except ValueError:
                    return pd.Series(features).to_frame("features")
            except AttributeError:
                features = self.getFeaturesAs("list")
                try:
                    return pd.DataFrame(features)
                except ValueError:
                    return pd.Series(features).to_frame("features")

        else:
            raise ValueError(f"Type {type} not allowed")

    @overload
    def getLabelsAs(self, type: Literal["array"]) -> np.ndarray[Any, Any]:
        ...

    @overload
    def getLabelsAs(self, type: Literal["pandas"]) -> pd.DataFrame:
        ...

    @overload
    def getLabelsAs(self, type: Literal["dict"]) -> Dict[Union[str, int], LabType]:
        ...

    @overload
    def getLabelsAs(self, type: Literal["list"]) -> List[LabType]:
        ...

    @overload
    def getLabelsAs(self, type: Literal["lazy"]) -> Iterator[LabType]:
        ...

    def getLabelsAs(self, type: AllowedTypes = "array") -> LabelsType[LabType]:
        """
        Return an object of the specified type containing the labels.

        :param type: type of return. Can be one of "pandas", "dict", "list" or "array
        :return: an object of the specified type containing the features
        :raises ValueError: if the provided type is not one of the allowed ones
        """
        if type == "array":
            return np.array([sample.label for sample in self])
        elif type == "dict":
            return {self.checkNames(sample.name): sample.label for sample in self}
        elif type == "list":
            return [sample.label for sample in self]
        elif type == "lazy":
            return (sample.label for sample in self)
        elif type == "pandas":
            try:
                labels: Union[
                    List[LabType], Dict[Union[str, int], LabType]
                ] = self.getLabelsAs("dict")
                try:
                    return pd.DataFrame(labels).T
                except ValueError:
                    return pd.Series(labels).to_frame("labels")
            except AttributeError:
                labels = self.getLabelsAs("list")
                try:
                    return pd.DataFrame(labels)
                except ValueError:
                    return pd.Series(labels).to_frame("labels")
        else:
            raise ValueError(f"Type {type} not allowed")

    @abstractmethod
    def union(
        self, other: TDatasetUtilsMixin
    ) -> "DatasetUtilsMixin[FeatType, LabType]":
        """
        Return a union of datasets.

        :param other: other dataset to join
        :return: union dataset
        """
        raise NotImplementedError

    @property
    def asPandasDataset(self) -> "PandasDataset[FeatType, LabType]":
        """
        Cast object as a PandasDataset.

        :return: dataset
        """
        return PandasDataset(self.getFeaturesAs("pandas"), self.getLabelsAs("pandas"))


class CachedDataset(
    DatasetUtilsMixin[FeatType, LabType],
    CachedIterable[Sample[FeatType, LabType]],
    DillSerialization,
):
    """Class that represents dataset cached in-memory, derived by a cached iterables of samples."""

    def to_df(self) -> pd.DataFrame:
        """
        Reformat the Features and Labels as a DataFrame.

        :return: DataFrame, Dataframe with features and labels
        """
        return pd.concat(
            {
                "features": self.getFeaturesAs("pandas"),
                "labels": self.getLabelsAs("pandas"),
            },
            axis=1,
        )

    def union(self, other: TDatasetUtilsMixin) -> "CachedDataset[FeatType, LabType]":
        """
        Perform union on CachedDatasets.

        :param other: CachedDataset
        :return: union of current and other CachedDataset
        """
        return CachedDataset([x for x in self.items] + [x for x in other.items])


@RegisterLazyCachedIterables(CachedDataset)
class LazyDataset(
    LazyIterable[Sample[FeatType, LabType]], DatasetUtilsMixin[FeatType, LabType]
):
    """Class that represents dataset derived by a lazy iterable of samples."""

    def withLookback(self, lookback: int) -> "LazyDataset[FeatType, LabType]":
        """
        Create a LazyDataset with features that are an array of ``lookback`` lists of samples' features.

        :param lookback: number of samples' features to look at
        :return: ``LazyDataset`` with changed samples
        """

        def _transformed_sample_generator() -> Iterator[Sample[FeatType, LabType]]:
            slices = [islice(self, n, None) for n in range(lookback)]
            for ss in zip(*slices):
                yield Sample(
                    features=cast(
                        FeatType, np.array([s.features for s in ss], dtype=object)
                    ),
                    label=ss[-1].label,
                )

        return LazyDataset(IterGenerator(_transformed_sample_generator))

    def features(self) -> Iterator[FeatType]:
        """
        Return an iterator over sample features.

        :return: iterable of features
        """
        return self.getFeaturesAs("lazy")

    def labels(self) -> Iterator[LabType]:
        """
        Return an iterator over sample labels.

        :return: iterable of labels
        """
        return self.getLabelsAs("lazy")

    @overload
    def getFeaturesAs(self, type: Literal["array"]) -> np.ndarray[Any, Any]:
        ...

    @overload
    def getFeaturesAs(self, type: Literal["pandas"]) -> pd.DataFrame:
        ...

    @overload
    def getFeaturesAs(self, type: Literal["dict"]) -> Dict[Union[str, int], FeatType]:
        ...

    @overload
    def getFeaturesAs(self, type: Literal["list"]) -> List[FeatType]:
        ...

    @overload
    def getFeaturesAs(self, type: Literal["lazy"]) -> Iterator[FeatType]:
        ...

    def getFeaturesAs(self, type: AllowedTypes = "lazy") -> FeaturesType[FeatType]:
        """
        Return object of the specified type containing the feature space.

        :param type: type of return. Can be one of "pandas", "dict", "list" or "array
        :return: an object of the specified type containing the features
        """
        return super(LazyDataset, self).getFeaturesAs(type)

    @overload
    def getLabelsAs(self, type: Literal["array"]) -> np.ndarray[Any, Any]:
        ...

    @overload
    def getLabelsAs(self, type: Literal["pandas"]) -> pd.DataFrame:
        ...

    @overload
    def getLabelsAs(self, type: Literal["dict"]) -> Dict[Union[str, int], LabType]:
        ...

    @overload
    def getLabelsAs(self, type: Literal["list"]) -> List[LabType]:
        ...

    @overload
    def getLabelsAs(self, type: Literal["lazy"]) -> Iterator[LabType]:
        ...

    def getLabelsAs(self, type: AllowedTypes = "lazy") -> LabelsType[LabType]:
        """
        Return an object of the specified type containing the labels.

        :param type: type of return. Can be one of "pandas", "dict", "list", "array" or iterators
        :return: an object of the specified type containing the features
        """
        return super(LazyDataset, self).getLabelsAs(type)

    def union(self, other: TDatasetUtilsMixin) -> "LazyDataset[FeatType, LabType]":
        """
        Perform union on LazyDatasets.

        :param other: LazyDataset
        :return: union of LazyDatasets
        """

        def _generator() -> Iterator[Sample[FeatType, LabType]]:
            for sample in self:
                yield sample
            for sample in other:
                yield sample

        return LazyDataset(IterGenerator(_generator))


@RegisterLazyCachedIterables(LazyDataset, unidirectional_link=True)
class PandasDataset(
    Generic[FeatType, LabType], DatasetUtilsMixin[FeatType, LabType], DillSerialization
):
    """Dataset represented via pandas Dataframes for features and labels."""

    def __init__(
        self,
        features: Union[DataFrame, Series],
        labels: Optional[Union[DataFrame, Series]] = None,
    ) -> None:
        """
        Return a datastructure built on top of pandas dataframes.

        The PandasDataFrame allows to pack features and labels together and obtain features and labels  as a pandas
        dataframe, numpy array or a dictionary. For unsupervised learning tasks the labels are left as None.

        :param features: a dataframe or a series of features
        :param labels: a dataframe or a series of labels. None in case no labels are present.
        :raises TypeError: if the labels or features are not DataFrames nor Series
        """
        if isinstance(features, pd.Series):
            self._features = features.to_frame()
        elif isinstance(features, pd.DataFrame):
            self._features = features
        else:
            raise TypeError(
                "Features must be of type pandas.Series or pandas.DataFrame"
            )

        if isinstance(labels, pd.Series):
            self._labels = labels.to_frame()
        elif isinstance(labels, pd.DataFrame):
            self._labels = labels
        elif labels is None:
            self._labels = pd.DataFrame(
                labels, index=self._features.index, columns=[None]
            )
        else:
            raise TypeError(
                "Labels must be of type pandas.Series or pandas.DataFrame or None"
            )

    @property
    def items(self) -> Iterator[Sample[FeatType, LabType]]:
        """
        Get features as an iterator of Samples.

        :yield: Iterator of objects of :class:`py4ai.data.model.ml.Sample`
        """
        for index, row in dict(self._features.to_dict(orient="index")).items():
            try:
                yield Sample(
                    name=index,
                    features=row,
                    label=self._labels.loc[index] if self._labels is not None else None,
                )
            except AttributeError:
                yield Sample(name=index, features=row, label=None)

    @property
    def cached(self) -> bool:
        """
        Return whether the dataset is cached or not in memory.

        :return: boolean
        """
        return True

    @lazy
    def features(self) -> pd.DataFrame:
        """
        Get features as pandas dataframe.

        :return: pd.DataFrame
        """
        return self.getFeaturesAs("pandas")

    @lazy
    def labels(self) -> pd.DataFrame:
        """
        Get labels as a pandas dataframe.

        :return: pd.DataFrame
        """
        return self.getLabelsAs("pandas")

    @property
    def index(self) -> pd.Index:
        """
        Get Dataset index.

        :return: pd.Index
        """
        return self.intersection().features.index

    @staticmethod
    def _check_none(lab: Optional[T]) -> Optional[T]:
        """
        Check whether the label is none (unsupervised or prediction) or not (training).

        :param lab: label to check
        :return: label itself
        """
        return lab if lab is not None else None

    @classmethod
    def empty(cls: Type[TPandasDataset]) -> TPandasDataset:  # type: ignore
        """Return empty object.

        :return: Empty instance of class
        """
        return cls(pd.DataFrame(), pd.DataFrame())

    @classmethod
    def createObject(
        cls: Type[TPandasDataset],
        features: Union[pd.DataFrame, pd.Series],
        labels: Optional[Union[pd.DataFrame, pd.Series]],
    ) -> TPandasDataset:
        """
        Create a PandasDataset object.

        :param features: features as pandas dataframe/series
        :param labels: labels as pandas dataframe/series
        :return: a ``PandasDataset`` object
        """
        return cls(features, labels)

    def __len__(self) -> int:
        """
        Get number of records in the dataset.

        :return: int, length of the dataset
        """
        return len(self.index)

    def takeAsPandas(self: TPandasDataset, n: int) -> TPandasDataset:
        """
        Return top n records as a PandasDataset.

        :param n: int specifying number of records to output
        :return: ``PandasDataset`` of length n
        """
        idx = (
            list(self.features.index.intersection(self.labels.index))
            if self.labels is not None
            else list(self.features.index)
        )
        return self.loc(idx[:n])

    def loc(self: TPandasDataset, idx: List[Any]) -> TPandasDataset:
        """
        Find given indices in features and labels.

        :param idx: input indices
        :return: PandasDataset with features and labels filtered on input indices
        """
        features = loc(self.features, idx)
        labels = self.labels.loc[idx] if self.labels is not None else None
        return self.createObject(features, labels)

    def dropna(self: TPandasDataset, **kwargs: Any) -> TPandasDataset:
        """
        Drop NAs from feature and labels.

        :param kwargs: keyworded arguments are passed to dropna
        :return: ``PandasDataset`` with features and labels without NAs
        """
        kwargs_feat = {
            (k.split("__")[1] if k.startswith("feat__") else k): v
            for k, v in kwargs.items()
            if not k.startswith("labs__")
        }
        kwargs_labs = {
            k.split("__")[1]: v for k, v in kwargs.items() if k.startswith("labs__")
        }

        return self.createObject(
            self.features.dropna(**kwargs_feat),
            self._check_none(
                self.labels.dropna(**kwargs_labs) if self.labels is not None else None
            ),
        )

    def intersection(self: TPandasDataset) -> TPandasDataset:
        """
        Intersect feature and labels indices.

        :return: ``PandasDataset`` with features and labels with intersected indices
        """
        idx = (
            list(self.features.index.intersection(self.labels.index))
            if self.labels is not None
            else list(self.features.index)
        )
        return self.loc(idx)

    @overload
    def getFeaturesAs(self, type: Literal["array"]) -> np.ndarray[Any, Any]:
        ...

    @overload
    def getFeaturesAs(self, type: Literal["pandas"]) -> pd.DataFrame:
        ...

    @overload
    def getFeaturesAs(self, type: Literal["dict"]) -> Dict[Union[str, int], FeatType]:
        ...

    @overload
    def getFeaturesAs(self, type: Literal["list"]) -> List[FeatType]:
        ...

    @overload
    def getFeaturesAs(self, type: Literal["lazy"]) -> Iterator[FeatType]:
        ...

    def getFeaturesAs(self, type: AllowedTypes = "array") -> FeaturesType[FeatType]:
        """
        Get features as numpy array, pandas dataframe or dictionary.

        :param type: str, default is 'array', can be 'array','pandas','dict'
        :return: features according to the given type
        :raises ValueError: provided type not allowed
        """
        if type == "array":
            return np.array(self._features)
        elif type == "pandas":
            return self._features
        elif type == "dict":
            return {
                self.checkNames(k): list(row) for k, row in self._features.iterrows()
            }
        else:
            raise ValueError(
                f'"type" value "{type}" not allowed. Only allowed values for "type" are "array", "dict" or '
                f'"pandas"'
            )

    @overload
    def getLabelsAs(self, type: Literal["array"]) -> np.ndarray[Any, Any]:
        ...

    @overload
    def getLabelsAs(self, type: Literal["pandas"]) -> pd.DataFrame:
        ...

    @overload
    def getLabelsAs(self, type: Literal["dict"]) -> Dict[Union[str, int], LabType]:
        ...

    @overload
    def getLabelsAs(self, type: Literal["list"]) -> List[LabType]:
        ...

    @overload
    def getLabelsAs(self, type: Literal["lazy"]) -> Iterator[LabType]:
        ...

    def getLabelsAs(self, type: AllowedTypes = "array") -> LabelsType[LabType]:
        """
        Get labels as numpy array, pandas dataframe or dictionary.

        :param type: str, default is 'array', can be 'array','pandas','dict'
        :return: labels according to the given type
        :raises ValueError: provided type not allowed
        """
        if self._labels is None:
            return None
        elif isinstance(self._labels, pd.DataFrame):
            if type == "array":
                nCols = len(self._labels.columns)
                return (
                    np.array(self._labels)
                    if nCols > 1
                    else np.array(self._labels[self._labels.columns[0]])
                )
            elif type == "pandas":
                return self._labels
            elif type == "dict":
                nCols = len(self._labels.columns)
                return (
                    dict(self._labels.to_dict(orient="index"))
                    if nCols > 1
                    else self._labels[self._labels.columns[0]].to_dict()
                )
            else:
                raise ValueError(
                    f'"type" value "{type}" not allowed. Only allowed values for "type" are "array", "dict" or '
                    f'"pandas"'
                )
        else:
            raise ValueError("type of labels not allowed for this function")

    @classmethod
    def from_sequence(
        cls: Type[TPandasDataset], datasets: Sequence[TPandasDataset]
    ) -> TPandasDataset:
        """
        Create a PandasDataset from a list of pandas datasets using pd.concat.

        :param datasets: list of PandasDatasets
        :return: ``PandasDataset``
        """
        features_iter, labels_iter = zip(
            *[(dataset.features, dataset.labels) for dataset in datasets]
        )
        labels = (
            None
            if all([lab is None for lab in labels_iter])
            else pd.concat(labels_iter)
        )
        features = pd.concat(features_iter)
        return cls.createObject(features, labels)

    @same_type
    def union(self, other: TPandasDataset) -> TPandasDataset:
        """
        Return a union between PandasDatasets.

        :param other: Dataset to be merged
        :return: Dataset resulting from the merge
        """
        features = pd.concat([self.features, other.features])
        labels = (
            pd.concat([self.labels, other.labels])
            if not (self.labels is None and other.labels is None)
            else None
        )
        return cast(TPandasDataset, self.createObject(features, labels))


class PandasTimeIndexedDataset(
    PandasDataset[FeatType, LabType], Generic[FeatType, LabType]
):
    """Class to be used for datasets that have time-indexed samples."""

    def __init__(
        self,
        features: Union[pd.DataFrame, pd.Series],
        labels: Optional[Union[pd.DataFrame, pd.Series]] = None,
    ) -> None:
        """
        Return a datastructure built on top of pandas dataframes that allows to pack features and labels that are time indexed.

        Features and labels can be obtained as a pandas dataframe, numpy array or a dictionary.
        For unsupervised learning tasks the labels are left as None.

        :param features: pandas dataframe/series where index elements are dates in string format
        :param labels: pandas dataframe/series where index elements are dates in string format
        """
        super(PandasTimeIndexedDataset, self).__init__(features, labels)
        self._features.rename(index=pd.to_datetime, inplace=True)
        if self.labels is not None:
            self._labels.rename(index=pd.to_datetime, inplace=True)
