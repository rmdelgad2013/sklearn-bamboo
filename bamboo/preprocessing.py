from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, Normalizer,
                                   MaxAbsScaler, RobustScaler, Binarizer, QuantileTransformer)

NUMPY_NUMERIC_DTYPES = ('int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64')

# ToDo: Investigate and resolve this issue:
#   RuntimeError: scikit-learn estimators should always specify their parameters in the signature of their __init__
#   (no varargs). <class '__main__.MinMaxScalerDF'> with constructor (self, subset_columns=(), *args, **kwargs) doesn't  follow this convention.

class NumericRescalerDF(BaseEstimator, TransformerMixin):

    _transformer_class = None

    def __init__(self, subset_columns=(), *args, **kwargs):
        self.subset_columns = subset_columns
        self._transformer = self._transformer_class(*args, **kwargs)

    def _is_all_numeric(self, X):
        return set(X[self.subset_columns].select_dtypes(include=NUMPY_NUMERIC_DTYPES).columns) == set(self.subset_columns)

    def fit(self, X, y=None):
        if len(self.subset_columns) < 1:
            self.subset_columns = X.columns
        subset_df = X[self.subset_columns]
        if not self._is_all_numeric(subset_df):
            raise TypeError('The columns to transform must all be numeric.')

        self._transformer.fit(subset_df.values)
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_copy[self.subset_columns] = self._transformer.transform(X_copy[self.subset_columns].values)
        return X_copy

    def inverse_transform(self, X):
        X_copy = X.copy()
        X_copy[self.subset_columns] = self._transformer.inverse_transform(X_copy[self.subset_columns].values)
        return X_copy

# ToDo: Maybe look into a class factory function.

class MinMaxScalerDF(NumericRescalerDF):
    _transformer_class = MinMaxScaler

class StandardScalerDF(NumericRescalerDF):
    _transformer_class = StandardScaler

class NormalizerDF(NumericRescalerDF):
    _transformer_class = Normalizer

class MaxAbsScalerDF(NumericRescalerDF):
    _transformer_class = MaxAbsScaler

class RobustScalerDF(NumericRescalerDF):
    _transformer_class = RobustScaler

class BinarizerDF(NumericRescalerDF):
    _transformer_class = Binarizer

class QuantileTransformerDF(NumericRescalerDF):
    _transformer_class = QuantileTransformer

