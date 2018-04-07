from inspect import signature

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, Normalizer,
                                   MaxAbsScaler, RobustScaler, Binarizer, QuantileTransformer)

def _extract_init_kwargs(obj):
    '''Helper function to extract the signature from the sklearn Transformer class. This signature is then
    Fed in to the DF extension Transformer class. This is done to handle this error:
    	RuntimeError: scikit-learn estimators should always specify their parameters in the signature of their __init__
    '''
    sig = signature(obj.__init__)
    return {sig.parameters[key].name: sig.parameters[key].default
                for key in sig.parameters.keys()}

NUMPY_NUMERIC_DTYPES = ('int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64')

# ToDo: Write unit tests for common use cases - fit, then transform, fit_transform, pipeline, inverse_transform

class NumericRescalerDF(BaseEstimator, TransformerMixin):

    _transformer_class = None
    _transformer_init_kwargs = _extract_init_kwargs(_transformer_class)

    def __init__(self, subset_columns=(), **_transformer_init_kwargs):
        self.subset_columns = subset_columns
        self._transformer = self._transformer_class(**_transformer_init_kwargs)

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


class OneHotEncoderDF(BaseEstimator, TransformerMixin):
    pass

class PolynomialFeaturesDF(BaseEstimator, TransformerMixin):
    pass