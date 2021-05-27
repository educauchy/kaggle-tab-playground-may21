from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from typing import List


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: List[str] = None):
        super().__init__()
        self.columns = columns
        if (columns is None or len(columns) == 0):
            warnings.warn('No columns specified')

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        self.X.drop(self.columns, axis=1, inplace=True)
        return self.X
