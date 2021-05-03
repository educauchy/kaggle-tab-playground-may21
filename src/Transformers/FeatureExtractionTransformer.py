from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from scipy.stats import boxcox


class FeatureExtractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        for col_name in self.X.columns:
            self.X[col_name + '_Log'] = np.nan_to_num(np.log10(self.X[col_name] + 1).astype(np.float32))
        self.X.replace([-np.inf, np.inf], 0, inplace=True)
        return self.X
