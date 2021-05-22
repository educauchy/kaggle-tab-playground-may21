from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from itertools import permutations

class FeatureExtractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, include_log=True, include_perm=False, log_base=10, perm_level=2, columns=None):
        super().__init__()
        self.include_log = include_log
        self.include_perm = include_perm
        self.log_base = log_base
        self.perm_level = perm_level
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        X_columns = self.X.columns if self.columns is None else self.columns

        if self.include_perm:
            train_cols_perm_2 = permutations(X_columns, self.perm_level)
            for feature1, feature2 in train_cols_perm_2:
                self.X[feature1 + '+' + feature2] = self.X[feature1] + self.X[feature2]
                self.X[feature1 + '-' + feature2] = self.X[feature1] - self.X[feature2]
                self.X[feature1 + '*' + feature2] = self.X[feature1] * self.X[feature2]

        if self.include_log:
            for col_name in X_columns:
                self.X[col_name + '_Log'] = np.log(self.X[col_name] - min(self.X[col_name]) + 1).astype('float32') / np.log(self.log_base)
        return self.X
