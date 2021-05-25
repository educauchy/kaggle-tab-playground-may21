from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pandas as pd
import numpy as np
from ml_tools.helpers import Logging


class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'kbest', metric: str = 'mutual_info_classif',
                 n_features: int = 10, columns: list = None):
        super().__init__()
        self.columns = columns
        self.n_features = n_features
        self.method = method
        self.metric = metric
        self.scores = []
        self.selectors = {
            'kbest': SelectKBest
        }
        self.metrics = {
            'mutual_info_classif': mutual_info_classif
        }
        self.selector = self.selectors[method](self.metrics[metric], k=n_features)

    @classmethod
    def __select_features(cls, X=None, scores=None, n_best=10):
        X.reset_index(drop=True, inplace=True, )
        scores_pd = pd.DataFrame(data=scores.reshape(1, scores.shape[0]), columns=X.columns)
        X_a = X.append(scores_pd, ignore_index=True)
        X_a = X_a.sort_values(by=X.shape[0], axis=1, ascending=False, na_position='last')
        X_a = X_a.iloc[:X_a.shape[0] - 1, :n_best]
        return X_a

    @Logging.logging_output('selection')
    def fit(self, X, y=None):
        self.selector.fit(X, y)
        self.scores = self.selector.scores_
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        self.X = self.__select_features(self.X, self.scores, self.n_features)
        return self.X