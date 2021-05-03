from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier


class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_features: int = 10, columns: list = None):
        super().__init__()
        self.columns = columns
        self.n_features = n_features
        knn = KNeighborsClassifier(n_neighbors=3)
        self.selector = SequentialFeatureSelector(knn, n_features_to_select=n_features)

    def __select_features(X, scores, n_best=10):
        X_a = X.append(pd.Series(scores), ignore_index=True)
        X_a = X_a.sort_values(by=X.shape[0], axis=1, ascending=False, na_position='last')
        X_a = X_a.iloc[:X_a.shape[0] - 1, :n_best]
        X_a = np.nan_to_num(X_a.astype(np.float32))
        return X_a

    def fit(self, X, y=None):
        self.selector.fit(X, y)
        return self

    def transform(self, X, y=None):
        self.X = X.copy()
        self.X = self.selector.transform(self.X, y)
        return self.X