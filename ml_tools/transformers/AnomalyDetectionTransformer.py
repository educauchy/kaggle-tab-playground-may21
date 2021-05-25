from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from ml_tools.helpers import Logging


class AnomalyDetectionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = 'isoforest',
                        verbose: bool = False,
                        columns: list = None,
                        **params):
        super().__init__()
        self.method = method
        self.verbose = verbose
        self.detectors = {
            'isoforest': IsolationForest,
            'lof': LocalOutlierFactor,
            'onesvm': OneClassSVM,
        }
        self.columns = columns
        self.detector = self.detectors[method](**params)

    @Logging.logging_output('anomaly')
    def fit(self, X, y=None):
        self.X = X.copy()
        self.X['Is_Anomaly'] = self.detector.fit_predict(self.X)
        if self.verbose:
            print('Anomalies found: ' + str(self.X[self.X.Is_Anomaly == -1].shape[0]))
        return self

    def transform(self, X, y=None):
        return X

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
