from ml_tools.transformers import *
from ml_tools.models import *
from ml_tools.helpers import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import log_loss
from sklearn.feature_selection import SelectKBest, mutual_info_classif

import pandas as pd
import numpy as np
import yaml
import os, sys
from shutil import copyfile
from imblearn.over_sampling import SMOTE


try:
    project_dir = os.path.dirname(__file__)
    config_file = os.path.join(project_dir, 'config.yaml')

    with open (config_file, 'r') as file:
        config = yaml.safe_load(file)
except yaml.YAMLError as exc:
    print(exc)
    sys.exit(1)
except Exception as e:
    print('Error reading the config file')
    sys.exit(1)


train = pd.read_csv(config['data']['train'])
test = pd.read_csv(config['data']['test'])

X = train[train.columns[~train.columns.isin([config['data']['target']])]]
X = X.astype('float32')
X.drop(['id'], axis=1, inplace=True)
y = train[config['data']['target']]
y = pd.Series(y.str.split('_', expand=True)[1], dtype=np.int64, name='target')

sm = SMOTE(sampling_strategy='minority', random_state=config['model']['random_state'], k_neighbors=10, n_jobs=-1)
X, y = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=config['data']['train_size'], \
                                                    random_state=config['model']['random_state'])


full_pipeline = Pipeline(steps=[
    ('f_extraction', FeatureExtractionTransformer(log_base=2)),
    ('anomaly', AnomalyDetectionTransformer(method=config['model']['anomaly']['method'], \
                                            random_state=config['model']['random_state'], \
                                            **config['model']['anomaly']['params'])),
    ('model', MetaClassifier(model=config['model']['model']['method'], \
                             random_state=config['model']['random_state'], \
                             params=config['model']['model']['params']
                             )),
])

if config['model']['strategy'] == 'grid':
    if config['model']['grid_random']:
        full_pipeline = RandomizedSearchCV(full_pipeline, config['model']['param_grid'], **config['model']['grid'],\
                                        random_state=config['model']['random_state'])
    else:
        full_pipeline = GridSearchCV(full_pipeline, config['model']['param_grid'], **config['model']['grid'],\
                                        random_state=config['model']['random_state'])
    full_pipeline.fit(X, y)
    print(full_pipeline.best_params_)
    y_pred = full_pipeline.predict_proba(X_test)
    test_score = log_loss(y_test, y_pred)
    print('Log-Loss: ' + str(test_score))
elif config['model']['strategy'] == 'model':
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict_proba(X_test)
    test_score = log_loss(y_test, y_pred)
    print('Log-Loss: ' + str(test_score))


if config['output']['save']:
    out = pd.DataFrame(data={'id': test['id'].astype(int)})
    test.drop(['id'], axis=1, inplace=True)
    out[['Class_1', 'Class_2', 'Class_3', 'Class_4']] = full_pipeline.predict_proba(test)

    output_folder = gen_submit(config, test_score)
    output_path = os.path.join(project_dir, 'data/submissions', output_folder)
    os.mkdir(output_path)

    out.to_csv( os.path.join(output_path, 'output.csv'), index=False)
    copyfile( config_file, os.path.join(output_path, 'config.yaml') )
