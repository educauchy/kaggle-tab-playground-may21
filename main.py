from ml_tools.transformers import *
from ml_tools.models import *
from ml_tools.helpers import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import log_loss

import pandas as pd
import numpy as np
import yaml
import os, sys
import random
from shutil import copyfile


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

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=config['data']['train_size'], \
                                                    random_state=config['model']['random_state'])

preprocess_steps = []
for item in config['model']['encoding']:
    data = all[item['column']] if item['data'] else None
    encoder = EncoderTransformer(type=item['type'], column=item['column'], out_column=item['out_column'], data=data)
    preprocess_steps.append( (item['column'] + '_encoder', encoder) )

preprocess_steps.append( ('log_tr', FunctionTransformer(np.log1p)) )

preprocess_pipeline = Pipeline(steps=preprocess_steps)


full_pipeline = Pipeline(steps=[
    ('f_extraction', FeatureExtractionTransformer()),
    # ('preprocess', preprocess_pipeline),
    # ('anomaly', AnomalyDetectionTransformer(type=config['model']['anomaly']['type'], \
    #                                         **config['model']['anomaly']['params'])),
    ('f_inter', FeatureInteractionTransformer(**config['model']['f_inter']['params'])),
    # ('f_selection', FeatureSelectionTransformer(**config['model']['f_selection']['params'])),
    # ('cluster', ClusteringTransformer(type=config['model']['cluster']['type'], \
    #                                   **config['model']['cluster']['params'])),
    ('model', MetaClassifier(model=config['model']['model']['type'], \
                             random_state=config['model']['random_state'], \
                             **config['model']['model']['params'])),
    # ('model', StackingClassifier(estimators=stacking_estimators, cv=10, final_estimator=LogisticRegression(max_iter=10000), n_jobs=-1)),
])

if config['model']['strategy'] == 'cv':
    cv = KFold(n_splits=config['model']['KFold_folds'], shuffle=True, random_state=config['model']['random_state'])
    scores = cross_val_score(full_pipeline, X, y, cv = cv)
    print('KFold scores:')
    print(scores)
elif config['model']['strategy'] == 'grid_search':
    training = GridSearchCV(full_pipeline, config['model']['param_grid'], scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    training.fit(X, y)
    print(training)
    print(training.best_params_)
    test_score = training.score(X_test, y_test)
    print('Score: ' + str(test_score))
elif config['model']['strategy'] == 'model':
    training = full_pipeline.fit(X_train, y_train)
    y_pred = training.predict_proba(X_test)
    test_score = log_loss(y_test, y_pred)
    print('Log-Loss: ' + str(test_score))


if config['output']['save']:
    out = pd.DataFrame(data={'id': test['id'].astype(int)})
    test.drop(['id'], axis=1, inplace=True)
    out[['Class_1', 'Class_2', 'Class_3', 'Class_4']] = training.predict_proba(test)

    output_folder = gen_submit(config, test_score)
    output_path = os.path.join(project_dir, 'data/submissions', output_folder)
    os.mkdir(output_path)

    out.to_csv( os.path.join(output_path, 'output.csv'), index=False)
    copyfile( config_file, os.path.join(output_path, 'config.yaml') )
