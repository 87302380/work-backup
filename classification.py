import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import sklearn
from sklearn.metrics import f1_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


x_train = pd.read_csv('./data/x_train.csv').values
y_train = pd.read_csv('./data/y_train.csv').values[:,0]
x_test = pd.read_csv('./data/x_test.csv').values
y_test = pd.read_csv('./data/y_test.csv').values[:,0]


x_train_resampled, y_train_resampled = x_train, y_train

train_data = lgb.Dataset(x_train_resampled, label=y_train_resampled)
test_data = lgb.Dataset(x_test, label=y_test)




print('Start predicting...')

def get_f1_score(model):
    predict = model.predict(x_test, num_iteration=model.best_iteration)
    predict_label = np.argmax(predict, axis=1)

    return f1_score(test_data.get_label(), predict_label, average='micro')

'''
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class':5,
    'metric_freq':1,
    'max_bin':25,
    'num_leaves': 31,
    'learning_rate': 0.1,
    'num_trees': 100,
    'bagging_fraction': 0.8,
    'bagging_freq': 10,
    'feature_fraction': 0.884666745361057,
    'lambda_l2': 2,
    'min_gain_to_split': 0.2143116702471033,
    'scale_pos_weight': 3.201815491344985
}


gbm = lgb.train(params,
                train_data,
                num_boost_round=10,
                valid_sets=test_data,
                early_stopping_rounds=5)
                
                
print(get_f1_score(gbm))

'''

def objective(params):

    num_leaves = int(params['num_leaves'])
    #min_data_in_leaf = int(params['min_data_in_leaf'])
    max_bin = int(params['max_bin'])
    bagging_fraction = params['bagging_fraction']
    bagging_freq = int(params['bagging_freq'])
    feature_fraction = params['feature_fraction']
    lambda_l2 = params['lambda_l2'],
    min_gain_to_split = params['min_gain_to_split']
    scale_pos_weight = params['scale_pos_weight']

    param = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 5,
        'num_leaves': num_leaves,
        #'min_data_in_leaf': min_data_in_leaf,
        'max_bin': max_bin,
        'learning_rate': 0.1,
        'num_trees': 1000,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'feature_fraction': feature_fraction,
        'verbose': -1,
        'lambda_l2': lambda_l2,
        'min_gain_to_split': min_gain_to_split,
        'scale_pos_weight': scale_pos_weight,
             }


    model_lgb = lgb.train(param,
                    train_data,
                    valid_sets=[test_data],
                    early_stopping_rounds=15,
                    )
    score = get_f1_score(model_lgb)
    print(score)

    return 1-score



space = {
         'num_leaves' : hp.quniform('num_leaves', 10, 200, 1),
         #'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 10, 300, 1),
         'max_bin' : hp.quniform('max_bin', 10, 255, 5),
         'bagging_fraction' : hp.uniform('bagging_fraction', 0.1, 1.0), # 0.0 < bagging_fraction <= 1.0
         'bagging_freq' : hp.quniform('bagging_freq', 0, 20, 1),
         'feature_fraction' :  hp.uniform('feature_fraction', 0.01, 1.0), # 0.0 < feature_fraction <= 1.0
         'lambda_l2' : hp.uniform('lambda_l2', 0.0, 20.0),
         'min_gain_to_split' : hp.uniform('min_gain_to_split', 0.0, 1.0),
         'scale_pos_weight' : hp.uniform('scale_pos_weight', 1.0, 10.0),
        }

best = fmin(objective,
    space=space,
    algo=tpe.suggest,
    max_evals = 100)

print(best)


