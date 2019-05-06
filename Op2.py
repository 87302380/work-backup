import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('./data/bacmen_vs_viral.csv', sep=',')#, skiprows=range(0, 20))

print(data['diagnosis'].unique())

data['label'] = 0
data.loc[data['diagnosis'] == 'BacMen', 'label'] = 1
data.loc[data['diagnosis'] == 'HSV', 'label'] = 2
data.loc[data['diagnosis'] == 'Z. men_enc', 'label'] = 3
data.loc[data['diagnosis'] == 'Ent. men', 'label'] = 4



#lgb.cv()

print(data['label'].unique())

data.drop(['Unnamed: 0'], axis=1, inplace=True)
data.drop(['diagnosis'], axis=1, inplace=True)

# create label
y = np.array(data['label'].tolist())

# create features
#data.drop('label', axis=1, inplace=True)
x = np.array(data.values)

print('x.shape:', x.shape)
print('y.shape:', y.shape)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
}

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2,
                                                    random_state=35,
                                                    stratify=y,
                                                    shuffle=True)

lgb_train = lgb.Dataset(x_train, y_train, free_raw_data=False)

min_merror = float('Inf')
best_params = {}

# params_test1={'max_depth': range(3,8,1), 'num_leaves':range(2, 100, 1)}
#
# gsearch1 = GridSearchCV(estimator = lgb.LGBMRegressor(
#                                     boosting_type='gbdt',
#                                     objective='regression_l2',
#                                     #metrics='l2',
#                                     max_depth=3,
#                                     num_leaves=10,
#                                     #learning_rate=0.05,
#                                     bagging_fraction=0.1,
#                                     bagging_freq=0,
#                                     feature_fraction=1,
#                                     ),
#                                     param_grid = params_test1,
#                                     scoring="neg_mean_absolute_error",
#                                     cv=3,
#                                     n_jobs=2)
# gsearch1.fit(x_train,y_train)
# print(gsearch1.score, gsearch1.best_params_, gsearch1.best_score_)


# params_test2={'max_bin': range(3,20,1), 'min_data_in_leaf':range(2,20,1)}
#
#
# gsearch2 = GridSearchCV(estimator = lgb.LGBMRegressor(
#                                     boosting_type='gbdt',
#                                     objective='regression_l2',
#                                     metrics='l2',
#                                     max_depth=3,
#                                     num_leaves=2,
#                                     learning_rate=0.1,
#                                     bagging_fraction=0.1,
#                                     bagging_freq=0,
#                                     feature_fraction=1,
#                                     ),
#                                     param_grid = params_test2,
#                                     scoring="neg_mean_absolute_error",
#                                     cv=3,
#                                     n_jobs=2)
# gsearch2.fit(x_train,y_train)
# print(gsearch2.score, gsearch2.best_params_, gsearch2.best_score_)
#
params_test3={'feature_fraction': [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              'bagging_fraction': [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
              'bagging_freq': range(0,50,5)
}

gsearch3 = GridSearchCV(estimator = lgb.LGBMRegressor(
                                    boosting_type='gbdt',
                                    objective='regression_l2',
                                    metrics='l2',
                                    max_depth=3,
                                    num_leaves=2,
                                    max_bin= 5,
                                    min_data_in_leaf=2,
                                    learning_rate=0.1,
                                    bagging_fraction=0.1,
                                    bagging_freq=0,
                                    feature_fraction=1,
                                    ),
                                    param_grid = params_test3,
                                    scoring="neg_mean_absolute_error",
                                    cv=3,
                                    n_jobs=4)
gsearch3.fit(x_train,y_train)
print(gsearch3.score, gsearch3.best_params_, gsearch3.best_score_)
#
# params_test4={'lambda_l1': [0.0,0.1,0.3,0.5,0.7,0.9,1.0],
#               'lambda_l2': [0.0,0.1,0.3,0.5,0.7,0.9,1.0]
# }
#
# gsearch4 = GridSearchCV(estimator = lgb.LGBMRegressor(
#                                     boosting_type='gbdt',
#                                     objective='regression_l2',
#                                     metrics='l2',
#                                     max_depth=3,
#                                     num_leaves=3,
#                                     max_bin= 5,
#                                     min_data_in_leaf=2,
#                                     learning_rate=0.1,
#                                     bagging_fraction=0.1,
#                                     bagging_freq=0,
#                                     feature_fraction=1,
#                                     ),
#                                     param_grid = params_test4,
#                                     scoring="neg_mean_absolute_error",
#                                     cv=3,
#                                     n_jobs=4)
# gsearch4.fit(x_train,y_train)
# print(gsearch4.score, gsearch4.best_params_, gsearch4.best_score_)

# params_test5={'min_split_gain':[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}
#
# gsearch5 = GridSearchCV(estimator = lgb.LGBMRegressor(
#                                     boosting_type='gbdt',
#                                     objective='regression_l2',
#                                     metrics='l2',
#                                     max_depth=3,
#                                     num_leaves=3,
#                                     max_bin= 5,
#                                     min_data_in_leaf=2,
#                                     learning_rate=0.05,
#                                     bagging_fraction=0.1,
#                                     bagging_freq=0,
#                                     feature_fraction=1,
#                                     ),
#                                     param_grid = params_test5,
#                                     scoring="neg_mean_absolute_error",
#                                     cv=3,
#                                     n_jobs=4)
# gsearch5.fit(x_train,y_train)
# print(gsearch5.score, gsearch5.best_params_, gsearch5.best_score_)