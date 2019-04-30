
# coding: utf-8
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import sklearn.metrics as skm
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from lightgbm.sklearn import LGBMModel
from lightgbm.basic import Booster
import operator
import warnings
import csv
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

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

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size = 0.2,
                                                    random_state=35,
                                                    stratify = y,
                                                    shuffle = True)

lgb_train = lgb.Dataset(x_train, y_train, free_raw_data=False)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train,free_raw_data=False)

params = {
          'boosting_type': 'gbdt',
          'objective': 'regression_l2',
          }
min_merror = float('Inf')
best_params = {}

for num_leaves in range(20, 200, 5):
    for max_depth in range(3, 8, 1):
        params['num_leaves'] = num_leaves
        params['max_depth'] = max_depth

        cv_results = lgb.cv(
            params,
            lgb_train,
            seed=2018,
            nfold=3,
            metrics=['binary_error'],
            early_stopping_rounds=10,
            verbose_eval=True
        )

        mean_merror = pd.Series(cv_results['binary_error-mean']).min()
        boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()

        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params['num_leaves'] = num_leaves
            best_params['max_depth'] = max_depth

params['num_leaves'] = best_params['num_leaves']
params['max_depth'] = best_params['max_depth']



def get_feature_importance(booster, importance_type='split'):

    #check type and assign private attribute to public
    if isinstance(booster, LGBMModel):
        booster = booster.booster_
    elif not isinstance(booster, Booster):
        raise TypeError('booster must be Booster or LGBMModel.')

    feature_importance = booster.feature_importance(importance_type=importance_type)
    #print('feature_importance:', feature_importance)
    feature_name = booster.feature_name()
    #print('feature_name:', feature_name)

    if not len(feature_importance):
        raise ValueError("Booster's feature_importance is empty.")

    #sort features by importance
    tuples = sorted(zip(feature_name, feature_importance), key=lambda x: x[1]) #zip verschränkt Listen (1a 2b 3c...)
    #tuples = [x for x in tuples if x[1] > 0]
    #save features where the importance is > 0 and dismiss the rest
    #[7:] to cut column_ from string or integer
    tuples = [(int(x[0][7:]), x[1]) for x in tuples if x[1] > 0] #comprehension
    
    #print(tuples)
    #return important features only
    return tuples

feature_loo = LeaveOneOut()

for feature_index, lable_index in feature_loo.split(range(x.shape[1])):

    start= time.time()
    selected_x = x[:,feature_index]
    selected_y = x[:,lable_index][:,0]
    
    #print('selected_x.shape:', selected_x.shape)
    #print('selected_y.shape:', selected_y.shape) 
    #print('Feature {} as target'.format(lable_index[0]))
    result_per_feature_dict = {'feature':lable_index[0]}
    result_per_feature_dict.update({'name':data.keys()[lable_index[0]]})

    for _ in range(10):

        sample_loo = LeaveOneOut()

        best_score_list = []
        feature_importance_dict = {}

        for loo_index, (train_index, test_index) in enumerate(sample_loo.split(selected_x)):
            #print('loo_index:', loo_index)
            
            x_train = selected_x[train_index]
            y_train = selected_y[train_index]
            x_test = selected_x[test_index]
            y_test = selected_y[test_index]

            x_scaler = StandardScaler() #welchen skalierer benutzen? BoxCox
            x_train = x_scaler.fit_transform(x_train)
            x_test = x_scaler.transform(x_test)

            y_scaler = StandardScaler()
            y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))[:, 0]
            y_test = y_scaler.transform(y_test.reshape(-1, 1))[:, 0]
            
            #print(x_train)
            #print(y_train)
            #print(x_test.shape)
            #print(y_test.shape)
            
            train_data = lgb.Dataset(x_train, label=y_train)
            test_data = lgb.Dataset(x_test, label=y_test)

            evals_result = {}

            #num_leaves = 2
            #min_data_in_leaf = 20
            #feature_fraction = 1
            #bagging_fraction = 0.5

            param = {#'num_leaves':num_leaves, 
                     #'boosting' : 'dart',
                     #'min_data_in_leaf':min_data_in_leaf, 
                     #'max_bin':max_bin,
                     #'learning_rate':0.1,
                     'num_trees':70,
                     'objective':'regression_l2',
                     'min_data_in_leaf':9,
                     #'num_threads ':8
                     #'bagging_fraction':bagging_fraction,
                     #'bagging_freq':bagging_freq,
                     #'feature_fraction':feature_fraction,
                     'verbose':-1,
                     #'lambda_l2':lambda_l2,
                     #'min_gain_to_split':min_gain_to_split,
                     #'metric' : 'rmse',
                    }

            bst = lgb.train(param, 
                        train_data, 
                        valid_sets=[test_data], 
                        early_stopping_rounds=5,
                        verbose_eval=False,
                        #feval=,
                        evals_result=evals_result,
                       )

            feature_importance = get_feature_importance(bst)
            best_score = min(evals_result['valid_0']['l2']) #welcher score ist hier am besten?
            best_score_list.append(best_score)

            for important_feature, importance in feature_importance:
                #sub_dict = result.setdefault(f_result_key, {})
                #sub_dict.setdefault(key, []).append(f_result_value)
                feature_importance_dict.setdefault(important_feature, 0)
                #feature_importance_dict[fi_key] = feature_importance_dict[fi_key] + ((fi_value ** 2) / best_score)
                #the best score is minimal -> importance/score to maximize ########## ELSE MULTIPLY #########!!!!!!
                feature_importance_dict[important_feature] = feature_importance_dict[important_feature] + (importance / best_score) #importance quadrieren?

        result_per_feature_dict.update({'score (mse)' : np.mean(best_score_list)})
        #print('Score (mse):', np.mean(best_score_list))

        sorted_feature_importance = sorted(feature_importance_dict.items(), key=operator.itemgetter(1))

        #print('Most important feature: {} with {:.2E}'.format(sorted_feature_importance[-1][0], sorted_feature_importance[-1][1]))
        #result_per_feature_dict.update({'most_important_feature' : (data.keys()[sorted_feature_importance[-1][0]],
        #                                                     sorted_feature_importance[-1][1])})

        #get importance value for label
        label_importance = [fs for fs in sorted_feature_importance if fs[0] == (113 - 2)] #Featureanzahl generisch programmieren

        
        if not label_importance:
            #drop id from most important feature
            drop_id = sorted_feature_importance[-1][0]
            print('dropped label id', drop_id)
            #print('Label was not important - droping label id:', drop_id)
            #result_per_feature_dict.update({'label_was_important': -1})
            selected_x[:,drop_id] = 0
        else:
            #print('Label was important: {} with {:.2E}'.format(important_label[0][0],
                                                                #important_label[0][1]))
            result_per_feature_dict.update({'label_was_important': label_importance[0][1]})
            break

    end = time.time()
    print(end-start)
    with open('./data/result.csv', 'a', newline='')as csv_file:
        writer = csv.DictWriter(csv_file, result_per_feature_dict.keys())
        writer.writerow(result_per_feature_dict)
        print('#################################################')
        for (key, value) in result_per_feature_dict.items():
            print(key, " : ", value)
            #writer.writerow([key,value])

with open('./data/result.csv', 'a', newline='')as csv_file:
    writer = csv.DictWriter(csv_file, result_per_feature_dict.keys())
    writer.writeheader()
    csv_file.close

# Das sind nur Notizen um 'min_data_in_leaf' zu optimieren.
# - 'min_data_in_leaf':7 -> Score: 0.5023174063040724
# - 'min_data_in_leaf':8 -> Score: 0.4949493530714073
# - **'min_data_in_leaf':9 -> Score: 0.4189500672637431**
# - 'min_data_in_leaf':10 -> Score: 0.4882291102231029
# - 'min_data_in_leaf':11 -> Score: 0.5132265904483768
# - 'min_data_in_leaf':12 -> Crash
# - 'min_data_in_leaf':15 -> Crash
# 
# 
# 
# 
