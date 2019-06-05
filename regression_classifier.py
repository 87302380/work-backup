# coding: utf-8
import time
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import data_preparation_arificial_data as to_be_deleted
#import data_preparation_bacmen_vs_viral as to_be_deleted
import warnings
import parameters
# import feature_weights

warnings.filterwarnings("ignore")

def get_weights():
    x = to_be_deleted.get_data_with_label_included()
    weights = np.ones(x.shape[1])
    for i in range(15):
       weights[i] = 1

    return weights

def _get_classification_result(true_data, predicted_samples):
    
    weights = get_weights()
    list_of_classes = np.unique(true_data[:, 0])
    number_of_samples = true_data.shape[0]

    # initialize the classification result array with False
    classification_result = np.full(number_of_samples, False)
    classified_classes = []

    # the classes equal the labels as integer representation
    true_classes = true_data[:, 0]
    true_classes.astype(int)

    for selected_sample in range(number_of_samples):
        real_sample = true_data[selected_sample]

        distance_to_each_class = []
        
        for class_number in list_of_classes:
            all_sample_indices_from_class = (true_data[:, 0] == class_number)

            all_predicted_samples_from_one_class = predicted_samples[all_sample_indices_from_class]

            weighted_distances = []
            for predicted_sample in all_predicted_samples_from_one_class:
                weighted_distances.append(np.sum(np.absolute(predicted_sample - real_sample) * weights))

            #TODO mean oder median?
            distance_to_each_class.append(np.median(weighted_distances))
            #distance_to_each_class.append(np.mean(weighted_distances))

        print('distance to classes:', distance_to_each_class)
        minimum_distance = min(distance_to_each_class)

        # the index of distance_to_each_class equals the integer representation of the class
        classified_class = distance_to_each_class.index(minimum_distance)
        print('classified class:', classified_class)
        print('true class:', true_data[selected_sample, 0])

        classified_classes.append(classified_class)

        classification_result[selected_sample] = (classified_class == true_data[selected_sample, 0])

    print('##')
    print(classified_classes == true_classes)
    print('##')
    #TODO return only classified classes to print or plot results?
    return classification_result


def get_accuracy(data, parameters, early_stopping_rounds):
    total_time_start = time.time()

    feature_loo = LeaveOneOut()

    # initialize 2D array for all prediction results
    predicted_samples = np.empty_like(data)

    # for each feature and the label train and validate a model and predict a sample with
    # independent test data not used for training or validation before
    for excluded_feature_index, target_feature_index in feature_loo.split(range(data.shape[1])):
        print('Feature {} from {}'.format(target_feature_index, data.shape[1]))
        #TODO Zeitprogressbar

        # measure the time needed to calculate one feature
        start = time.time()

        data_without_target_feature = data[:, excluded_feature_index]
        target_feature = data[:, target_feature_index][:, 0]

        # leave one out cross validation to select the test sample
        sample_loo_outer = LeaveOneOut()

        # TODO enumerate still needed? If not: delete!
        for loo_index_outer, (remaining_data_index, test_sample_index) in enumerate(sample_loo_outer.split(data_without_target_feature)):

            data_without_test_sample = data_without_target_feature[remaining_data_index]
            test_sample = data_without_target_feature[test_sample_index]

            predictions_from_cv_for_a_single_value = []

            # leave one out cross validation to train and validate a model based on the data excluding a
            # test sample (test_sample) and predict a sample (predictions_from_cv_for_a_single_value) based on this excluded test sample
            # selected in the outer cross validation loop
            sample_loo_inner = LeaveOneOut()
            for loo_index_inner, (train_index, validation_index) in enumerate(sample_loo_inner.split(data_without_test_sample)):
                x_train = data_without_target_feature[train_index]
                y_train = target_feature[train_index]
                x_validation = data_without_target_feature[validation_index]
                y_validation = target_feature[validation_index]

                # TODO Skalierung?
                x_scaler = StandardScaler() #welchen Skalierer benutzen? BoxCox
                x_train = x_scaler.fit_transform(x_train)
                x_validation = x_scaler.transform(x_validation)

                y_scaler = StandardScaler()
                y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))[:, 0]
                y_validation = y_scaler.transform(y_validation.reshape(-1, 1))[:, 0]

                train_data = lgb.Dataset(x_train, label=y_train)
                validation_data = lgb.Dataset(x_validation, label=y_validation)

                booster = lgb.train(parameters,
                                    train_data,
                                    valid_sets=[validation_data],
                                    early_stopping_rounds=early_stopping_rounds,
                                    verbose_eval=False,
                                    # feval=,
                                    # evals_result=evals_result,
                                    )

                prediction = booster.predict(test_sample)
                predictions_from_cv_for_a_single_value.append(prediction[0])

            #print('standard deviation for all predictions from cv for a single value:', np.std(predictions_from_cv_for_a_single_value))

            # save the mean of all predictions from the cross validation for the indexed value
            #TODO mean oder median?
            predicted_samples[test_sample_index, target_feature_index] = np.median(predictions_from_cv_for_a_single_value)

        #y_scaler.inverse_transform(y_scaler, predicted_samples[target_feature_index])
        print('calculation time per feature:', (time.time() - start))
    # reverse transform

    classification_result = _get_classification_result(data, predicted_samples)
    print(classification_result)
    print('classification error:', len(classification_result) - sum(classification_result))

    total_calculation_time_end = time.time()
    overall_time = total_calculation_time_end - total_time_start
    day = overall_time // (24 * 3600)
    overall_time = overall_time % (24 * 3600)
    hour = overall_time // 3600
    overall_time %= 3600
    minutes = overall_time // 60
    overall_time %= 60
    seconds = overall_time
    duration = (day, hour, minutes, seconds)
    print("total calculation time: %d:%d:%d:%d" % (day, hour, minutes, seconds))

    return (sum(classification_result)/len(classification_result)*100)

data = to_be_deleted.get_data_with_label_included()
param = parameters.param
print('accuracy:', get_accuracy(data, param, 15))