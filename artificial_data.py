import numpy as np
import pandas as pd
from matplotlib import pyplot
import seaborn as sns

path = "./data/small.csv"

number_of_class_features = 15
number_of_random_features = 10

# each class is assigned the same number of samples TODO: different sample numbers
# total number of samples = number of classes * number of samples per class
number_of_samples_per_class = 4

#scale = standard deviation of normal distribution
scale = 2
#sigma = standard deviation of the underlying normal distribution. Should be greater than zero. Default is 1.
sigma = 1
mean_of_normal_distribution = 0
mean_of_lognormal_distribution = 0
shift_of_lognormal_distribution = 10
number_of_normal_distributed_classes = 1
number_of_lognormal_distributed_classes = 1


# generate artificial data to simulate the samples from healthy patients
def generate_normal_distributed_class(label:int, number_of_samples:int, mean):
    # generate labels
    class_data = np.full((number_of_samples, 1), label)
    features = np.random.normal(loc=mean, scale=scale, size=(number_of_samples, number_of_class_features))
    class_data = np.hstack((class_data, features))
    return class_data

# generate artificial data to simulate the samples from ill patients
def generate_lognormal_distributed_class(label:int, number_of_samples:int):
    # generate labels
    class_data = np.full((number_of_samples, 1), label)

    # sigma = Standard deviation of the underlying normal distribution. Should be greater than zero. Default is 1.
    # mean = Mean value of the underlying normal distribution.
    features = np.random.lognormal(mean=mean_of_lognormal_distribution, sigma=sigma,
                                   size=(number_of_samples, number_of_class_features))
    features = features + shift_of_lognormal_distribution
    class_data = np.hstack((class_data, features))
    return(class_data)

def generate_artificial_data():
    # TODO: update to different sample numbers per class
    number_of_classes = number_of_normal_distributed_classes + number_of_lognormal_distributed_classes
    number_of_all_samples = number_of_samples_per_class * number_of_classes

    # generate "healthy" data class 0
    labeled_features_class_0 = generate_normal_distributed_class(0, number_of_samples_per_class, 0)
    all_features = generate_normal_distributed_class(0, number_of_samples_per_class, 0)

    # TODO: extend data generation to multiple diagnoses
    # generate data with diagnosis of a specific disease
    # labeled_features_class_1 = generate_lognormal_distributed_class(1, number_of_samples_per_class)
    labeled_features_class_1 = generate_normal_distributed_class(1, number_of_samples_per_class, 6)
    all_features = np.vstack((all_features, labeled_features_class_1))

    # visualize the distributions
    sns.set(color_codes=True)
    input = labeled_features_class_1.flatten()
    sns.distplot(input);
    input2 = labeled_features_class_0.flatten()
    sns.distplot(input2);
    pyplot.show()

    # generate random data
    # mean for the random data is 0 TODO: ist mean = 0 realistisch? Macht der mean hier einen Unterschied?
    random_features = np.random.normal(loc=0.0, scale=2, size=(number_of_all_samples, number_of_random_features))
    all_features = np.hstack((all_features, random_features))
    pd.DataFrame(all_features).to_csv(path, index=False)

    print("Data generated successfully. You can find the generated file relative to artificial_data.py in: ", path)

generate_artificial_data()