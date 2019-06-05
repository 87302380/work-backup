import pandas as pd
import numpy as np

#TODO Pfad anpassen
data = pd.read_csv('./data/small.csv', sep=',')

def get_data_with_label_included():
    # features with label included

    x = np.array(data.values)
    return x

def get_x_with_label_excluded():

    temp_data = data.copy()

    # create features without label
    temp_data.drop('0', axis=1, inplace=True)
    x = np.array(temp_data.values)
    return x

#returns labels as numpy array
def get_labels():

    # delete me if not needed for testing
    y = np.array(data['0'].tolist())

    return np.array(data['0'].tolist())

def get_number_of_features():
    #all features excluding the label
    return len(data.values[0])-1

def get_label_id():
    return 0
