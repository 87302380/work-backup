import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('./data/bacmen_vs_viral.csv', header=0, sep=',')

data.loc[data.diagnosis == 'BacMen', 'diagnosis'] = 1.0
data.loc[data.diagnosis == 'HSV', 'diagnosis'] = 2.0
data.loc[data.diagnosis == 'Z. men_enc', 'diagnosis'] = 3.0
data.loc[data.diagnosis == 'Ent. men', 'diagnosis'] = 4.0

data['diagnosis'] = data['diagnosis'].astype('float64')

print(data)

y = np.array(data['diagnosis'].tolist())

data = data.drop('diagnosis', 1)
x = np.array(data.values)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size = 0.2,
                                                    random_state=42,
                                                    stratify = y,
                                                    shuffle = True)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train_df = pd.DataFrame(x_train)
y_train_df = pd.DataFrame(y_train)
x_test_df = pd.DataFrame(x_test)
y_test_df = pd.DataFrame(y_test)


x_train_df.to_csv('./data/x_train.csv', index=False)
y_train_df.to_csv('./data/y_train.csv', index=False)
x_test_df.to_csv('./data/x_test.csv', index=False)
y_test_df.to_csv('./data/y_test.csv', index=False)






