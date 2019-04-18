import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('U:/pythonTest/bacmen_vs_viral.csv', header=0, sep=',')

BacMen = data[data['diagnosis'].isin(['BacMen'])]
HSV = data[data['diagnosis'].isin(['HSV'])]
Z_men_enc = data[data['diagnosis'].isin(['Z. men_enc'])]
Ent_men = data[data['diagnosis'].isin(['Ent. men'])]

y = np.array(data['diagnosis'].tolist())

data = data.drop('diagnosis', 1)
x = np.array(data.values)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    test_size = 0.2,
                                                    random_state=42,
                                                    stratify = y,
                                                    shuffle = True)

print(np.std(x_train, axis=0))



