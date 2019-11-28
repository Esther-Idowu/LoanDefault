#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
from functools import reduce

#IMPORTING THE DATASETS
dataset_1 = pd.read_csv('traindemographics.csv')
dataset_2 = pd.read_csv('trainperf.csv')
dataset_3 = pd.read_csv('trainprevloans.csv')
data_frames = [dataset_1, dataset_3, dataset_2]
dataset = reduce(lambda  left,right: pd.merge(left,right,on=['customerid'],
                                            how='outer'), data_frames).fillna('nan')
X = dataset.iloc[:, [2,3,4,5,6,7,8,10,13,14,15,17,20,21,24,25,26,27]].values
y = dataset.iloc[:, 28].values

#ENCODING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
data = []
for n in range(0, 18):
    if type(X[0, n]) == str:
        X[:,n]=labelencoder_X.fit_transform(X[:,n])
        data.append(n)
        
for i in range(0, 19281):
    for j in range(0, 18):
        if X[i, j] == 'nan':
            X[i, j] = 0
onehotencoder=OneHotEncoder(categorical_features=[0, 3, 4, 5, 6, 11, 17])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
for i in range(0, 19281):
    if y[i, ] == 2:
        y[i, ] = 0
        
#SPLITTING THE DATA INTO TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.2, random_state = 0)

#FEATURE SCALING
#from sklearn.preprocessing import StandardScaler
#sc_X=StandardScaler()
#X_train=sc_X.fit_transform(X_train)
#X_test=sc_X.transform(X_test)

##FITTING MULTIPLE LINEAR REGRESSION TO TRAINING SET
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

#PREDICTING THE TEST RESULT
y_pred=regressor.predict(X_test)
for i in range(0, 3857):
    if y_pred[i, ] > 0.5:
        y_pred[i, ] = 1
    else:
        y_pred[i, ] = 0

def accuracy(y_pred, y_test):
    absolute_error = np.zeros((len(y_pred), 1), dtype = float)
    relative_error = np.zeros((len(y_pred), 1), dtype = float)
    accuracies = np.zeros((len(y_pred), 1), dtype = float)
    for i in range(0, len(y_pred)):
        absolute_error[i, 0] = np.abs(y_pred[i, ] - y_test[i, ])
        accuracies[i, 0] = (1 - absolute_error[i, 0]) * 100
    mean_accuracy = accuracies.mean()
    print('THE MEAN ACCURACY IS ' + str(mean_accuracy))
    return absolute_error, relative_error, accuracies
absolute_error, relative_error, accuracies= accuracy(y_pred, y_test)




#SUBMISSION
dataset_1 = pd.read_csv('testdemographics.csv')
dataset_2 = pd.read_csv('testperf.csv')
dataset_3 = pd.read_csv('testprevloans.csv')
data_frames = [dataset_1, dataset_3, dataset_2]
dataset = reduce(lambda  left,right: pd.merge(left,right,on=['customerid'],
                                            how='outer'), data_frames).fillna('nan')
Xsubm = dataset.iloc[:, [2,3,4,5,6,7,8,10,13,14,15,17,20,21,24,25,26,27]].values

#ENCODING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
data = []
for n in range(0, 18):
    if type(Xsubm[0, n]) == str:
        Xsubm[:,n]=labelencoder_X.fit_transform(Xsubm[:,n])
        data.append(n)
        
for i in range(0, 7017):
    for j in range(0, 18):
        if Xsubm[i, j] == 'nan':
            Xsubm[i, j] = 0
onehotencoder=OneHotEncoder(categorical_features=[0, 3, 4, 5, 6, 11, 17])
Xsubm=onehotencoder.fit_transform(Xsubm).toarray()

#PREDICTING THE TEST RESULT
y_pred=regressor.predict(Xsubm)
for i in range(0, 7017):
    if y_pred[i, ] > 0.5:
        y_pred[i, ] = 1
    else:
        y_pred[i, ] = 0
        
