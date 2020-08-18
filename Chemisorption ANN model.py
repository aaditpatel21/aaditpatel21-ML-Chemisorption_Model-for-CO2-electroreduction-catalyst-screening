'''Chemisorption model for co2 by Prof. Xin data analysis
    - Aadit Patel
    Started: 8/8/2020
    Last Edited: 8/12/2020
    '''

#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

#read chemisorption csv data
chemisorption_data = pd.read_csv("chemisorption_model_co2_data_mlfriendly.csv")
print(chemisorption_data.info(),chemisorption_data.columns)
print(chemisorption_data.describe())

#histograms to visualize all the data
'''chemisorption_data.hist(bins = 50 , figsize=(20,15))
plt.show()'''


#train test split
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(chemisorption_data,test_size=.2,random_state=20)

#correlation analysis to delta E
corr_matrix = chemisorption_data.corr()
print('Input variables correlation analysis:')
print(corr_matrix['deltaE *CO'].sort_values(ascending=False))
print('Further agrees with paper that d-band values are most influential to Delta E *CO')

#drop alloy symbols from train set
train_set_numdata = train_set.drop('Alloy Symbol',axis = 1)

#split input data and outputs
train_set_numdata_inputs = train_set_numdata.drop('deltaE *CO',axis = 1)
train_set_trueE = train_set_numdata['deltaE *CO'].copy()

#split into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(train_set_numdata_inputs,train_set_trueE,test_size=.2)
print(len(X_train),len(X_valid),len(y_train),len(y_valid))

#Normalizing data using Standard Scaler
from sklearn.preprocessing import StandardScaler
stdscaler = StandardScaler()
X_train_scaled = stdscaler.fit_transform(X_train)
X_valid_scaled = stdscaler.transform(X_valid)


#preparing test set data
test_set_numdata = test_set.drop('Alloy Symbol',axis = 1)
test_set_input_vals = test_set_numdata.drop('deltaE *CO',axis = 1)
test_set_trueE = test_set['deltaE *CO'].copy()

#scaling test set data (only transforming no fit)
test_set_input_scaled = stdscaler.transform(test_set_input_vals)

model = keras.models.Sequential([
    keras.layers.Dense(5,activation = 'sigmoid',input_shape=X_train_scaled.shape[1:]),
    keras.layers.Dense(2,activation= 'sigmoid'),
    keras.layers.Dense(1)
])

model.compile(loss = 'mean_squared_error',optimizer = 'sgd',metrics = [tf.keras.metrics.RootMeanSquaredError()])
history = model.fit(X_train_scaled,y_train,epochs = 5000,validation_data = (X_valid_scaled,y_valid))
rmse_test = model.evaluate(test_set_input_scaled,test_set_trueE)
print(rmse_test)

training_pred = model.predict(X_train_scaled)
test_pred = model.predict(test_set_input_scaled)

plt.scatter(training_pred,y_train,label = 'Training set')
plt.scatter(test_pred,test_set_trueE, label = 'Test set')
plt.xlabel('Predicted deltaE')
plt.ylabel('Actual deltaE')
plt.title('Feed Forward Artificial Neural Network model for Chemisportion of CO2\n RMSE of test set: {}'.format(rmse_test[1]))
plt.plot([-2,0],[-2,0])
plt.legend()
plt.ylim(-2,0)
plt.xlim(-2,0)
plt.show()
