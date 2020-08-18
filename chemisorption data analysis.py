'''Chemisorption model for co2 by Prof. Xin data analysis
    - Aadit Patel
    Started: 6/1/2020
    Last Edited: 6/3/2020
    '''

#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

#Normalizing data using Standard Scaler
from sklearn.preprocessing import StandardScaler
stdscaler = StandardScaler()
train_set_scaled = stdscaler.fit_transform(train_set_numdata_inputs)

#linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(train_set_scaled,train_set_trueE)

#RMSE
from sklearn.metrics import mean_squared_error
deltaE_pred = lin_reg.predict(train_set_scaled)

def rmse(trueE,predE):
    mse = mean_squared_error(trueE,predE)
    rmse = np.sqrt(mse)
    print('RMSE: ',rmse)
    return rmse

print('\nLin reg:')
rmse(train_set_trueE,deltaE_pred)

#cross_val_score evaluation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_reg,train_set_scaled,train_set_trueE,scoring = 'neg_mean_squared_error',cv=10)
lin_reg_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print('Cross Validation scores:')
    print('Scores:',scores)
    print('Mean:',scores.mean())
    print('Std',scores.std())

print('\nLin reg cross val scores:')
display_scores(lin_reg_rmse_scores)


#decision tree regression
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_set_scaled,train_set_trueE)

deltaE_pred_decision_tree = tree_reg.predict(train_set_scaled)

print('\nDecision tree:')
rmse(train_set_trueE,deltaE_pred_decision_tree)
tree_scores = cross_val_score(tree_reg,train_set_scaled,train_set_trueE,scoring='neg_mean_squared_error',cv = 10)
dec_tree_cross_val_rmse = np.sqrt(-tree_scores)
display_scores(dec_tree_cross_val_rmse)


#random forest regressor
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(random_state=2)
forest_reg.fit(train_set_scaled,train_set_trueE)

forest_reg_predection = forest_reg.predict(train_set_scaled)
print('\nForest Regressor:')
rmse(train_set_trueE,forest_reg_predection)
forest_cross_val = cross_val_score(forest_reg,train_set_scaled,train_set_trueE,scoring='neg_mean_squared_error',cv = 10)
forest_cross_val_rmse = np.sqrt(-forest_cross_val)
display_scores(forest_cross_val_rmse)


### Model evaluation on test set
print('\n\nMODEL EVALUATION ON TEST SET')
#preparing test set data
test_set_numdata = test_set.drop('Alloy Symbol',axis = 1)
test_set_input_vals = test_set_numdata.drop('deltaE *CO',axis = 1)
test_set_trueE = test_set['deltaE *CO'].copy()

#scaling test set data (only transforming no fit)
test_set_input_scaled = stdscaler.transform(test_set_input_vals)

#linear regression
lin_reg_test_predictions = lin_reg.predict(test_set_input_scaled)
print('\nlinear regression test set predictions')
rmse(test_set_trueE,lin_reg_test_predictions)

#Random Forest Regressor
forest_reg_test_predictions = forest_reg.predict(test_set_input_scaled)
print('\nRandom Forest Regressor test set predictions')
forest_rmse = rmse(test_set_trueE,forest_reg_test_predictions)


#plotting real values vs predicted
plt.scatter(forest_reg_predection,train_set_trueE,label = 'Training set')
plt.scatter(forest_reg_test_predictions,test_set_trueE, label = 'Test set')
plt.xlabel('Predicted deltaE')
plt.ylabel('Actual deltaE')
plt.title('Random Forest Regressor model for Chemisportion of CO2\n RMSE of test set: {}'.format(forest_rmse))
plt.plot([-2,0],[-2,0])
plt.legend()
plt.ylim(-2,0)
plt.xlim(-2,0)
plt.show()

print('\nRandom forest regressor consistently more accurate than the feedforward neural network used in paper (RMSE: .12)')