# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:19:49 2024
This script is to using Random Forest to predict O3 concentration 
from PMF-resolved VOC sources in Bronx, NYC. 
SHAP was implemented to deconvolute the results of 3 ML algorithms.
@author: lb945465
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error

# Load the dataset
data = pd.read_excel(r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Spyder\SHAP\Merged Ozone and PMF.xlsx')

data.dropna(inplace = True)

#Add a few more features to help the model. I chose day, week, month, and year to account for seasonality in the trends. 
data['Unix'] = (data['Date'] - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
#data['Day'] = pd.DatetimeIndex(data['Date']).dayofyear
data['Hour'] = pd.DatetimeIndex(data['Date']).hour

# Set 'Date' as the index (if needed)
data.set_index('Date', inplace=True)

# Separate features and target
X = data.drop('Ozone', axis=1)
y = data['Ozone']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Initialize and train the model
#model = RandomForestRegressor(random_state=42) # Test Score  66.329 R^2 Score 0.6633
#model = ExtraTreesRegressor(random_state=42)
model = XGBRegressor(random_state=42)

model.fit(X_train, y_train)

# Initialize SHAP Explainer and calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

shap_explanation = explainer(X_train)

# Set the default DPI for all plots
plt.rcParams['figure.dpi'] = 200

# List of features to display in the plot
features_to_display = ['Fuel evaporation', 'Combustion', 'Natural gas', 
                       "Diesel traffic", "Industrial solvents", "Gasoline traffic", 
                       "Biogenic"] 

# Filter the SHAP values to include only the selected features
filtered_shap_values = shap_explanation[:, features_to_display]

# Plots 
shap.plots.bar(filtered_shap_values)
shap.plots.beeswarm(filtered_shap_values, show=True)
shap.plots.waterfall(filtered_shap_values[0], show=True)

predict = model.predict(X_test)

#Statistical metrics and performance evaluation
#print("Out-of-bag score", round(model.oob_score_, 4)) #Use this if oob_score=True
print('Importances', model.feature_importances_)
print("Mean Absolute Error", round(metrics.mean_absolute_error(y_test, predict), 4))
print("Mean Squared Error", round(metrics.mean_squared_error(y_test, predict), 4))
print("Root Mean Squared Error", round(np.sqrt(metrics.mean_squared_error(y_test, predict)), 4))
print("(R^2) Score", round(metrics.r2_score(y_test, predict), 4))
print(f'Train Score  {model.score(X_train, y_train) * 100:.3f}') # Take note that this is percentage
print(f'Test Score  {model.score(X_test, y_test) * 100:.3f}') # Take note that this is percentage
errors = abs(predict - y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy', round(accuracy, 3)) # Take note that this is percentage

##### Additional plots

plt.figure(figsize=(5, 5))
a=plt.scatter(y=predict, x=y_test, 
            c ="blue",
            linewidths = 0.5,
            marker ="o",
            edgecolor ="black",
            alpha=0.3,
            s = 50)
a=plt.plot([0, max(predict)+max(predict*0.2)], [0, max(predict)+max(predict*0.2)], color = 'red', linestyle = 'solid')

a=plt.xlabel('Observed Ozone concentration (ppbv)')
a=plt.ylabel('Predicted Ozone concentration (ppbv)')

a=plt.ylim(0,max(predict)+max(predict*0.2))
a=plt.xlim(0,max(predict)+max(predict*0.2))
a=plt.annotate(
    "Test set:" + 
    "\n" + 
    "r-squared = {:.2f}".format(r2_score(y_test, predict)) + 
    "\n" + 
    "RMSE = {:.2f}".format(np.sqrt(mean_squared_error(y_test, predict))), 
    xy=(0.05, 0.95), xycoords='axes fraction', 
    horizontalalignment='left', verticalalignment='top'
)

plt.figure(figsize=(5, 5))
predict_train = model.predict(X_train)
b=plt.scatter(y=predict_train, x=y_train, 
            c ="blue",
            linewidths = 0.5,
            marker ="o",
            edgecolor ="black",
            alpha=0.3,
            s = 50)
b=plt.plot([0, max(predict_train)+max(predict_train*0.2)], [0, max(predict_train)+max(predict_train*0.2)], color = 'red', linestyle = 'solid')

b=plt.xlabel('Observed Ozone concentration (ppbv)')
b=plt.ylabel('Predicted Ozone concentration (ppbv)')

b=plt.ylim(0,max(predict_train)+max(predict_train*0.2))
b=plt.xlim(0,max(predict_train)+max(predict_train*0.2))
b=plt.annotate("Train set:" + 
                "\n" + 
                "r-squared = {:.2f}".format(r2_score(y_train, predict_train)) + 
                "\n" + 
                "RMSE = {:.2f}".format(np.sqrt(mean_squared_error(y_train, y_train))), 
                xy=(0.05, 0.95), xycoords='axes fraction', 
                horizontalalignment='left', verticalalignment='top'
)