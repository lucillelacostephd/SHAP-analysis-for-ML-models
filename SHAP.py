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
from scipy import stats
import seaborn as sns
import scipy
import xgboost

# Load the dataset
data = pd.read_excel(r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Spyder\SHAP\Merged Ozone and NO and PMF.xlsx')

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True) # test size 0.25

# Initialize and train the model
#model = RandomForestRegressor(random_state=42) # Test Score  66.329 R^2 Score 0.6633
#model = ExtraTreesRegressor(random_state=42)
model = XGBRegressor(random_state=42)

model.fit(X_train, y_train)

# Initialize SHAP Explainer and calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap_explanation = explainer(X_test)
predict = model.predict(X_test)

# Compute SHAP values for X_train and X_test
shap_values_train = explainer.shap_values(X_train)
shap_values_test = explainer.shap_values(X_test)

# Convert SHAP values to DataFrames
# Assuming shap_values_train and shap_values_test are arrays
shap_values_train_df = pd.DataFrame(shap_values_train, columns=X_train.columns, index=X_train.index)
shap_values_test_df = pd.DataFrame(shap_values_test, columns=X_test.columns, index=X_test.index)
combined_shap_values_df = pd.concat([shap_values_train_df, shap_values_test_df])

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

##### SHAP plots
# Set the default DPI for all plots
plt.rcParams['figure.dpi'] = 200  # Replace 100 with your desired DPI

# List of features to display in the plot
features_to_display = ['Fuel evaporation', 'Combustion', 'Natural gas', 
                       "Diesel traffic", "Industrial solvents", "Gasoline traffic", 
                       "Biogenic",
                       #'Nitric oxide (NO)',
                       ] 

# Define the colors for specific column names
color_mapping = {
    "Fuel evaporation": "#1f77b4",   # blue
    "Combustion": "#ff7f0e",   # orange
    "Natural gas": "#2ca02c",   # green
    "Diesel traffic": "#d62728",   # red
    "Industrial solvents": "#9467bd",   # purple
    "Gasoline traffic": "#8c564b",   # brown
    "Biogenic": "#e377c2", # pink
    }

# Filter the SHAP values to include only the selected features
filtered_shap_values = shap_explanation[:, features_to_display]

# Plots 
## The mean absolute SHAP value is a point estimate that represents the average impact of a feature on the model's predictions.
shap.plots.bar(filtered_shap_values[0])
plt.savefig('Shap_barplot.png', bbox_inches='tight')
plt.show() 

shap.plots.beeswarm(shap_explanation[:, features_to_display])
plt.savefig('Shap_beeswarm.png', bbox_inches='tight')
plt.show() 

# Additional plots
shap.plots.waterfall(filtered_shap_values[0])

shap.plots.scatter(
    filtered_shap_values, ylabel="SHAP value",
)

# Old skool plots
# xgboost.plot_importance(model)
# plt.title("xgboost.plot_importance(model)")
# plt.show()

##### Training vs test sets plots

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

# Save the figure
plt.savefig('Test_plot.png', transparent=True, bbox_inches='tight')
plt.show()

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

# Save the figure
plt.savefig('Train_plot.png', transparent=True, bbox_inches='tight')
plt.show()

##### Ozone observed vs measured plots
# Generate predictions for the test set
y_pred = model.predict(X_test)

# Create a DataFrame for comparison
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)

# Resample to yearly frequency and compute means and SEM
yearly_stats = comparison_df.resample('Y').agg(['mean', 'sem'])
yearly_stats.columns = ['_'.join(col) for col in yearly_stats.columns]

# Calculate the t-critical value for 95% confidence interval
confidence_level = 0.95
degrees_freedom = yearly_stats.shape[0] - 1
t_critical = stats.t.ppf((1 + confidence_level) / 2, df=degrees_freedom)

# Calculate the margin of error (error bars length)
yearly_stats['Actual_mean_error'] = yearly_stats['Actual_sem'] * t_critical
yearly_stats['Predicted_mean_error'] = yearly_stats['Predicted_sem'] * t_critical

# Plotting the bar graph
fig, ax = plt.subplots(figsize=(15, 4))

# Width of a bar
width = 0.35

# Positions of the left bar-boundaries
bar_l = np.arange(1, len(yearly_stats) + 1)

# Positions of the x-axis ticks (center of the bars as bar labels)
tick_pos = [i + width / 2 for i in bar_l]

# Create the bar plot
ax.bar(bar_l, yearly_stats['Actual_mean'], width=width, label='Observed', yerr=yearly_stats['Actual_mean_error'], capsize=5)
ax.bar(bar_l + width, yearly_stats['Predicted_mean'], width=width, label='Predicted', yerr=yearly_stats['Predicted_mean_error'], capsize=5)

# Set the x ticks with names
plt.xticks(tick_pos, yearly_stats.index.year)

# Set the labels and title
plt.ylabel('Ozone concentration (ppbv)', fontweight="bold")
plt.xlabel('')
#plt.title('Yearly Comparison of Actual vs. Predicted with Confidence Intervals')
plt.legend()

# Set a buffer around the edge with increased space
buffer_space = width * 2  # You can adjust the multiplier to increase or decrease the space
plt.xlim([min(tick_pos) - buffer_space, max(tick_pos) + buffer_space])

# Set ylim
plt.ylim(0, 50)

# Save the figure
plt.savefig('Ozone_plot.png', transparent=True, bbox_inches='tight')
plt.show()