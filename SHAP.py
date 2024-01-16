# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:19:49 2024
This script is to using Random Forest to predict O3 concentration 
from PMF-resolved VOC sources in Bronx, NYC. 
SHAP was implemented to deconvolute the results of Random Forest.
@author: lb945465
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from xgboost import XGBRegressor

# Load the dataset
data = pd.read_excel(r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Spyder\SHAP\Merged Ozone and PMF.xlsx')

# Set 'Date' as the index (if needed)
data.set_index('Date', inplace=True)

# Separate features and target
X = data.drop('O3', axis=1)
y = data['O3']

outlier_indices = []

# Iterate over each column
for column in X.columns:
    # Skip non-numeric columns if necessary
    if X[column].dtype == 'object':
        continue

    # Calculate Q1 (25th percentile) and Q3 (75th percentile) of the column
    Q1 = np.percentile(X[column], 25)
    Q3 = np.percentile(X[column], 75)

    # Calculate the IQR (Interquartile Range)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Determine a list of indices of outliers for feature col
    outlier_list_col = X[(X[column] < lower_bound) | (X[column] > upper_bound)].index

    # Append the found outlier indices for col to the list of outlier indices 
    outlier_indices.extend(outlier_list_col)

# Select observations containing more than one outlier
from collections import Counter
multiple_outliers = list(k for k, v in Counter(outlier_indices).items() if v > 1)

print("Number of observations with more than one outlier:", len(multiple_outliers))

# Drop outliers
X_cleaned = X.drop(multiple_outliers)
y_cleaned = y.drop(multiple_outliers)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.3, random_state=42)

# # Define the search space for XGBoost hyperparameters
# search_space = {
#     'n_estimators': Integer(100, 500),
#     'max_depth': Integer(3, 10),
#     'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
#     'subsample': Real(0.5, 1.0, prior='uniform'),
#     'colsample_bytree': Real(0.5, 1.0, prior='uniform'),
#     'gamma': Real(0, 5, prior='uniform'),
#     # Add other XGBoost-specific parameters you want to tune
# }

# # Initialize the Bayesian optimizer with the search space and the estimator
# bayes_search = BayesSearchCV(
#     estimator=XGBRegressor(random_state=42),
#     search_spaces=search_space,
#     n_iter=32,  # Number of iterations
#     scoring='neg_mean_squared_error',  # Specify your scoring metric
#     n_jobs=-1,
#     cv=5,  # Cross-validation splitting strategy
#     random_state=42
# )

# # Fit the BayesSearchCV object to find the best hyperparameters
# bayes_search.fit(X_train, y_train)

# # Retrieve the best model
# model = bayes_search.best_estimator_

# Initialize and train the model
#model = RandomForestRegressor(random_state=42)
#model = ExtraTreesRegressor(random_state=42)
model = XGBRegressor(random_state=42)

model.fit(X_train, y_train)

# Initialize SHAP Explainer and calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

shap_explanation = explainer(X_train)

# Set the default DPI for all plots
plt.rcParams['figure.dpi'] = 200

# Plots 
shap.summary_plot(shap_values, X_train, plot_type="bar")
shap.plots.bar(shap_explanation.abs.max(0))
shap.plots.beeswarm(shap_explanation, show=True)
shap.plots.waterfall(shap_explanation[0], show=True)
shap.plots.heatmap(shap_explanation)
shap.plots.scatter(shap_explanation[:, "Industrial solvents"])
force_plot = shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
shap.save_html('force_plot.html', force_plot)

# Calculate the expected value, which is the base value for the decision plot
expected_value = explainer.expected_value
# If you are dealing with a multi-output model, you will have an expected value for each output
if isinstance(expected_value, np.ndarray):
    expected_value = expected_value[0]  # Take the expected value for the first output
# Create a decision plot for all predictions in the test set
shap.decision_plot(expected_value, shap_values, X_test, show=True)
# If you wish to save this plot
plt.savefig('decision_plot_all.png', bbox_inches='tight')

# Define custom scorer functions for cross-validation
def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Custom scorers for cross-validation
r2_scorer = make_scorer(r2_score)
mse_scorer = make_scorer(mean_squared_error)
rmse_scorer = make_scorer(rmse_score)
mae_scorer = make_scorer(mean_absolute_error)

# Perform 10-fold cross-validation for different metrics
scores_r2 = cross_val_score(model, X, y, cv=10, scoring=r2_scorer)
scores_mse = cross_val_score(model, X, y, cv=10, scoring=mse_scorer)
scores_rmse = cross_val_score(model, X, y, cv=10, scoring=rmse_scorer)
scores_mae = cross_val_score(model, X, y, cv=10, scoring=mae_scorer)

# Calculate mean and standard deviation for each metric
mean_r2 = np.mean(scores_r2)
std_r2 = np.std(scores_r2)
mean_mse = np.mean(scores_mse)
std_mse = np.std(scores_mse)
mean_rmse = np.mean(scores_rmse)
std_rmse = np.std(scores_rmse)
mean_mae = np.mean(scores_mae)
std_mae = np.std(scores_mae)

# Print the results
print(f'R^2: {mean_r2:.4f} (± {std_r2:.4f})')
print(f'MSE: {mean_mse:.4f} (± {std_mse:.4f})')
print(f'RMSE: {mean_rmse:.4f} (± {std_rmse:.4f})')
print(f'MAE: {mean_mae:.4f} (± {std_mae:.4f})')

