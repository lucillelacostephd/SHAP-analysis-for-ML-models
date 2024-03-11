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
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from scipy import stats
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error

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

# Preparing the scaler
scaler = StandardScaler()

# Standardize the dataframe
X_scaled = scaler.fit_transform(X)

# Convert the array back to a pandas DataFrame
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, shuffle=True) # test size 0.25

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
features_to_display = ['Industrial evaporation', 'Biogenic', 'Fuel evaporation',
       'Polymer production', 'Industrial', 'Vehicle exhaust',
       'Diesel emissions', 'Gasoline emissions', 
       #'Nitric oxide (NO)',
       #'Unix', 'Hour'
       ]

# Define the colors for specific column names
color_mapping = {
    "Fuel evaporation": "#1f77b4",   # blue
    "Diesel emissions": "#ff7f0e",   # orange
    "Gasoline emissions": "#2ca02c",   # green
    "Vehicle exhaust": "#d62728",   # red
    "Polymer production": "#9467bd",   # purple
    "Industrial": "#8c564b",   # brown
    "Biogenic": "#e377c2", # pink
    "Industrial evaporation": "#7f7f7f", # gray
    #"Styrene-rich": "#17becf",  # cyan
    #"Diethylbenzene-rich": "#9edae5",  # light blue
    }

# Filter the SHAP values to include only the selected features
filtered_shap_values = shap_explanation[:, features_to_display]

# Plots 
shap.plots.bar(filtered_shap_values)
plt.show() 
plt.savefig('Shap_barplot_global.png', bbox_inches='tight', dpi=200)

# Assuming filtered_shap_values is a SHAP Explanation object
shap_values_array = filtered_shap_values.values  # Access the values in the Explanation object

# # Find the index of the first instance where all SHAP values are positive
# positive_shap_index = None
# for i in range(shap_values_array.shape[0]):
#     if np.all(shap_values_array[i] > 0):
#         positive_shap_index = i
#         break

# # Check if such an instance was found
# if positive_shap_index is not None:
#     # Generate the local plot for this instance
#     shap.plots.bar(filtered_shap_values[positive_shap_index])
#     plt.show()
# else:
#     print("No instance found where all SHAP values are positive.")
# plt.savefig('Shap_barplot_local.png', bbox_inches='tight')

shap.plots.beeswarm(shap_explanation[:, features_to_display])
plt.show() 
plt.savefig('Shap_beeswarm.png', bbox_inches='tight', dpi=200)

# Additional plots
shap.plots.waterfall(filtered_shap_values[0])

# shap.plots.scatter(
#     filtered_shap_values, ylabel="SHAP value",
# )

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
                "RMSE = {:.2f}".format(np.sqrt(mean_squared_error(y_train, predict_train))), 
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

##### Contour plots with Ozone
# Make sure 'X' is a DataFrame and has the expected columns
assert isinstance(X_scaled, pd.DataFrame), "X must be a pandas DataFrame."
assert all(feature in X_scaled.columns for feature in features_to_display), "All features must be columns in X."

# Compute the mean of the features
mean_feature_values = X_scaled.mean()

# Define the number of bins for the features and the predicted Ozone
num_feature_bins = 8
num_ozone_bins = 8

# Formatter for the colorbar ticks to have 1 decimal place
formatter = FormatStrFormatter('%.0f')

# Function to bin the features and predict Ozone
def predict_and_bin_features(feature_x_name, feature_y_name, model, scaler, mean_feature_values):
    # Calculate the bins for each feature
    x_bins = np.linspace(X[feature_x_name].min(), X[feature_x_name].max(), num_feature_bins)
    y_bins = np.linspace(X[feature_y_name].min(), X[feature_y_name].max(), num_feature_bins)
    
    # Create a grid for the binned features
    grid_x, grid_y = np.meshgrid(x_bins, y_bins)
    
    # Initialize an array to store the predictions
    Z_predicted = np.zeros((num_feature_bins, num_feature_bins))
    
    # Iterate over the grid and predict Ozone
    for i in range(num_feature_bins):
        for j in range(num_feature_bins):
            # Construct the feature vector for prediction with mean values for other features
            features = mean_feature_values.copy()
            features[feature_x_name] = x_bins[i]
            features[feature_y_name] = y_bins[j]
            
            # Scale the features
            features_scaled = scaler.transform([features])
            
            # Predict Ozone and store in Z_predicted
            Z_predicted[j, i] = model.predict(features_scaled)
    
    return grid_x, grid_y, Z_predicted

# Now, generate contour plots for each pair of features
for i, feature_i in enumerate(features_to_display):
    for j, feature_j in enumerate(features_to_display):
        if i >= j:  # Avoid duplicate pairs
            continue
        
        # Bin the features and predict Ozone
        grid_x, grid_y, Z_predicted = predict_and_bin_features(feature_i, feature_j, model, scaler, mean_feature_values)
        
        # Calculate the range of predicted Ozone values and create bins
        min_ozone = Z_predicted.min()
        max_ozone = Z_predicted.max()
        ozone_bins = np.linspace(min_ozone, max_ozone, num_ozone_bins)
        
        # Plot the contour using the bins
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(grid_x, grid_y, Z_predicted, levels=ozone_bins, cmap='coolwarm')
        fig.colorbar(contour, ax=ax, label='Predicted Ozone', format=formatter)
        
        ax.set_title(f'Predicted Ozone Levels for {feature_i} vs {feature_j}')
        ax.set_xlabel(feature_i)
        ax.set_ylabel(feature_j)
        
        # Save the current figure
        plt.tight_layout()
        plt.savefig(f'contour_{feature_i}_vs_{feature_j}.png', transparent=True, dpi=200)
        plt.show()

# Define the number of bins for the features and the predicted Ozone
num_feature_bins = 7
num_ozone_bins = 7

# Number of features to display
num_features = len(features_to_display)

# Create a figure with a grid of subplots
fig, axes = plt.subplots(nrows=num_features, ncols=num_features, figsize=(20, 15))

# Adjust the spacing of the subplots
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.4, hspace=0.4)

# Formatter for the colorbar ticks to have 1 decimal place
formatter = FormatStrFormatter('%.0f')

# Loop over the features and create contour plots for each pair
for i, feature_i in enumerate(features_to_display):
    for j, feature_j in enumerate(features_to_display):
        ax = axes[i, j]

        # Hide the upper triangle and the diagonal plots
        if i >= j:
            ax.set_visible(False)
            continue

        # Bin the features and predict Ozone
        grid_x, grid_y, Z_predicted = predict_and_bin_features(feature_i, feature_j, model, scaler, mean_feature_values)
        
        # Calculate the range of predicted Ozone values and create bins
        min_ozone = Z_predicted.min()
        max_ozone = Z_predicted.max()
        ozone_bins = np.linspace(min_ozone, max_ozone, num_ozone_bins)
        
        # Create the contour plot
        contour = ax.contourf(grid_x, grid_y, Z_predicted, levels=ozone_bins, cmap='coolwarm')
        
        # Create the colorbar and apply the formatter
        cbar = fig.colorbar(contour, ax=ax, format=formatter)
        cbar.ax.set_title('')
        
        # Set the title and labels
        # ax.set_title(f'{feature_i} vs {feature_j}')
        ax.set_xlabel(feature_i)
        ax.set_ylabel(feature_j)
  
# Save the current figure
plt.tight_layout()
plt.savefig('Contour_combined.png', transparent=True, dpi=200)
        
# Show the plot matrix
plt.show()

