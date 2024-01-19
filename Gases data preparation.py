# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:41:26 2024

@author: lb945465
"""

import pandas as pd

# Load the datasets
file_path = r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Spyder\SHAP\O3_NOy_NOx.xlsx'
Base_results_path = r'C:\mydata\Optimal solution\ICDN\Base_results.xlsx'

PMF = pd.read_excel(Base_results_path, 
                      sheet_name='Contributions_conc',
                      index_col="Date",
                      parse_dates=["Date"])

VC = pd.read_excel(r"C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Spyder\ICDN-PMF Article\VC_clean.xlsx",
                   index_col="Date"
                   )

# Reindex VC to match the index of OCDN
PMF = PMF.divide(VC['VC_ratio'], axis=0)

# Handling negative values - Example: Setting them to zero
# Adjust this according to your data's context
PMF[PMF < 0] = 0

# Recalculate row sums
row_sums = PMF.sum(axis=1)

# Normalize each value by its row sum
normalized_PMF = PMF.div(row_sums, axis=0)

# Function to read specific columns from a worksheet
def read_specific_columns(sheet_name, columns, file_path):
    return pd.read_excel(file_path, sheet_name=sheet_name, usecols=columns)

# Columns to load
columns_to_load = ['Value.parameter', 'Value.date_local', 'Value.time_local', 'Value.sample_measurement']

def process_parameter_data(parameter_name, file_path, columns_to_load):
    # Initialize an empty dataframe
    parameter_data = pd.DataFrame()

    # Process each worksheet and append the data
    for year in range(2000, 2021):
        year_data = read_specific_columns(str(year), columns_to_load, file_path)
        parameter_data = pd.concat([parameter_data, year_data], ignore_index=True)

    # Filter the dataframe for the specified parameter
    parameter_only_data = parameter_data[parameter_data['Value.parameter'] == parameter_name]

    # Convert 'Value.date_local' and 'Value.time_local' into a single datetime column and set it as index
    parameter_only_data['Date'] = pd.to_datetime(parameter_only_data['Value.date_local'] + ' ' + parameter_only_data['Value.time_local'])
    parameter_only_data.set_index('Date', inplace=True)

    # Drop unnecessary columns
    parameter_only_data.drop(['Value.parameter', 'Value.date_local', 'Value.time_local'], axis=1, inplace=True)

    # Rename 'Value.sample_measurement' to the parameter name
    parameter_only_data.rename(columns={'Value.sample_measurement': parameter_name}, inplace=True)

    return parameter_only_data

# Apply the function for each parameter
ozone_data = process_parameter_data('Ozone', file_path, columns_to_load)
nitric_oxide_data = process_parameter_data('Nitric oxide (NO)', file_path, columns_to_load)
noy_data = process_parameter_data('Reactive oxides of nitrogen (NOy)', file_path, columns_to_load)

ozone_data = ozone_data[ozone_data != 0]
ozone_data=ozone_data*1000

# Merge the two dataframes on the 'Date' column
merged_df = pd.merge(ozone_data, normalized_PMF, on='Date', how='inner')

merged_df.to_excel("Merged Ozone and PMF.xlsx")
