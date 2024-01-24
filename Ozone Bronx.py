# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 12:38:15 2024
This script is to acquire ozone data from OpenAQ and then merge it with PMF results. 
@author: lb945465
"""

import requests
import pandas as pd

def fetch_data(url, params):
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def extract_local_date(date_dict):
    try:
        local_date = pd.to_datetime(date_dict['local'])
        return local_date
    except KeyError:
        return None

base_url = "https://api.openaq.org/v2/measurements"
csv_file = 'ozone_data.csv'


for year in range(2000, 2023):
    page = 1
    more_data = True

    while more_data:
        params = {
            'location_id': 666,
            'parameter': 'o3',
            'date_from': f'{year}-01-01T00:00:00Z',
            'date_to': f'{year}-12-31T23:59:59Z',
            'limit': 1000,
            'page': page
        }

        data = fetch_data(base_url, params)
        if not data or 'results' not in data or not data['results']:
            more_data = False
            break

        df = pd.DataFrame(data['results'])
        df['Date'] = df['date'].apply(extract_local_date)
        df_cleaned = df[['Date', 'value']].rename(columns={'value': 'Ozone Concentration'})
        df_cleaned.to_csv(csv_file, mode='a', header=not page > 1 and year == 2000, index=False)

        # Updated section starts here
        if 'meta' in data and 'found' in data['meta']:
            found_str = str(data['meta']['found'])
            if (found_str.startswith('>') and int(found_str[1:]) > params['limit'] * params['page']) or \
               (not found_str.startswith('>') and int(found_str) > params['limit'] * params['page']):
                page += 1
            else:
                more_data = False
        else:
            more_data = False

    print(f"Data fetching and saving for {year} complete.")

print("All data fetching and saving complete.")

Base_results_path = r'C:\mydata\Optimal solution\ICDN\Base_results.xlsx'
Ozone_path = r'C:\mydata\Optimal solution\ICDN\ozone_data_Bronx.csv'

PMF = pd.read_excel(Base_results_path, 
                      sheet_name='Contributions_conc',
                      index_col="Date",
                      parse_dates=["Date"])

VC = pd.read_excel(r"C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Spyder\ICDN-PMF Article\VC_clean.xlsx",
                   index_col="Date"
                   )

# Reindex VC to match the index of OCDN
PMF = PMF.divide(VC['VC_ratio'], axis=0)

# Calculate the sum of each row
row_sums = PMF.sum(axis=1)

# Normalize each value by its row sum
normalized_PMF = PMF.div(row_sums, axis=0)

# Handling negative values - Example: Setting them to zero
# Adjust this according to your data's context
PMF[PMF < 0] = 0

# Recalculate row sums
row_sums = PMF.sum(axis=1)

# Remove rows where the sum is zero to avoid division by zero
PMF = PMF[row_sums != 0]
row_sums = row_sums[row_sums != 0]

# Normalize each value by its row sum
normalized_PMF = PMF.div(row_sums, axis=0)

ozone_df = pd.read_csv(Ozone_path, 
                      index_col="Date",
                      parse_dates=["Date"])

ozone_df = ozone_df[ozone_df != 0]
ozone_df=ozone_df*1000

# Merge the two dataframes on the 'Date' column
merged_df = pd.merge(ozone_df, normalized_PMF, on='Date', how='inner')

merged_df.to_excel("Merged Ozone and PMF.xlsx")

