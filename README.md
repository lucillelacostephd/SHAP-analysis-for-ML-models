# Applying ML + SHAP analysis to deconvolute the impact of VOC sources on ozone pollution analysis 
This repository stores the updated Python code and data to reproduce the analysis presented in the article:

Lucille Borlaza-Lacoste, Md. Aynul Bari, Cheng-Hsuan Lu, Philip K. Hopke,
Long-term contributions of VOC sources and their link to ozone pollution in Bronx, New York City,
Environment International, 2024, 108993, ISSN 0160-4120, https://doi.org/10.1016/j.envint.2024.108993.

This work performed several machine learning techniques to determine the influence of PMF-resolved VOC sources on ozone pollution in an urban site in Bronx, New York City. The model that showed best performance was XGBoost, compared to Random Forest and Extremely Randomized Trees models. All data was acquired from USEPA AirNow. SHAP (SHapley Additive exPlanations) approach was utilized to explain and visualize the output of the machine learning models. 
