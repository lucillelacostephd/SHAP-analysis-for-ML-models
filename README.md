# SHAP Analysis for Machine Learning Models on Ozone

This repository contains the scripts and data used in my study:

**Borlaza-Lacoste et al. (2024).**  
*Long-term contributions of VOC sources and their link to ozone pollution in Bronx, New York City.*  
Published in *Environment International*.  
[DOI link](http://dx.doi.org/10.2139/ssrn.4830442)

The repo demonstrates how machine learning models (specifically XGBoost) can be combined with SHAP (SHapley Additive exPlanations) to interpret the influence of VOC sources and meteorological variables on ozone concentrations.

---

## Repository structure

SHAP-analysis-for-ML-models/
ozone_data.csv # Example dataset (VOC + O3 + meteorology)

ozone_data_Bronx.csv # Bronx-specific dataset

Gases data preparation.py # Data cleaning and feature preparation

Ozone Bronx.py # XGBoost training and prediction

SHAP.py # SHAP analysis (global + local explanations)

force_plot.html # Example SHAP force plot output

README.md # You are here

## Getting started

### 1. Set up environment

Create a conda environment with pinned versions (Python 3.7.16 is recommended for compatibility):

```bash
conda create -n shap-ml python=3.7.16
conda activate shap-ml
pip install numpy pandas scikit-learn xgboost shap matplotlib

```
Optional: install jupyterlab if you prefer running scripts as notebooks.

### 2. Input data schema

The dataset (ozone_data.csv) must contain the following columns:
  date (datetime) – measurement date/time
  ozone_ppb – ozone concentration (ppb)
  nox_ppb – nitrogen oxides (ppb)
  temp_C, rh_pct, ws_mps, pblh_m – meteorological variables
  pmf_* – PMF-resolved VOC source contribution factors
  doy, hour, dow – day-of-year, hour, and day-of-week (used for cyclic encoding)

### 3. Run the workflow

```bash
# Prepare data
python "Gases data preparation.py"

# Train the XGBoost model
python "Ozone Bronx.py"

# Run SHAP analysis
python "SHAP.py"
```
#### Outputs:

Global SHAP bar plots and summary plots will be generated in your working directory.

Local explanation example is saved as force_plot.html (open in browser).

## Example results

  Global SHAP summary plot – shows which PMF factors and meteorological variables most strongly influence ozone.
 
  Force plot (HTML) – illustrates contributions for an individual prediction.
  
  Predicted vs. Observed O₃ – model evaluation (RMSE, MAE, R²).

### Caveats

SHAP highlights associations, not causality. Factor importance can reflect collinearity and regime dependence (e.g., VOC-limited vs. NOx-limited ozone formation).

Autocorrelation: models should use time-blocked validation, not random splits.

Data in this repo is simplified for demonstration; replicate with caution.

### Citation

If you use this code or workflow, please cite:

Borlaza-Lacoste, L. et al. (2024). Long-term contributions of VOC sources and their link to ozone pollution in Bronx, New York City. Environment International.

