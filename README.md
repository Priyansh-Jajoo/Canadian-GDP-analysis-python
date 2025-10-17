# Canadian Macroeconomic Analysis & Forecasting (1961-2023)

## Project Overview

This repository contains a comprehensive, end-to-end data science pipeline for analyzing and forecasting key Canadian macroeconomic indicators using over 60 years of quarterly data. The primary goal is to demonstrate a rigorous workflow in Python for time-series analysis, feature engineering, and predictive modeling.

The project involves cleaning and structuring a longitudinal dataset, performing exploratory data analysis, and systematically comparing multiple forecasting models (Naive, Exponential Smoothing, ARIMA) to identify the optimal approach. Additionally, it employs regression models to determine the statistical drivers of key economic variables.

The project demonstrates a full data science workflow, from data sourcing and cleaning to advanced modeling and interpretation of results.

![Economic Analysis Plots](economic_analysis_plots.png)

---

## Key Technical Features

- **End-to-End Pipeline: Implements a complete data science workflow from data ingestion and cleaning to model evaluation and results reporting using Python.
- **Time-Series Forecasting**:
  - Rigorously evaluates multiple time-series models using statsmodels and pmdarima.
  - The final ARIMA model demonstrates superior performance, reducing forecast error (RMSE) by 76.65% compared to the baseline Naive model.
- **Feature Importance Analysis:**:
  - A multiple linear regression model (scikit-learn) explains 99.8% of the variance in consumer spending (R²=0.9981)..
  - A Random Forest model identifies the primary driver of consumption with over 99% feature importance.
- **Automated Reporting:**: The script automatically generates and saves all analytical plots and exports a detailed, multi-sheet Excel report of the findings.
- **Alternative Implementation**: An alternative implementation of this analysis was also completed in R, showcasing methodological breadth. The repository for the R version can be found at: https://github.com/Priyansh-Jajoo/Canadian-GDP-analysis-R.

---

## Technologies & Libraries

- **Language: Python 3.x
- **Core Libraries:
pandas for data manipulation
scikit-learn for regression modeling
statsmodels & pmdarima for time-series analysis
matplotlib & seaborn for data visualization
openpyxl for report generation


---

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Priyansh-Jajoo/Canadian-GDP-analysis-python.git
    cd Canadian-GDP-analysis-python
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


    ```

3.  **Install dependencies:**
    `pip install -r requirements.txt`

4. **Run the analysis script:**
     `python Canada_GDP_Python.py`

---

## Summary of Results

The ARIMA model was identified as the optimal forecasting approach based on the lowest Root Mean Squared Error (RMSE) and highest accuracy.


| Model                 | RMSE      | Accuracy (%) | Improvement over Baseline (%) |
| :-------------------- | :-------- | :----------- | :---------------------------- |
| Naive                 | 123371.72 | 92.20%       | -                             |
| Exponential Smoothing | 97997.44  | 93.40%       | 20.57%                        |
| **ARIMA** | **28808.62** | **98.87%** | **76.65%** |

*Note: Accuracy is calculated as 100 - MAPE (Mean Absolute Percentage Error).*


