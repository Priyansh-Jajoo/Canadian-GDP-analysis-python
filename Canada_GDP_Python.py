#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import skew
from sklearn.metrics import r2_score
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# 1. Data Preprocessing
def preprocess_data(file_path):
    # Read data
    try:
        data = pd.read_excel(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_path} not found")

    required_columns = ["date", "Value", "consumption_expenditure", "unemployment", "Inflation"]
    if not all(col in data.columns for col in required_columns):
        raise ValueError("Missing required columns in the dataset")

    data['date'] = pd.to_datetime(data['date'])
    
    # Calculate growth rates
    data['gdp_growth'] = data['Value'].pct_change() * 100
    data['consumption_growth'] = data['consumption_expenditure'].pct_change() * 100
    
    # Remove rows with NA growth rates
    data = data.dropna(subset=['gdp_growth', 'consumption_growth'])
    
    return data

# 2. Visualization Functions
def create_plots(data):
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(15, 12))
    
    # GDP and Consumption Over Time
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(data['date'], data['Value'], label='GDP', color='blue')
    ax1.plot(data['date'], data['consumption_expenditure'], label='Consumption', color='red')
    ax1.set_title('Real GDP and Consumption Trends\nCanada (1961-2023)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Millions of Chained 2012 CAD')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    # Format dates for better readability
    ax1.xaxis.set_major_locator(mdates.YearLocator(5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    
    # Unemployment and Inflation
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(data['date'], data['unemployment'], label='Unemployment', color='purple')
    ax2.plot(data['date'], data['Inflation'], label='Inflation', color='orange')
    ax2.set_title('Unemployment and Inflation Trends')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Rate (%)')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    # Format dates for better readability
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # GDP vs Consumption Scatter
    ax3 = plt.subplot(2, 2, 3)
    sns.regplot(data=data, x='Value', y='consumption_expenditure', ax=ax3)
    ax3.set_title('Consumption vs GDP Relationship')
    ax3.set_xlabel('Real GDP')
    ax3.set_ylabel('Real Consumption')
    
    # Phillips Curve
    ax4 = plt.subplot(2, 2, 4)
    sns.regplot(data=data, x='unemployment', y='Inflation', ax=ax4, lowess=True)
    ax4.set_title('Phillips Curve Analysis')
    ax4.set_xlabel('Unemployment Rate (%)')
    ax4.set_ylabel('Inflation Rate (%)')
    
    plt.tight_layout()
    return fig

# 3. Statistical Analysis
def calculate_statistics(data):
    stats = {
        'gdp_mean': data['Value'].mean(),
        'gdp_sd': data['Value'].std(),
        'gdp_skew': skew(data['Value'].dropna()),
        'cons_mean': data['consumption_expenditure'].mean(),
        'cons_sd': data['consumption_expenditure'].std(),
        'cons_skew': skew(data['consumption_expenditure'].dropna()),
        'unemp_mean': data['unemployment'].mean(),
        'unemp_sd': data['unemployment'].std(),
        'inf_mean': data['Inflation'].mean(),
        'inf_sd': data['Inflation'].std()
    }
    return pd.DataFrame([stats])

# 4. Regression Analysis
def perform_regression(data):
    from sklearn.linear_model import LinearRegression
    
    # Basic consumption function
    X1 = data[['Value']]
    y = data['consumption_expenditure']
    model1 = LinearRegression().fit(X1, y)
    
    # Multiple regression
    X2 = data[['Value', 'unemployment', 'Inflation']]
    model2 = LinearRegression().fit(X2, y)
    
    return {
        'model1': model1,
        'model2': model2,
        'r2_model1': model1.score(X1, y),
        'r2_model2': model2.score(X2, y),
        'mpc': model1.coef_[0]  # Marginal Propensity to Consume
    }

# 5. Time Series Analysis
def perform_time_series_analysis(data):
    monthly_gdp = data.set_index('date')['Value'].resample('M').mean()
    monthly_consumption = data.set_index('date')['consumption_expenditure'].resample('M').mean()
    
    # GDP forecast
    gdp_model = auto_arima(monthly_gdp, seasonal=False, m=12, suppress_warnings=True)
    gdp_forecast = pd.DataFrame(
        gdp_model.predict(n_periods=5),
        columns=['forecast']
    )
    
    # Consumption forecast
    cons_model = auto_arima(monthly_consumption, seasonal=False, m=12, suppress_warnings=True)
    cons_forecast = pd.DataFrame(
        cons_model.predict(n_periods=5),
        columns=['forecast']
    )
    
    return gdp_forecast, cons_forecast

# 6. Machine Learning
def train_random_forest(data):
    X = data[['Value', 'unemployment', 'Inflation']]
    y = data['consumption_expenditure']
    
    rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
    rf_model.fit(X, y)
    
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return rf_model, importance

# 7. Economic Analysis
def calculate_economic_metrics(data, regression_results):
    # Okun's Law coefficient
    okun_model = LinearRegression()
    okun_model.fit(data[['gdp_growth']], data['unemployment'])
    okun_coef = okun_model.coef_[0]
    
    return {
        'mpc': regression_results['mpc'],
        'okun_coef': okun_coef
    }

# 8. Results Export
def export_results(stats, forecasts, regression_metrics, importance, filename='economic_analysis_results.xlsx'):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        stats.to_excel(writer, sheet_name='Statistics', index=False)
        forecasts[0].to_excel(writer, sheet_name='GDP_Forecast', index=False)
        forecasts[1].to_excel(writer, sheet_name='Consumption_Forecast', index=False)
        importance.to_excel(writer, sheet_name='RF_Importance', index=False)

def main():
    try:
        # Load and preprocess data
        data = preprocess_data("Canada_Real_GDP.xlsx")
        
        # Generate visualizations
        fig = create_plots(data)
        plt.savefig('economic_analysis_plots.png')
        
        # Perform analyses
        stats = calculate_statistics(data)
        regression_results = perform_regression(data)
        forecasts = perform_time_series_analysis(data)
        rf_model, importance = train_random_forest(data)
        economic_metrics = calculate_economic_metrics(data, regression_results)
        
        # Export results
        export_results(stats, forecasts, regression_results, importance)
        
        # Print key findings
        print("\nKey Findings:")
        print(f"1. Marginal Propensity to Consume: {economic_metrics['mpc']:.3f}")
        print(f"2. Okun's Law Coefficient: {economic_metrics['okun_coef']:.3f}")
        print(f"3. R-squared (Multiple Regression): {regression_results['r2_model2']:.3f}")
        print(f"4. Most Important Predictor: {importance.iloc[0]['feature']}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    try:
        from sklearn.linear_model import LinearRegression  
        
        # Load and preprocess data
        data = preprocess_data("Canada_Real_GDP.xlsx")
        print("\n Data Preprocessing Completed")

        # Generate visualizations
        fig = create_plots(data)
        plt.savefig('economic_analysis_plots.png')
        print("\n Visualization Generated and Saved as 'economic_analysis_plots.png'")

        # Perform Statistical Analysis
        stats = calculate_statistics(data)
        print("\n Statistical Analysis:")
        print(stats.to_string(index=False))

        # Perform Regression Analysis
        regression_results = perform_regression(data)
        print("\n Regression Analysis Results:")
        print(f"R² (Simple Regression, GDP → Consumption): {regression_results['r2_model1']:.3f}")
        print(f"R² (Multiple Regression, GDP+Unemployment+Inflation → Consumption): {regression_results['r2_model2']:.3f}")
        print(f"Marginal Propensity to Consume (MPC): {regression_results['mpc']:.3f}")

        # Perform Time Series Forecasting
        try:
            forecasts = perform_time_series_analysis(data)
            print("\n Time Series Forecasting (Next 5 Months):")
            print("\nGDP Forecast:")
            print(forecasts[0].to_string(index=False))
            print("\nConsumption Forecast:")
            print(forecasts[1].to_string(index=False))
        except Exception as e:
            print("\n Time Series Analysis Failed:", str(e))
            forecasts = (None, None)

        # Train Machine Learning Model (Random Forest)
        rf_model, importance = train_random_forest(data)
        print("\n Random Forest Feature Importance:")
        print(importance.to_string(index=False))

        # Economic Analysis (Okun's Law, MPC)
        economic_metrics = calculate_economic_metrics(data, regression_results)
        print("\n Economic Analysis Results:")
        print(f"Okun's Law Coefficient: {economic_metrics['okun_coef']:.3f}")
        print(f"Marginal Propensity to Consume (MPC): {economic_metrics['mpc']:.3f}")


    except Exception as e:
        print(f"\n An error occurred: {str(e)}")

if __name__ == "__main__":
    main()





