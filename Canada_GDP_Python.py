#!/usr/bin/env python
# coding: utf-8


# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.api import ExponentialSmoothing
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')

# --- UTILITY FUNCTION FOR CLEAN HEADERS ---
def print_header(title):
    """Prints a formatted header to the console."""
    print("\n" + "="*70)
    print(f"| {title.upper():^66} |")
    print("="*70)

# 2. DATA PREPROCESSING
def preprocess_data(file_path):
    """Loads and preprocesses the economic data from a CSV file."""
    print_header("Step 1: Data Preprocessing")
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully loaded '{file_path}'.")
    except FileNotFoundError:
        raise FileNotFoundError(f"ERROR: The file '{file_path}' was not found.")

    required_columns = ["date", "Value", "consumption_expenditure", "unemployment", "Inflation", "interest_rate"]
    if not all(col in data.columns for col in required_columns):
        missing = set(required_columns) - set(data.columns)
        raise ValueError(f"ERROR: Missing required columns: {missing}.")

    data['date'] = pd.to_datetime(data['date'])
    data['gdp_growth'] = data['Value'].pct_change() * 100
    data['consumption_growth'] = data['consumption_expenditure'].pct_change() * 100
    data = data.dropna(subset=['gdp_growth', 'consumption_growth']).reset_index(drop=True)
    print("Data validation and feature engineering complete.")
    return data

# 3. VISUALIZATION
def create_plots(data):
    """Creates and saves a 2x2 grid of key economic visualizations."""
    print_header("Step 2: Generating Visualizations")
    plt.style.use('seaborn-v0_8-deep')
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Economic Analysis of Canadian Data (1961-2023)', fontsize=20)

    axes[0, 0].plot(data['date'], data['Value'], label='Real GDP', color='navy')
    axes[0, 0].plot(data['date'], data['consumption_expenditure'], label='Consumption', color='firebrick')
    axes[0, 0].set_title('Real GDP and Consumption Trends')
    axes[0, 0].set_ylabel('Millions of Chained 2012 CAD')
    axes[0, 0].legend()
    axes[0, 1].plot(data['date'], data['unemployment'], label='Unemployment Rate', color='purple')
    axes[0, 1].plot(data['date'], data['Inflation'], label='Inflation Rate', color='orange')
    axes[0, 1].plot(data['date'], data['interest_rate'], label='Interest Rate', color='green')
    axes[0, 1].set_title('Unemployment, Inflation, and Interest Rates')
    axes[0, 1].set_ylabel('Rate (%)')
    axes[0, 1].legend()
    sns.regplot(data=data, x='Value', y='consumption_expenditure', ax=axes[1, 0], line_kws={"color": "red"})
    axes[1, 0].set_title('Consumption vs GDP Relationship')
    axes[1, 0].set_xlabel('Real GDP')
    axes[1, 0].set_ylabel('Real Consumption')
    sns.regplot(data=data, x='unemployment', y='Inflation', ax=axes[1, 1], lowess=True, line_kws={"color": "red"})
    axes[1, 1].set_title('Phillips Curve (Inflation vs Unemployment)')
    axes[1, 1].set_xlabel('Unemployment Rate (%)')
    axes[1, 1].set_ylabel('Inflation Rate (%)')
    for ax_row in axes:
        for ax in ax_row:
            ax.tick_params(axis='x', rotation=45)
            ax.xaxis.set_major_locator(mdates.YearLocator(5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('economic_analysis_plots.png')
    print("Visualizations saved to 'economic_analysis_plots.png'.")
    return fig

# 4. TIME SERIES MODEL EVALUATION
def evaluate_time_series_models(data):
    """Evaluates multiple time series models on quarterly GDP data."""
    print_header("Step 3: Time Series Forecasting Evaluation")

    seasonal_periods = 4
    gdp_quarterly = data.set_index('date')['Value'].resample('Q').mean().dropna()
    train_size = int(len(gdp_quarterly) * 0.8)
    train, test = gdp_quarterly[:train_size], gdp_quarterly[train_size:]

    predictions = {}

    # Model 1: Naive Baseline -- THIS IS THE FIXED SECTION
    last_train_value = train.iloc[-1]
    predictions['Naive'] = pd.Series([last_train_value] * len(test), index=test.index)

    # Model 2: Exponential Smoothing (with safety check)
    if len(train) >= 2 * seasonal_periods:
        es_model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=seasonal_periods, trend='add', damped_trend=True).fit()
        predictions['Exponential Smoothing'] = es_model.forecast(len(test))
    else:
        print(f"\n[Warning] Skipping Exponential Smoothing model: Training data is too small for seasonality (needs {2*seasonal_periods} points, has {len(train)}).")

    # Model 3: Auto ARIMA (with safety check)
    if len(train) >= 10:
        arima_model = auto_arima(train, seasonal=True, m=seasonal_periods, suppress_warnings=True, stepwise=True, error_action='ignore')
        predictions['ARIMA'] = arima_model.predict(n_periods=len(test))
    else:
        print(f"\n[Warning] Skipping ARIMA model: Training data is too small (needs at least 10 points, has {len(train)}).")

    # Evaluate models that were successfully run
    results = []
    for name, pred in predictions.items():
        rmse = np.sqrt(mean_squared_error(test, pred))
        mape = np.mean(np.abs((test - pred) / test)) * 100
        accuracy = 100 - mape
        results.append({'Model': name, 'RMSE': rmse, 'MAPE (%)': mape, 'Accuracy (%)': accuracy})
    
    results_df = pd.DataFrame(results).set_index('Model')
    
    if 'Naive' in results_df.index:
        baseline_rmse = results_df.loc['Naive', 'RMSE']
        for model_name in ['ARIMA', 'Exponential Smoothing']:
            if model_name in results_df.index:
                model_rmse = results_df.loc[model_name, 'RMSE']
                improvement = (baseline_rmse - model_rmse) / baseline_rmse * 100
                results_df.loc[model_name, 'Improvement over Baseline (%)'] = improvement
    
    print("\nModel Comparison on Test Data (Predicting Future GDP):")
    print(results_df.to_string(float_format="%.2f"))
    print("\n* Accuracy = 100 - MAPE (Mean Absolute Percentage Error)")
    print("* Improvement = % reduction in RMSE compared to the Naive model")
    return results_df

# 5. REGRESSION AND MACHINE LEARNING
def perform_regression_and_ml(data):
    """Performs regression and Random Forest analysis."""
    print_header("Step 4: Regression & Machine Learning Analysis")
    X = data[['Value', 'unemployment', 'Inflation', 'interest_rate']]
    y = data['consumption_expenditure']

    multi_reg_model = LinearRegression().fit(X, y)
    r2_multi_reg = multi_reg_model.score(X, y)
    print(f"Multiple Regression R-squared (to predict Consumption): {r2_multi_reg:.4f}")

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    
    print("\nRandom Forest Feature Importance (Drivers of Consumption):")
    print(importance.to_string(index=False))
    return {'r2': r2_multi_reg, 'feature_importance': importance}

# 6. ECONOMIC ANALYSIS
def calculate_economic_metrics(data):
    """Calculates key economic metrics like MPC and Okun's Law coefficient."""
    print_header("Step 5: Specific Economic Metrics Calculation")

    X_gdp = data[['Value']]
    y_cons = data['consumption_expenditure']
    mpc = LinearRegression().fit(X_gdp, y_cons).coef_[0]
    print(f"Marginal Propensity to Consume (MPC): {mpc:.3f}")
    
    X_growth = data[['gdp_growth']]
    y_unemp_change = data['unemployment'].diff().dropna()
    okun_coef = LinearRegression().fit(X_growth.iloc[1:], y_unemp_change).coef_[0]
    print(f"Okun's Law Coefficient: {okun_coef:.3f}")
    return {'mpc': mpc, 'okun_coef': okun_coef}

# 7. EXPORT RESULTS
def export_results(data, ts_results, ml_results, econ_metrics):
    """Exports all analysis results to a multi-sheet Excel file."""
    print_header("Step 6: Exporting Full Report")
    filename = 'economic_analysis_report.xlsx'
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        data.describe().to_excel(writer, sheet_name='Descriptive_Stats')
        ts_results.to_excel(writer, sheet_name='TimeSeries_Evaluation')
        ml_results['feature_importance'].to_excel(writer, sheet_name='ML_Feature_Importance', index=False)
        summary_data = {'Metric': ['R-squared', 'MPC', 'Okun\'s Coef'],
                        'Value': [ml_results['r2'], econ_metrics['mpc'], econ_metrics['okun_coef']]}
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Key_Metrics', index=False)
    print(f"All results exported to '{filename}'.")

# 8. FINAL SUMMARY REPORT (Updated Version)
def print_summary_report(ts_results, ml_results, econ_metrics):
    """Prints a final, clean summary of the key findings."""
    print_header("Final Analysis Report Summary")

    # Best forecasting model
    if not ts_results.empty and 'Improvement over Baseline (%)' in ts_results.columns:
        if ts_results['Improvement over Baseline (%)'].notna().any():
            best_model_name = ts_results['Improvement over Baseline (%)'].idxmax()
            best_model_accuracy = ts_results.loc[best_model_name, 'Accuracy (%)']
            best_model_improvement = ts_results.loc[best_model_name, 'Improvement over Baseline (%)']
            
            print("1. Forecasting Performance:")
            print(f"   - The best forecasting model for GDP, based on RMSE, is '{best_model_name}'.")
            
            # Check if accuracy is NaN and print a user-friendly message
            if pd.isna(best_model_accuracy):
                print("   - Accuracy (based on MAPE) could not be calculated due to data properties, but RMSE is a robust measure.")
            else:
                print(f"   - It achieved an accuracy of {best_model_accuracy:.2f}% on unseen data.")
            
            print(f"   - This was an improvement of {best_model_improvement:.2f}% in forecast error (RMSE) over the simple baseline model.")
        else:
            print("1. Forecasting Performance:")
            print("   - No advanced forecasting models could be run due to data size. Only the Naive baseline was evaluated.")
    else:
        print("1. Forecasting Performance:")
        print("   - Forecasting models were not evaluated due to insufficient data.")
    
    # Most important feature
    most_important_feature = ml_results['feature_importance'].loc[0, 'Feature']
    
    print("\n2. Key Economic Drivers:")
    print(f"   - The Random Forest model identified '{most_important_feature}' (Real GDP) as the most significant driver of consumer spending.")
    print(f"   - The multiple regression model was able to explain {ml_results['r2']:.1%} of the variance in consumption (R²).")
    
    print("\n3. Calculated Economic Metrics:")
    print(f"   - Marginal Propensity to Consume (MPC): {econ_metrics['mpc']:.3f}")
    print(f"   - Okun's Law Coefficient: {econ_metrics['okun_coef']:.3f}")

# 9. MAIN FUNCTION
def main():
    """Main function to run the entire analysis pipeline."""
    file_path = "Canada_Economic_Data_1961-2023.csv"
    
    try:
        data = preprocess_data(file_path)
        create_plots(data)
        ts_results = evaluate_time_series_models(data)
        ml_results = perform_regression_and_ml(data)
        econ_metrics = calculate_economic_metrics(data)
        export_results(data, ts_results, ml_results, econ_metrics)
        print_summary_report(ts_results, ml_results, econ_metrics)
        
        print("\n" + "="*70)
        print("| ANALYSIS FINISHED SUCCESSFULLY".ljust(68) + "|")
        print("="*70)

    except (FileNotFoundError, ValueError) as e:
        print(f"\nANALYSIS HALTED: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()




