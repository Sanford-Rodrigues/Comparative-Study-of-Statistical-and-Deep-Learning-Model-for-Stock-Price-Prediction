# Comparative-Study-of-Statistical-and-Deep-Learning-Model-for-Stock-Price-Prediction
The study explores and compares the effectiveness of traditional econometric models against advanced deep learning architectures in forecasting the **FTSE 100 index**.

## Project Overview
The primary aim of this research is to identify the most suitable model for predicting daily adjusted close prices using historical data from the London Stock Exchange. The study evaluates four distinct approaches.

**ARIMA:** A traditional statistical model using Auto-Regressive, Integrated, and Moving Average components.

**SARIMA:** An extension of ARIMA designed to handle seasonal fluctuations in financial data.

**LSTM:** A deep learning Recurrent Neural Network (RNN) capable of capturing long-term dependencies and non-linear trends.

<img width="719" height="355" alt="image" src="https://github.com/user-attachments/assets/0763ab87-f81c-47ea-9c38-87ae9f1bc6f3" />


**Hybrid ARIMA-LSTM:** A combined framework where ARIMA handles linear patterns and LSTM models the remaining residuals.

### Data Cleaning

To ensure data quality and model stability, the following preprocessing steps were taken:

Handling Null Values: The dataset was verified using the `isnull` function, confirming there were no missing entries.

Outlier Analysis: A box plot was utilized to identify outliers in the Open, High, Low, and Close prices.

<img width="757" height="465" alt="image" src="https://github.com/user-attachments/assets/c5f8d09d-5e45-4041-88ce-ce79db7ff090" />


Strategic Retention: Outliers were intentionally retained as they represented genuine market movements rather than data anomalies.

Feature Engineering: Data was standardized using `MinMaxScaler` to a range of 0 to 1, reducing data redundancy and preventing model bias.

### Exploratory Data Analysis (EDA)

Comprehensive analysis was performed to understand the underlying characteristics of the **FTSE 100 dataset**.

Dataset Scope: The research analyzed approximately 5 years of daily stock price data (1,219 entries) ranging from January 2021 to October 2025 sourced from yahoo fincance using yfinance API and saved as csv file.

Correlation Mapping: Heatmaps and scatter plots confirmed a high correlation (above 0.95) between the independent variables (Open, High, Low) and the target variable (Adj Close).

<img width="788" height="532" alt="image" src="https://github.com/user-attachments/assets/8e5457ae-f79e-43ec-aa73-da86b5486b9a" />


Distribution Patterns: Price variables exhibited a multimodal pattern indicating different market phases, while trading volume followed a log-normal distribution.

Stationarity Testing: The Augmented Dickey-Fuller (ADF) test revealed the raw data was non-stationary, requiring first-order differencing for statistical modeling.

## Methodology & Tools

The project follows the Data Science Lifecycle, including data collection via the Yahoo Finance API (yfinance), preprocessing, and evaluation.

Statistical Tests: ADF test for stationarity, and ACF/PACF plots for parameter selection.

Evaluation Metrics: Performance was measured using **Root Mean Squared Error (RMSE), Mean Squared Error (MSE), and Mean Absolute Error (MAE)**.

Libraries: `python pmdarima` `scikit-learn` `tensorflow/keras` `matplotlib` `seaborn` `numpy` `pandas` `matplotlib` `statsmodel`

## Key Findings

Winning Model: The **LSTM model with a 60-day window implemented with Principal Component Analysis (PCA)** outperformed all other models.

 <img width="1036" height="653" alt="image" src="https://github.com/user-attachments/assets/351b2e94-da48-481a-aab9-1381a7ea9d5e" />


Dimensionality Reduction: PCA significantly enhanced performance by reducing data dimensionality and noise.

Model Comparison: While ARIMA and SARIMA were effective for short-term linear data, they struggled to adapt to the high volatility and non-linear patterns of the FTSE 100.

Optimization: The final LSTM architecture utilized a 3-layer design with the 'adam' optimizer and a 0.2 dropout rate to prevent overfitting.

## Repository Structure

`notebooks/`: Implementation of models in Google Colab.

`data/`: FTSE 100 historical datasets (2021–2025).

`visualizations/`: Correlation heatmaps, error plots, and forecast vs. actual charts.
