import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fbprophet import Prophet
import warnings

warnings.filterwarnings("ignore")


# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=["transaction_date"], index_col="transaction_date")
    df = df.sort_index()
    return df


# Feature Engineering
def create_features(df):
    df["lag_1"] = df["sales"].shift(1)
    df["rolling_mean_7"] = df["sales"].rolling(window=7).mean()
    df["month"] = df.index.month
    df["day_of_week"] = df.index.dayofweek
    df.dropna(inplace=True)
    return df


# Train-Test Split
def split_data(df):
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    return train, test


# ARIMA Model Training
def train_arima(train):
    model = ARIMA(train["sales"], order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit


# Random Forest Model Training
def train_rf(train):
    features = ["lag_1", "rolling_mean_7", "month", "day_of_week"]
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(train[features], train["sales"])
    return rf


# Prophet Model Training
def train_prophet(train):
    df_prophet = train.reset_index().rename(columns={"transaction_date": "ds", "sales": "y"})
    model = Prophet()
    model.fit(df_prophet)
    return model


# Model Evaluation
def evaluate_model(model, test, model_type):
    if model_type == "arima":
        predictions = model.forecast(len(test))
    elif model_type == "rf":
        features = ["lag_1", "rolling_mean_7", "month", "day_of_week"]
        predictions = model.predict(test[features])
    elif model_type == "prophet":
        future = pd.DataFrame(test.index, columns=["ds"])
        forecast = model.predict(future)
        predictions = forecast["yhat"].values

    mae = mean_absolute_error(test["sales"], predictions)
    rmse = np.sqrt(mean_squared_error(test["sales"], predictions))
    print(f"{model_type.upper()} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return predictions


# Plot Results
def plot_results(test, predictions, model_type):
    plt.figure(figsize=(10, 5))
    plt.plot(test.index, test["sales"], label="Actual Sales", color="blue")
    plt.plot(test.index, predictions, label=f"{model_type} Predictions", color="red")
    plt.legend()
    plt.title(f"{model_type} Forecasting Results")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.show()


if __name__ == "__main__":
    file_path = "data/sales_data.csv"
    df = load_data(file_path)
    df = create_features(df)
    train, test = split_data(df)

    # Train and evaluate models
    arima_model = train_arima(train)
    rf_model = train_rf(train)
    prophet_model = train_prophet(train)

    # Get Predictions
    arima_preds = evaluate_model(arima_model, test, "arima")
    rf_preds = evaluate_model(rf_model, test, "rf")
    prophet_preds = evaluate_model(prophet_model, test, "prophet")

    # Plot Forecasts
    plot_results(test, arima_preds, "ARIMA")
    plot_results(test, rf_preds, "Random Forest")
    plot_results(test, prophet_preds, "Prophet")

    # Save Best Model (Example: Using Prophet if best performance)
    joblib.dump(prophet_model, "prophet_model.pkl")