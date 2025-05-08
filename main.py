import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("rental_data.csv")
months = np.arange(len(df)).reshape(-1, 1)

predictions = {}
metrics = {}

for city in df.columns[1:]:
    y = df[city].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(months, y)
    y_pred = model.predict(months)

    # Predict next month
    next_month = np.array([[len(df)]])
    next_pred = model.predict(next_month)[0][0]
    predictions[city] = next_pred

    # Evaluation
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    metrics[city] = {'MSE': mse, 'R²': r2}

    # Plot
    plt.figure(figsize=(8, 4))
    plt.title(f"Trend for {city}")
    plt.plot(df['Month'], y, label='Actual')
    plt.plot(df['Month'], y_pred, linestyle='--', label='Predicted')
    plt.xticks(rotation=45)
    plt.xlabel("Month")
    plt.ylabel("Rent (₹)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"trend_{city}.png")
    plt.close()

# Output Results
print("Next Month Predictions:")
for city, rent in predictions.items():
    print(f"{city}: ₹{rent:.2f}")

print("\nModel Evaluation Metrics:")
for city, metric in metrics.items():
    print(f"{city} - MSE: {metric['MSE']:.2f}, R²: {metric['R²']:.4f}")

