import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


try:
    data = pd.read_csv("data.csv")
    print("✅ data.csv loaded successfully!")
except FileNotFoundError:
    print("⚠ data.csv not found, generating dummy dataset...")
    np.random.seed(42)
    data = pd.DataFrame({
        "feature1": np.random.rand(100) * 10,
        "feature2": np.random.rand(100) * 5,
        "target": np.random.rand(100) * 50
    })
    data.to_csv("data.csv", index=False)
    print("✅ Dummy data.csv created and saved!")


print("\n--- Dataset Preview ---")
print(data.head())


data = data.dropna()
target_column = "target"

X = data.drop(columns=[target_column])
y = data[target_column]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

results = []


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({"Model": name, "MSE": mse, "R²": r2})
    
    print(f"\n{name}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")


results_df = pd.DataFrame(results)
print("\n--- Model Comparison ---")
print(results_df)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, models["Linear Regression"].predict(X_test), alpha=0.6, label="Linear Regression")
plt.scatter(y_test, models["Decision Tree"].predict(X_test), alpha=0.6, label="Decision Tree")
plt.scatter(y_test, models["Random Forest"].predict(X_test), alpha=0.6, label="Random Forest")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs Actual Comparison")
plt.legend()
plt.grid(True)
plt.show()
