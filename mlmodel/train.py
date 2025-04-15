import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import mlflow


# Load data
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_scaled, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Optional: Track with MLflow
mlflow.start_run()
mlflow.log_param("model_type", "LogisticRegression")
mlflow.log_metric("accuracy", model.score(X_scaled, y))
mlflow.sklearn.log_model(model, "model")
mlflow.end_run()

print("✅ model.pkl created successfully")


