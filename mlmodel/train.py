import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv')
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

model = LogisticRegression()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

import mlflow
mlflow.start_run()
mlflow.log_param("model", "LogisticRegression")
mlflow.log_metric("accuracy", model.score(X, y))
mlflow.sklearn.log_model(model, "model")
mlflow.end_run()

