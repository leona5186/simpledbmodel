import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# data loading
df = pd.read_csv("sales.csv")

X = df[["units_sold", "region", "product"]]
y = df["revenue"]

# onehot
preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(), ["region", "product"])
], remainder="passthrough")

# pipeline
model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X, y)

# model save
joblib.dump(model, "revenue_model.pkl")
