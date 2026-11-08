import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
data = pd.read_csv("diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline: scaler + GaussianNB
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GaussianNB())
])

# Train model
pipeline.fit(X_train, y_train)

# Save model
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model trained and saved!")
