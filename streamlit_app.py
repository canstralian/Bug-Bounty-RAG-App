# Import libraries
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Set up the Streamlit app
st.title("Machine Learning Dashboard")

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

# Display dataset
st.subheader("Dataset")
if st.checkbox("Show raw data"):
    st.write(X.head())
    st.write(f"Target labels: {iris.target_names}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model performance
st.subheader("Model Performance")
report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
st.write(pd.DataFrame(report).transpose())

# Feature importance visualization
st.subheader("Feature Importance")
importance = pd.Series(model.feature_importances_, index=X.columns)
fig, ax = plt.subplots()
importance.plot(kind="barh", ax=ax, color="skyblue")
plt.title("Feature Importance")
st.pyplot(fig)

# Make predictions with user input
st.subheader("Make Predictions")
input_data = {}
for feature in X.columns:
    input_data[feature] = st.slider(f"{feature}", float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.write(f"Predicted Class: {iris.target_names[prediction[0]]}")