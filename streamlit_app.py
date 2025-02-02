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

# Organize layout into tabs
tabs = st.tabs(["Training", "Prediction"])

with tabs[0]:
    st.subheader("Dataset")
    if st.checkbox("Show raw data", help="Check to display the raw dataset"):
        st.write(X.head())
        st.write(f"Target labels: {iris.target_names}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model with loading spinner
    with st.spinner("Training model..."):
        try:
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.success("Model training completed successfully!")
        except Exception as e:
            st.error(f"Error during model training: {e}")

    # Model performance
    st.subheader("Model Performance")
    st.write("Classification report of the model's performance on the test set", help="This report shows the precision, recall, and F1-score for each class.")
    try:
        report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
        st.write(pd.DataFrame(report).transpose())
    except Exception as e:
        st.error(f"Error during model performance evaluation: {e}")

    # Feature importance visualization
    st.subheader("Feature Importance")
    st.write("Bar chart showing the importance of each feature in the model", help="This chart shows how important each feature is in making predictions.")
    try:
        importance = pd.Series(model.feature_importances_, index=X.columns)
        fig, ax = plt.subplots()
        importance.plot(kind="barh", ax=ax, color="skyblue")
        plt.title("Feature Importance")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error during feature importance visualization: {e}")

with tabs[1]:
    st.subheader("Make Predictions")
    st.write("Use the sliders to input feature values and get real-time predictions", help="Adjust the sliders to set the values for each feature and click 'Predict' to see the predicted class.")
    input_data = {}
    for feature in X.columns:
        input_data[feature] = st.slider(f"{feature}", float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()), help=f"Set the value for {feature}")

    if st.button("Predict"):
        with st.spinner("Making prediction..."):
            try:
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)
                st.success(f"Predicted Class: {iris.target_names[prediction[0]]}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
