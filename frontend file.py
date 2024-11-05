# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from lime.lime_tabular import LimeTabularExplainer

# Set up the Streamlit app
st.set_page_config(page_title="Disease Prediction App", page_icon=":microscope:")
st.title("Disease Prediction Based on Blood Test Results")
st.write("This application uses machine learning models to predict diseases based on blood test results.")

# Load the dataset
data = pd.read_csv('blood_samples_dataset_test.csv')

# Preprocess data
X = data.drop('Disease', axis=1)
y = data['Disease']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Apply SMOTE for balancing the dataset
smote = SMOTE(k_neighbors=3, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# Train and evaluate each model to find the best one
best_model = None
best_accuracy = 0

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train_balanced)
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    st.write(f"{model_name} Accuracy: {accuracy:.4f}")

    # Update the best model if the current one is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

st.write(f"**Best Model:** {best_model.__class__.__name__} with Accuracy: {best_accuracy:.4f}")

# User input for prediction
st.sidebar.header("Input Blood Test Features")
input_values = []
for feature in X.columns:
    value = st.sidebar.slider(feature, 0.0, 1.0, 0.5, 0.01)
    input_values.append(value)

# Scale input data for prediction
input_scaled = scaler.transform([input_values])

# Make predictions using the best model
if st.sidebar.button("Predict Disease"):
    predictions = best_model.predict(input_scaled)
    predicted_class = label_encoder.inverse_transform(predictions)[0]
    st.write(f"### Predicted Disease: **{predicted_class}**")

    # Set up LIME
    explainer = LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=X.columns.tolist(),
        class_names=label_encoder.classes_,
        mode='classification'
    )

    # Explain prediction
    exp = explainer.explain_instance(
        input_scaled[0],
        best_model.predict_proba,
        num_features=10
    )

    # Display explanation results
    st.write("### Explanation of the Prediction")
    for feature, weight in exp.as_list():
        st.write(f"- **{feature}**: {weight:.4f}")

    # Bar chart of feature importance
    st.write("### Feature Importance (Top 10)")
    feature_impact = pd.DataFrame(exp.as_list(), columns=["Feature", "Weight"]).set_index("Feature")
    st.bar_chart(feature_impact)

# Footer
st.markdown("""
    <hr style="border-top: 1px solid #bbb;">
    <footer style="text-align: center;">
        Developed with ❤️ using Streamlit | Powered by Machine Learning & LIME
    </footer>
    """, unsafe_allow_html=True)