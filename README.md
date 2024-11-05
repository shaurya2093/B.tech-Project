# Enhancing Healthcare Transparency: Predicting Disease with Explainable AI

## Overview
This project aims to predict potential diseases based on blood report data using machine learning algorithms. By analyzing various health indicators, the model can determine if a patient may have conditions like anemia, diabetes, or if they are generally healthy. 

## Models Used
We tested and evaluated multiple machine learning models, including:
- **Random Forest**: Accuracy - 86.73%
- **Decision Tree**: Accuracy - 65.31%
- **Naive Bayes**: Accuracy - 74.49%
- **XGBoost**: Accuracy - 95.92% (Best Model)

The best-performing model, XGBoost, was selected for final predictions due to its superior accuracy.

## Features
- User input for blood report metrics such as:
  - White Blood Cells
  - Red Blood Cells
  - Hematocrit
  - Glucose
  - BMI
  - Triglycerides
  - ALT
- Model interpretability using **LIME** (Local Interpretable Model-agnostic Explanations) to provide insights into how predictions are made.

## Installation
To run this project, you need to have Python installed. You can clone this repository and install the required libraries
