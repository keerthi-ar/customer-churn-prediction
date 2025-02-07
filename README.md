# Customer Churn Prediction

## Objective:
The goal of this project is to predict which customers are likely to leave a service or subscription (churn), enabling businesses to take proactive actions to retain them.

## Dataset:
This project uses the [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle. The dataset contains customer information such as demographics, account information, and services subscribed to, with a target variable indicating whether the customer churned (left the service).

## Project Overview:
In this project, we analyze customer data to identify patterns associated with churn. We build a predictive model using Gradient Boosting Classifier and use model interpretation tools like SHAP and LIME to identify the key factors influencing churn.

### Steps Taken:
1. **Data Exploration and Preprocessing**:
   - Explored and cleaned the dataset by handling missing values and outliers.
   - Encoded categorical variables for machine learning models.
   - Split the dataset into features and target variable for model training.

2. **Model Building**:
   - Trained a Gradient Boosting Classifier model to predict customer churn.

3. **Model Interpretation**:
   - Used **SHAP** (SHapley Additive exPlanations) to interpret the model and identify important features influencing churn.
   - Used **LIME** (Local Interpretable Model-agnostic Explanations) to further explain individual predictions and understand factors affecting churn for specific customers.

4. **Model Testing**:
   - Tested the model with a sample customer to predict whether they would churn or not.

## Citation:
Kaggle. (2019). Telco Customer Churn. Retrieved from https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Installation:
To run this project on your local machine, you'll need to install the following libraries:

```bash
pip install shap lime
```
## Additional Dependencies:
The project was developed using Google Colab and requires the following libraries:

   - pandas
   - numpy
   - seaborn
   - matplotlib
   - scikit-learn
   - xgboost 

### Usage:

## Clone the repository:

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
```

## Navigate to the project directory:

```bash
cd customer-churn-prediction
```

## Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the code:
```bash
jupyter notebook customer-churn-prediction.ipynb
```
