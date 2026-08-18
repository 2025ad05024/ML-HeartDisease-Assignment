
# Heart Disease Prediction Assignment

## Project Overview
This project implements a machine learning pipeline to predict the presence of heart disease based on a comprehensive dataset from Kaggle. The goal of this assignment was to demonstrate proficiency in data loading, exploratory data analysis (EDA), data preprocessing, model selection, training, evaluation, and hyperparameter tuning.

## Dataset
The dataset used is the "Heart Failure Prediction Dataset" by fedesoriano, available on Kaggle. It contains various patient attributes such as age, sex, chest pain type, resting blood pressure, cholesterol, and other relevant medical indicators, with `HeartDisease` as the target variable (1 for heart disease, 0 for no heart disease).

## Project Structure
-   `heart.csv`: The raw dataset used for this project.
-   `heart_disease_prediction.ipynb` (or similar notebook name): This Jupyter/Colab notebook contains the complete source code, including all steps of the machine learning pipeline.
-   `requirements.txt`: Lists all Python dependencies required to run the notebook.
-   `README.md`: This file, providing an overview of the project.

## Machine Learning Pipeline
The following steps were performed:
1.  **Problem Definition**: Defined the problem as binary classification for heart disease prediction.
2.  **Data Loading**: Loaded `heart.csv` into a pandas DataFrame.
3.  **Exploratory Data Analysis (EDA)**: Analyzed data distributions, identified outliers, and examined relationships between features and the target variable.
4.  **Data Preprocessing**: Handled implausible '0' values in 'RestingBP' and 'Cholesterol' by imputing them with the median. Categorical features (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) were one-hot encoded, and numerical features were scaled using `StandardScaler`.
5.  **Data Splitting**: The data was split into training (80%) and testing (20%) sets, stratified by the target variable.
6.  **Model Selection & Training**: Implemented and trained five classification models: Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Tree, and Random Forest.
7.  **Model Evaluation**: Evaluated initial model performance using Accuracy, Precision, Recall, F1-Score, and ROC AUC.
8.  **Hyperparameter Tuning**: Applied `GridSearchCV` to optimize hyperparameters for Logistic Regression, K-Nearest Neighbors, Support Vector Machine, and Random Forest.
9.  **Conclusion & Reporting**: Summarized the findings, compared model performances, and identified the best model.

## Key Findings
-   The dataset required cleaning, specifically addressing implausible '0' values in `RestingBP` and `Cholesterol`.
-   One-hot encoding of categorical features and scaling of numerical features were crucial preprocessing steps.
-   After hyperparameter tuning, the **K-Nearest Neighbors** model emerged as the top performer.
-   **Best Tuned Model Performance (K-Nearest Neighbors on Test Set)**:
    -   ROC AUC: `0.9385`
    -   Accuracy: `0.8696`
    -   F1-Score: `0.8835`

## How to Run the Code

### 1. Clone the Repository
First, clone this repository to your local machine:
```bash
git clone [YOUR_REPOSITORY_URL]
cd [YOUR_REPOSITORY_NAME]
```

### 2. Install Dependencies
It is recommended to use a virtual environment. Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Obtain the Dataset
Download the `heart.csv` dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) and place it in the root directory of the cloned repository (or `/content/` if running in Google Colab).

### 4. Run the Notebook
Open and run the Jupyter/Colab notebook (`heart_disease_prediction.ipynb`) to execute the entire machine learning pipeline:

-   **Jupyter Notebook**: `jupyter notebook heart_disease_prediction.ipynb`
-   **Google Colab**: Upload the notebook to Google Colab and run all cells.

## Further Steps (Optional)
-   Exploring more advanced feature engineering techniques.
-   Investigating ensemble methods like stacking or boosting.
-   Applying deep learning models for comparison.
-   Using model interpretability tools (e.g., SHAP, LIME) to better understand predictions.

---
