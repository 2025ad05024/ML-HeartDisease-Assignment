
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

 ## Model Performance Comparison
| Model                   | Accuracy  | Precision | Recall   | F1-Score | MCC     | AUC  |

|--------------------------|-----------|-----------|----------|----------|---------|------|

| Logistic Regression      | 0.88587   | 0.885714  | 0.911765 | 0.898551 | 0.7686  | 0.90 |

| Random Forest            | 0.880435  | 0.877358  | 0.911765 | 0.894231 | 0.757588| 0.89 |

| Naive Bayes              | 0.820125  | 0.800126  | 0.813470 | 0.800001 | 0.62131 | 0.82 |

| Support Vector Machine   | 0.875     | 0.862385  | 0.921569 | 0.890995 | 0.747131| 0.88 |

| K-Nearest Neighbors      | 0.86413   | 0.881188  | 0.872549 | 0.876847 | 0.725384| 0.86 |

| Decision Tree            | 0.804348  | 0.823529  | 0.823529 | 0.823529 | 0.604017| 0.81 |

## Model Performance Observations

| ML Model Name            | Observation about model performance |

|---------------------------|-------------------------------------|

| \*\*Logistic Regression\*\*   | This model consistently delivered strong results and showed modest gains after tuning, ultimately achieving the highest MCC score. It stands out as the most effective predictor of heart disease in this study.Untuned MCC: 0.7686Tuned MCC: 0.7797 |

| \*\*Decision Tree\*\*         | Among the untuned models, the Decision Tree recorded low MCC, highlighting its limited effectiveness for this dataset. It was therefore excluded from the final tuned comparison, as tuning either did not improve its MCC or yielded no valid result.Untuned MCC: 0.6040Tuned MCC: did not improve significantly |

| \*\*K-Nearest Neighbors\*\*   | The kNN classifier delivered moderate performance overall, and hyperparameter tuning provided a small but positive improvement in its MCC score, indicating a slight boost in predictive reliability. Untuned MCC: 0.7254 Tuned MCC: 0.7356 |

| \*\*Naive Bayes\*\*           | Naive Bayes delivered modest results, with MCC values remaining relatively low. Tuning did not yield meaningful improvement, and its performance lagged behind Logistic Regression and Random Forest. Untuned MCC: \~0.68Tuned MCC: \~0.69 |

| \*\*Random Forest (Ensemble)\*\* | This model demonstrated strong performance overall, with its MCC improving slightly after tuning. While it remained competitive, it ranked just behind Logistic Regression in the tuned comparison.Untuned MCC: 0.7576Tuned MCC: 0.7598 |

| \*\*Overall Winner\*\*        | Based on my analysis and assignment execution, the Logistic Regression model is the best performing model overall. After hyperparameter tuning, it achieved the highest MCC of 0.7797. |

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
