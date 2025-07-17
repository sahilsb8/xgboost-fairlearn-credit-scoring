# Fair Credit Risk Scoring Model (XGBoost + Logistic Regression + Fairlearn)

This project presents a machine learning-based **Credit Risk Scoring** system using **XGBoost** and **Logistic Regression**, with an integrated **Fairness-Aware** evaluation and mitigation layer using **Fairlearn**. The objective is to predict whether a loan applicant is likely to default, while ensuring that the model treats all demographic groups fairly especially with regard to sensitive attributes like gender.

## Problem Statement

Credit risk models are widely used in financial institutions to assess the likelihood of a borrower defaulting on a loan. However, these models can inadvertently encode and amplify biases present in the data, leading to **discriminatory outcomes** for certain groups.

This project explores:
- **Binary classification** for loan default prediction
- **Bias detection and mitigation** using fairness constraints
- Trade-off analysis between **accuracy** and **fairness**

## Technologies & Tools

| Component              | Description                                       |
|------------------------|---------------------------------------------------|
| Python                 | Core programming language                         |
| XGBoost                | Gradient boosting classifier                      |
| Logistic Regression    | Baseline linear classifier                        |
| Fairlearn              | Fairness-aware evaluation and mitigation          |
| Scikit-learn           | Preprocessing, metrics, pipelines                 |
| Pandas, NumPy          | Data manipulation and analysis                    |
| Matplotlib / Seaborn   | Visualization                                     |
| Jupyter Notebook       | Interactive development and experimentation       |

## Dataset

We used a publicly available **credit dataset**, containing demographic and financial features such as:

- Age
- Income
- Loan Amount
- Employment Status
- Credit History
- Gender *(used as a protected attribute)*

### Target Variable:
- `Loan Default`: 1 if defaulted, 0 otherwise

## Notes
- This code is provided "as-is" and may require further optimization or modification based on the specific usecases and requirements.
- Make sure to refer to the documentation of the libraries used in code above to learn more about them and the reason for their utilization.
