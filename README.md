# Credit Wise Loan Approval Prediction

## Project Overview

Credit Wise Loan Approval Prediction is a Machine Learning classification project that predicts loan approval decisions based on applicant financial and demographic information.

The project uses multiple classification algorithms and selects a tuned Random Forest model based on evaluation performance.

## Objectives

- Analyze applicant and loan-related data.
- Perform data cleaning and preprocessing.
- Train and compare classification algorithms.
- Optimize model performance using RandomizedSearchCV.
- Evaluate model performance and conduct an initial fairness audit.
- Generate predictions for new applicants.

## Technologies Used

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Jupyter Notebook
- Joblib

## Machine Learning Models

- Logistic Regression
- K-Nearest Neighbors
- Naive Bayes
- Decision Tree
- Random Forest

## Final Model Performance

**Selected Model: Tuned Random Forest**

| Metric | Score |
|---|---:|
| Accuracy | 95.79% |
| Precision | 89.39% |
| Recall | 98.33% |
| F1-Score | 93.65% |
| ROC-AUC | 98.21% |

## Key Findings

- DTI Ratio was the most influential feature.
- Credit Score was the second most influential feature.
- Applicant Income and Loan Amount also contributed to predictions.
- Initial gender-wise evaluation showed relatively similar performance across male and female groups.

## Project Structure

- `notebooks/`: Data analysis and model development.
- `models/`: Saved trained model.
- `images/`: Project visualizations.
- `requirements.txt`: Required Python libraries.

## Limitations

- Model evaluation was conducted on a limited dataset.
- Further validation on larger independent datasets is required.
- Feature importance does not establish causality.
- Initial fairness testing does not prove the absence of bias.
- Predicted probabilities are not guaranteed real-world approval probabilities.

## Future Improvements

- SHAP-based explainability.
- Probability calibration.
- Advanced fairness evaluation.
- Streamlit or FastAPI deployment.

## Disclaimer

This project is developed for educational and portfolio purposes. It should not be used as the sole basis for real-world lending decisions.