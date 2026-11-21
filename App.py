from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =============================================================================
# 1. APP CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Credit Wise | Loan Approval",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1450px;
            padding-top: 2.0rem;
            padding-bottom: 3.0rem;
        }
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 85% 0%, rgba(37, 99, 235, 0.10), transparent 28%),
                        var(--background-color);
        }
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.18);
        }
        [data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.075);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 14px;
            padding: 16px 18px;
        }
        [data-testid="stMetricLabel"] p {
            font-size: 0.88rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.75rem;
            font-weight: 700;
        }
        div[data-testid="stForm"] {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 16px;
            padding: 1.25rem;
            background: rgba(128, 128, 128, 0.025);
        }
        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 10px;
            min-height: 2.8rem;
            font-weight: 650;
        }
        div[data-testid="stExpander"] {
            border-radius: 12px;
        }
        .cw-eyebrow {
            color: #60a5fa;
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .cw-muted {
            color: var(--text-color);
            opacity: 0.72;
            font-size: 0.92rem;
        }
        .cw-footer {
            text-align: center;
            opacity: 0.60;
            font-size: 0.82rem;
            padding-top: 0.6rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 2. CONSTANTS AND MODEL LOADING
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "Models" / "loan_approval_model.pkl"

MODEL_METRICS = {
    "Accuracy": "95.79%",
    "Precision": "89.39%",
    "Recall": "98.33%",
    "F1 Score": "93.65%",
    "ROC-AUC": "98.21%",
}

FEATURE_ORDER = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Age",
    "Dependents",
    "Credit_Score",
    "Existing_Loans",
    "DTI_Ratio",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category",
    "Education_Level",
]


@st.cache_resource(show_spinner="Loading the trained Random Forest pipeline...")
def load_model(model_path: str):
    """Load and cache the saved preprocessing + classifier pipeline."""
    return joblib.load(model_path)


if not MODEL_PATH.exists():
    st.error("The trained model file could not be found.")
    st.write("Expected location:")
    st.code(str(MODEL_PATH))
    st.info(
        "Keep `loan_approval_model.pkl` inside the `Models` folder, "
        "next to this App.py file's project directory."
    )
    st.stop()

try:
    loan_model = load_model(str(MODEL_PATH))
    if not hasattr(loan_model, "predict"):
        raise TypeError("The saved object does not provide a predict() method.")
    if not hasattr(loan_model, "predict_proba"):
        raise TypeError("The saved object does not provide predict_proba().")
    if not hasattr(loan_model, "classes_") or 1 not in list(loan_model.classes_):
        raise ValueError("The model does not contain the expected Approved class (1).")
except Exception as exc:
    st.error("The model file was found, but it could not be loaded correctly.")
    st.caption(
        "Check that the installed scikit-learn and joblib versions are compatible "
        "with the versions used when the model was saved."
    )
    with st.expander("Technical error details"):
        st.exception(exc)
    st.stop()


# =============================================================================
# 3. SHARED UI HELPERS
# =============================================================================
def render_page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(f'<div class="cw-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="cw-muted">{description}</div>', unsafe_allow_html=True)
    st.write("")


def render_performance_metrics() -> None:
    row_one = st.columns(3)
    row_one[0].metric("Accuracy", MODEL_METRICS["Accuracy"])
    row_one[1].metric("Precision", MODEL_METRICS["Precision"])
    row_one[2].metric("Recall", MODEL_METRICS["Recall"])

    row_two = st.columns(2)
    row_two[0].metric("F1 Score", MODEL_METRICS["F1 Score"])
    row_two[1].metric("ROC-AUC", MODEL_METRICS["ROC-AUC"])


# =============================================================================
# 4. SIDEBAR NAVIGATION
# =============================================================================
st.sidebar.markdown("# 🏦 Credit Wise")
st.sidebar.caption("LOAN APPROVAL PREDICTION SYSTEM")
st.sidebar.divider()

page = st.sidebar.radio(
    "WORKSPACE",
    ["Overview", "Loan Prediction", "Model Insights", "About"],
    key="navigation_page",
)

st.sidebar.divider()
st.sidebar.markdown("**Model status**")
st.sidebar.success("Random Forest pipeline loaded")
st.sidebar.caption("Preprocessing + classifier")
st.sidebar.caption("For educational use only")


# =============================================================================
# 5. OVERVIEW PAGE
# =============================================================================
if page == "Overview":
    render_page_header(
        "CREDIT RISK PROJECT",
        "Credit Wise Dashboard",
        "A machine learning application for exploring loan approval predictions.",
    )

    st.info(
        "Enter applicant information in **Loan Prediction** to generate a result "
        "using the saved preprocessing and Random Forest pipeline."
    )

    st.subheader("Model performance")
    render_performance_metrics()
    st.caption("Metrics are from the project's 190-record hold-out test set.")

    st.divider()
    overview_col, workflow_col = st.columns([1.25, 1], gap="large")

    with overview_col:
        st.subheader("Project overview")
        st.write(
            "Credit Wise is a binary classification project that predicts whether "
            "a loan applicant is classified as **Approved** or **Not Approved** "
            "using financial, employment and demographic features."
        )
        st.write(
            "The model artifact contains the preprocessing transformations and "
            "the tuned Random Forest classifier in one scikit-learn pipeline."
        )

    with workflow_col:
        st.subheader("How it works")
        st.markdown(
            "1. Enter the applicant's 18 input features.\n"
            "2. Submit the application form.\n"
            "3. The saved pipeline handles preprocessing.\n"
            "4. The classifier returns a predicted class and model score."
        )

    st.divider()
    st.warning(
        "**Important:** This is an educational demonstration. The displayed "
        "approval score is uncalibrated and is not a guaranteed probability. "
        "Do not use this application as the sole basis for a real lending decision."
    )


# =============================================================================
# 6. LOAN PREDICTION PAGE
# =============================================================================
elif page == "Loan Prediction":
    render_page_header(
        "APPLICANT WORKSPACE",
        "Loan Approval Prediction",
        "Provide the applicant details below. All 18 features are passed to the trained pipeline.",
    )

    st.caption(
        "Input limits and default values are based on the observed ranges and "
        "medians in the project dataset. They are not official lending limits. "
        "Enter values using the same units as the training data."
    )

    with st.form("loan_application_form", clear_on_submit=False):
        personal_col, financial_col = st.columns(2, gap="large")

        with personal_col:
            st.markdown("### 👤 Personal & employment")
            employment = st.selectbox(
                "Employment status",
                ["Salaried", "Self-employed", "Contract", "Unemployed"],
            )
            marital = st.selectbox("Marital status", ["Married", "Single"])
            education = st.selectbox(
                "Education level", ["Not Graduate", "Graduate"]
            )
            gender = st.selectbox("Gender", ["Female", "Male"])
            employer = st.selectbox(
                "Employer category",
                ["Private", "Government", "Unemployed", "MNC", "Business"],
            )
            applicant_income = st.number_input(
                "Applicant income",
                min_value=2009,
                max_value=19988,
                value=10453,
                step=1,
                help="Observed dataset range: 2,009 to 19,988.",
            )
            coapplicant_income = st.number_input(
                "Coapplicant income",
                min_value=1,
                max_value=9996,
                value=5143,
                step=1,
                help="Observed dataset range: 1 to 9,996.",
            )
            age = st.number_input(
                "Age", min_value=21, max_value=59, value=40, step=1
            )
            dependents = st.number_input(
                "Number of dependents", min_value=0, max_value=3, value=1, step=1
            )

        with financial_col:
            st.markdown("### 💳 Financial & loan details")
            credit_score = st.number_input(
                "Credit score", min_value=550, max_value=799, value=677, step=1
            )
            existing_loans = st.number_input(
                "Existing loans", min_value=0, max_value=4, value=2, step=1
            )
            dti_ratio = st.number_input(
                "Debt-to-income ratio",
                min_value=0.10,
                max_value=0.60,
                value=0.34,
                step=0.01,
                format="%.2f",
                help="Enter the ratio as a decimal, for example 0.34.",
            )
            savings = st.number_input(
                "Savings", min_value=73, max_value=19996, value=9910, step=1
            )
            collateral_value = st.number_input(
                "Collateral value", min_value=36, max_value=49954, value=24059, step=1
            )
            loan_amount = st.number_input(
                "Loan amount", min_value=1015, max_value=39995, value=21282, step=1
            )
            loan_term = st.number_input(
                "Loan term", min_value=12, max_value=84, value=48, step=12,
                help="The input step is 12, matching the form's current term increments.",
            )
            loan_purpose = st.selectbox(
                "Loan purpose", ["Personal", "Car", "Business", "Home", "Education"]
            )
            property_area = st.selectbox(
                "Property area", ["Urban", "Semiurban", "Rural"]
            )

        st.divider()
        submitted = st.form_submit_button(
            "🔍 Predict loan approval", type="primary", use_container_width=True
        )

    if submitted:
        applicant_data = {
            "Applicant_Income": applicant_income,
            "Coapplicant_Income": coapplicant_income,
            "Age": age,
            "Dependents": dependents,
            "Credit_Score": credit_score,
            "Existing_Loans": existing_loans,
            "DTI_Ratio": dti_ratio,
            "Savings": savings,
            "Collateral_Value": collateral_value,
            "Loan_Amount": loan_amount,
            "Loan_Term": loan_term,
            "Employment_Status": employment,
            "Marital_Status": marital,
            "Loan_Purpose": loan_purpose,
            "Property_Area": property_area,
            "Gender": gender,
            "Employer_Category": employer,
            "Education_Level": education,
        }
        input_df = pd.DataFrame([applicant_data], columns=FEATURE_ORDER)

        st.session_state.pop("credit_wise_last_result", None)
        try:
            prediction = int(loan_model.predict(input_df)[0])
            class_probabilities = loan_model.predict_proba(input_df)[0]
            class_labels = list(loan_model.classes_)
            approved_score = float(class_probabilities[class_labels.index(1)])
            not_approved_score = float(class_probabilities[class_labels.index(0)])

            st.session_state["credit_wise_last_result"] = {
                "prediction": prediction,
                "approved_score": approved_score,
                "not_approved_score": not_approved_score,
                "input_df": input_df,
            }
        except Exception as exc:
            st.error("The model could not generate a prediction for these inputs.")
            with st.expander("Technical error details"):
                st.exception(exc)

    result = st.session_state.get("credit_wise_last_result")

    if result is not None:
        st.divider()
        st.markdown('<div class="cw-eyebrow">MODEL OUTPUT</div>', unsafe_allow_html=True)
        st.subheader("Prediction result")

        result_col, scores_col = st.columns([1.1, 1], gap="large")
        with result_col:
            if result["prediction"] == 1:
                st.success("### ✅ Loan Approved")
                st.write("The model classified this applicant as **Approved**.")
            else:
                st.error("### ❌ Loan Not Approved")
                st.write("The model classified this applicant as **Not Approved**.")

        with scores_col:
            approved_col, rejected_col = st.columns(2)
            approved_col.metric("Approved class score", f"{result['approved_score']:.2%}")
            rejected_col.metric("Not Approved class score", f"{result['not_approved_score']:.2%}")
            st.progress(result["approved_score"])

        st.caption(
            "These are uncalibrated class scores returned by the model, not "
            "calibrated probabilities or guaranteed lending outcomes."
        )

        with st.expander("Review submitted applicant details"):
            st.dataframe(result["input_df"], use_container_width=True, hide_index=True)

        if st.button("Clear displayed prediction", type="secondary"):
            st.session_state.pop("credit_wise_last_result", None)
            st.rerun()


# =============================================================================
# 7. MODEL INSIGHTS PAGE
# =============================================================================
elif page == "Model Insights":
    render_page_header(
        "MODEL TRANSPARENCY",
        "Model Insights",
        "Review the tuned model's recorded test performance and feature importance.",
    )

    st.subheader("Hold-out test performance")
    render_performance_metrics()
    st.caption("Recorded evaluation on 190 test records.")

    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **Accuracy:** Overall proportion of correct predictions.\n"
            "- **Precision:** Of the applicants predicted as Approved, the proportion that were actually Approved.\n"
            "- **Recall:** Of the actually Approved applicants, the proportion identified by the model.\n"
            "- **F1 Score:** Harmonic mean of precision and recall.\n"
            "- **ROC-AUC:** Measures how well the model ranks positive cases above negative cases across thresholds."
        )

    st.divider()
    st.subheader("Confusion matrix")
    confusion_matrix_df = pd.DataFrame(
        [[123, 7], [1, 59]],
        index=["Actual: Not Approved", "Actual: Approved"],
        columns=["Predicted: Not Approved", "Predicted: Approved"],
    )
    st.dataframe(confusion_matrix_df, use_container_width=True)

    tn_col, fp_col, fn_col, tp_col = st.columns(4)
    tn_col.metric("True negatives", "123")
    fp_col.metric("False positives", "7")
    fn_col.metric("False negatives", "1")
    tp_col.metric("True positives", "59")

    st.caption(
        "Rows represent actual classes and columns represent predicted classes. "
        "Class 1 (Approved) is treated as the positive class."
    )

    st.divider()
    st.subheader("Random Forest feature importance")

    try:
        preprocessor = loan_model.named_steps["preprocessor"]
        classifier = loan_model.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_

        if len(feature_names) != len(importances):
            raise ValueError("Feature names and importance values do not match.")

        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        )
        importance_df["Feature"] = importance_df["Feature"].str.replace(
            r"^(num|ohe|ordinal)__", "", regex=True
        )
        importance_df = importance_df.sort_values(
            "Importance", ascending=False
        ).reset_index(drop=True)

        top_features = importance_df.head(10).sort_values("Importance")
        st.bar_chart(top_features, x="Importance", y="Feature", horizontal=True)

        with st.expander("View all feature importance values"):
            st.dataframe(importance_df, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.warning("Feature importance could not be extracted from the model artifact.")
        with st.expander("Technical details"):
            st.exception(exc)

    st.info(
        "Feature importance describes how the fitted Random Forest used features "
        "for its splits. It does not establish causation and is not an explanation "
        "for an individual applicant's result."
    )


# =============================================================================
# 8. ABOUT PAGE
# =============================================================================
elif page == "About":
    render_page_header(
        "PROJECT INFORMATION",
        "About Credit Wise",
        "An end-to-end machine learning project built with Python and Streamlit.",
    )

    st.subheader("Project summary")
    st.write(
        "Credit Wise demonstrates a binary classification workflow: data "
        "preparation and model evaluation in a notebook, saving the fitted "
        "preprocessing and prediction pipeline, and serving predictions through "
        "an interactive Streamlit application."
    )

    st.divider()
    stack_col, pipeline_col = st.columns(2, gap="large")

    with stack_col:
        st.subheader("Technology stack")
        st.markdown(
            "- **Python** — application and model code\n"
            "- **Pandas** — tabular input data\n"
            "- **Scikit-learn** — preprocessing and machine learning\n"
            "- **Random Forest** — final tuned classifier\n"
            "- **Joblib** — loading the saved pipeline\n"
            "- **Streamlit** — interactive web application"
        )

    with pipeline_col:
        st.subheader("Saved model pipeline")
        st.markdown(
            "- Numerical features: median imputation\n"
            "- Nominal categorical features: most-frequent imputation and one-hot encoding\n"
            "- Education level: most-frequent imputation and ordinal encoding\n"
            "- Final estimator: tuned Random Forest classifier"
        )

    st.divider()
    st.subheader("Limitations and responsible use")
    st.markdown(
        "- The project dataset is relatively small (950 records).\n"
        "- Performance should be validated on larger, independent and representative data.\n"
        "- The displayed class scores have not been calibrated.\n"
        "- Feature importance is not causal explanation.\n"
        "- The fairness checks in the project are preliminary and do not prove that the model is free from bias."
    )

    st.warning(
        "This application is for learning and demonstration only. It is not a "
        "validated credit decision system and must not be used as the sole basis "
        "for approving or rejecting a real loan application."
    )


# =============================================================================
# 9. FOOTER
# =============================================================================
st.divider()
st.markdown(
    '<div class="cw-footer">Credit Wise · Machine Learning Portfolio Project</div>',
    unsafe_allow_html=True,
)
