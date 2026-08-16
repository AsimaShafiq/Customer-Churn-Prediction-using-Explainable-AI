import streamlit as st
import pandas as pd
import joblib
import shap


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# Load Saved Components
# --------------------------------------------------

@st.cache_resource
def load_artifacts():

    preprocessor = joblib.load("Preprocessor.pkl")
    model = joblib.load("Model.pkl")

    return preprocessor, model


preprocessor, model = load_artifacts()


# --------------------------------------------------
# SHAP Explanation Function
# --------------------------------------------------

def get_shap_explanation(
    model,
    processed_data,
    feature_names
):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        processed_data
    )


    # Handle different SHAP output formats
    if isinstance(shap_values, list):

        # Binary classification:
        # Select SHAP values for churn class
        shap_values = shap_values[1][0]

    else:

        # Some SHAP versions return:
        # (samples, features, classes)
        if len(shap_values.shape) == 3:

            shap_values = shap_values[0, :, 1]

        # Standard 2D output
        elif len(shap_values.shape) == 2:

            shap_values = shap_values[0]


    explanation_df = pd.DataFrame({
        "Transformed Feature": feature_names,
        "SHAP Value": shap_values
    })

    return explanation_df


# --------------------------------------------------
# Convert Transformed Feature Names
# Back to Readable Original Feature Names
# --------------------------------------------------

def get_original_feature_name(
    transformed_feature,
    original_features
):

    # Remove transformer prefix
    feature_name = transformed_feature.split(
        "__",
        1
    )[-1]


    # Match encoded feature back to original feature
    for feature in sorted(
        original_features,
        key=len,
        reverse=True
    ):

        if (
            feature_name == feature
            or feature_name.startswith(
                feature + "_"
            )
        ):

            return feature


    return feature_name


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("About the Project")

    st.write(
        """
        This application predicts whether a customer
        is likely to churn based on customer,
        service, contract, and billing information.
        """
    )

    st.markdown("---")

    st.caption(
        "Machine Learning Customer Churn Prediction System"
    )


# --------------------------------------------------
# Application Header
# --------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer information below to predict the likelihood of churn."
)

st.markdown("---")


# --------------------------------------------------
# Customer Input Form
# --------------------------------------------------

with st.form("customer_form"):

    # --------------------------------------------------
    # Personal Information
    # --------------------------------------------------

    st.subheader("👤 Personal Information")

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )


    with col2:

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )


    with col3:

        partner = st.selectbox(
            "Partner",
            ["No", "Yes"]
        )


    with col4:

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"]
        )


    # --------------------------------------------------
    # Service Information
    # --------------------------------------------------

    st.subheader("📡 Service Information")

    col1, col2, col3 = st.columns(3)


    with col1:

        phone_service = st.selectbox(
            "Phone Service",
            ["No", "Yes"]
        )


    with col2:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )


    with col3:

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )


    col1, col2, col3 = st.columns(3)


    with col1:

        online_security = st.selectbox(
            "Online Security",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )


        device_protection = st.selectbox(
            "Device Protection",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )


    with col2:

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )


        tech_support = st.selectbox(
            "Tech Support",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )


    with col3:

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )


        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )


    # --------------------------------------------------
    # Contract and Payment Information
    # --------------------------------------------------

    st.subheader(
        "📄 Contract and Payment Information"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )


    with col2:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["No", "Yes"]
        )


    with col3:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check"
            ]
        )


    # --------------------------------------------------
    # Numerical Information
    # --------------------------------------------------

    st.subheader("💰 Customer Information")

    col1, col2 = st.columns(2)


    with col1:

        tenure_months = st.number_input(
            "Tenure Months",
            min_value=0,
            step=1
        )


        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            step=0.01
        )


    with col2:

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            step=0.01
        )


        cltv = st.number_input(
            "CLTV",
            min_value=0.0,
            step=1.0
        )


    # --------------------------------------------------
    # Submit Button
    # --------------------------------------------------

    submitted = st.form_submit_button(
        "Predict Churn"
    )


# ==================================================
# Prediction Process
# ==================================================

if submitted:


    # --------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------

    additional_services = [

        online_security,
        online_backup,
        device_protection,
        tech_support

    ]


    total_additional_services = sum(

        service == "Yes"

        for service in additional_services

    )


    streaming_services = [

        streaming_tv,
        streaming_movies

    ]


    total_streaming_services = sum(

        service == "Yes"

        for service in streaming_services

    )


    security_services = [

        online_security,
        online_backup,
        device_protection

    ]


    total_security_services = sum(

        service == "Yes"

        for service in security_services

    )


    has_multiple_services = int(

        total_additional_services >= 2

    )


    # --------------------------------------------------
    # Tenure Group
    # --------------------------------------------------

    if tenure_months <= 12:

        tenure_group = "New"


    elif tenure_months <= 24:

        tenure_group = "Early"


    elif tenure_months <= 48:

        tenure_group = "Established"


    else:

        tenure_group = "Long-term"


    # --------------------------------------------------
    # Create Customer Data
    # --------------------------------------------------

    customer_data = pd.DataFrame({

        "Tenure Months": [
            tenure_months
        ],

        "Monthly Charges": [
            monthly_charges
        ],

        "Total Charges": [
            total_charges
        ],

        "CLTV": [
            cltv
        ],


        "Total Aditional Services": [

            total_additional_services

        ],


        "Total Streaming Services": [

            total_streaming_services

        ],


        "Total Security Services": [

            total_security_services

        ],


        "Has Multiple Services": [

            has_multiple_services

        ],


        "Gender": [
            gender
        ],

        "Senior Citizen": [
            senior_citizen
        ],

        "Partner": [
            partner
        ],

        "Dependents": [
            dependents
        ],


        "Phone Service": [
            phone_service
        ],

        "Multiple Lines": [
            multiple_lines
        ],

        "Internet Service": [
            internet_service
        ],


        "Online Security": [
            online_security
        ],

        "Online Backup": [
            online_backup
        ],

        "Device Protection": [
            device_protection
        ],

        "Tech Support": [
            tech_support
        ],


        "Streaming TV": [
            streaming_tv
        ],

        "Streaming Movies": [
            streaming_movies
        ],


        "Contract": [
            contract
        ],

        "Paperless Billing": [
            paperless_billing
        ],

        "Payment Method": [
            payment_method
        ],


        "Tenure Group": [
            tenure_group
        ]

    })


    # --------------------------------------------------
    # Align Feature Order
    # --------------------------------------------------

    customer_data = customer_data[
        preprocessor.feature_names_in_
    ]


    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    processed_data = preprocessor.transform(
        customer_data
    )


    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    prediction = model.predict(
        processed_data
    )[0]


    probability = model.predict_proba(
        processed_data
    )[0, 1]


    # --------------------------------------------------
    # Display Results
    # --------------------------------------------------

    st.markdown("---")

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        if prediction == 1:

            st.error(
                "⚠️ Likely to Churn"
            )


        else:

            st.success(
                "✅ Likely to Stay"
            )


    with result_col2:

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )


    # --------------------------------------------------
    # Risk Interpretation
    # --------------------------------------------------

    if probability >= 0.70:

        st.warning(
            "High churn risk detected."
        )


    elif probability >= 0.40:

        st.info(
            "Moderate churn risk detected."
        )


    else:

        st.success(
            "Low churn risk detected."
        )


    # ==================================================
    # WHY THIS PREDICTION?
    # ==================================================

    st.markdown("---")

    st.subheader(
        "🔍 Why This Prediction?"
    )

    st.write(
        "The factors below show which customer characteristics "
        "had the strongest influence on the churn prediction."
    )


    try:


        # --------------------------------------------------
        # Get Feature Names After Preprocessing
        # --------------------------------------------------

        transformed_feature_names = (
            preprocessor.get_feature_names_out()
        )


        # --------------------------------------------------
        # Generate SHAP Explanation
        # --------------------------------------------------

        explanation_df = get_shap_explanation(

            model,
            processed_data,
            transformed_feature_names

        )


        # --------------------------------------------------
        # Map Encoded Features to Original Features
        # --------------------------------------------------

        original_features = list(
            customer_data.columns
        )


        explanation_df["Feature"] = (

            explanation_df[
                "Transformed Feature"
            ].apply(

                lambda feature:

                get_original_feature_name(

                    feature,
                    original_features

                )

            )

        )


        # --------------------------------------------------
        # Combine Contributions of Same Original Feature
        # --------------------------------------------------

        explanation_df = (

            explanation_df

            .groupby(
                "Feature",
                as_index=False
            )

            .agg({

                "SHAP Value": "sum"

            })

        )


        # --------------------------------------------------
        # Add Customer's Actual Values
        # --------------------------------------------------

        customer_values = (
            customer_data
            .iloc[0]
            .to_dict()
        )


        explanation_df[
            "Customer Value"
        ] = (

            explanation_df[
                "Feature"
            ].map(
                customer_values
            )

        )


        # --------------------------------------------------
        # Factors Increasing Churn Risk
        #
        # Positive SHAP values push prediction
        # toward churn
        # --------------------------------------------------

        increasing_risk = (

            explanation_df[

                explanation_df[
                    "SHAP Value"
                ] > 0

            ]

            .sort_values(

                "SHAP Value",

                ascending=False

            )

            .head(5)

        )


        # --------------------------------------------------
        # Factors Reducing Churn Risk
        #
        # Negative SHAP values push prediction
        # away from churn
        # --------------------------------------------------

        reducing_risk = (

            explanation_df[

                explanation_df[
                    "SHAP Value"
                ] < 0

            ]

            .sort_values(

                "SHAP Value",

                ascending=True

            )

            .head(5)

        )


        # --------------------------------------------------
        # Display Factors
        # --------------------------------------------------

        col1, col2 = st.columns(2)


        # --------------------------------------------------
        # Increasing Churn Risk
        # --------------------------------------------------

        with col1:


            st.markdown(
                "### 🔴 Factors Increasing Churn Risk"
            )


            if not increasing_risk.empty:


                for _, row in (
                    increasing_risk.iterrows()
                ):


                    st.markdown(

                        f"""
                        **⬆️ {row["Feature"]}**

                        Customer value: `{row["Customer Value"]}`
                        """

                    )


            else:


                st.info(
                    "No major factors were identified "
                    "as increasing churn risk."
                )


        # --------------------------------------------------
        # Reducing Churn Risk
        # --------------------------------------------------

        with col2:


            st.markdown(
                "### 🟢 Factors Reducing Churn Risk"
            )


            if not reducing_risk.empty:


                for _, row in (
                    reducing_risk.iterrows()
                ):


                    st.markdown(

                        f"""
                        **⬇️ {row["Feature"]}**

                        Customer value: `{row["Customer Value"]}`
                        """

                    )


            else:


                st.info(
                    "No major factors were identified "
                    "as reducing churn risk."
                )


        # --------------------------------------------------
        # Prediction Summary
        # --------------------------------------------------

        st.markdown("---")

        st.subheader(
            "📌 Prediction Summary"
        )


        if prediction == 1:


            st.write(

                f"""
                The model predicts that this customer is
                **likely to churn** with a churn probability
                of **{probability:.2%}**.

                The factors above show which customer
                characteristics contributed most toward
                increasing or reducing churn risk.
                """

            )


        else:


            st.write(

                f"""
                The model predicts that this customer is
                **likely to stay** with a churn probability
                of **{probability:.2%}**.

                The factors above show which customer
                characteristics had the strongest influence
                on this prediction.
                """

            )


    except Exception as e:


        st.warning(
            "The prediction was successful, but the "
            "explanation could not be generated."
        )


        st.caption(
            f"Explanation error: {e}"
        )
