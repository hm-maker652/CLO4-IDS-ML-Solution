import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="IDS Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
    color: #111827;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #1e293b 100%);
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

[data-testid="stSidebar"] label {
    color: #ffffff !important;
}

[data-testid="stSidebar"] .stRadio label {
    color: #ffffff !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #111827 !important;
}

p, span, label {
    color: #111827;
}

.page-title {
    font-size: 36px;
    font-weight: 800;
    color: #111827 !important;
    margin-bottom: 2px;
}

.page-subtitle {
    color: #64748b !important;
    font-size: 15px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 23px;
    font-weight: 750;
    color: #111827 !important;
    margin-top: 30px;
    margin-bottom: 15px;
}

.card {
    background-color: #ffffff;
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.06);
    min-height: 120px;
}

.card-label {
    color: #64748b !important;
    font-size: 14px;
    font-weight: 600;
}

.card-value {
    color: #111827 !important;
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}

.card-small {
    color: #94a3b8 !important;
    font-size: 12px;
    margin-top: 5px;
}

.info-card {
    background-color: #ffffff;
    border-radius: 16px;
    padding: 22px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
    color: #111827 !important;
}

.info-card h3 {
    color: #111827 !important;
}

.info-card p {
    color: #475569 !important;
}

.success-box {
    background-color: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 15px;
    padding: 20px;
    color: #065f46 !important;
}

.success-box h3,
.success-box p {
    color: #065f46 !important;
}

.warning-box {
    background-color: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 15px;
    padding: 20px;
    color: #92400e !important;
}

.warning-box b {
    color: #92400e !important;
}

.normal-result {
    background-color: #ecfdf5;
    border: 2px solid #86efac;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    color: #166534 !important;
    font-size: 30px;
    font-weight: 800;
}

.malicious-result {
    background-color: #fef2f2;
    border: 2px solid #fca5a5;
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    color: #991b1b !important;
    font-size: 30px;
    font-weight: 800;
}

div.stButton > button {
    border-radius: 10px;
    height: 45px;
    font-weight: 700;
}

div.stButton > button p {
    color: inherit !important;
}

div[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
}

div[data-testid="stMetric"] label {
    color: #64748b !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #111827 !important;
}

.stSelectbox label,
.stTextInput label,
.stNumberInput label,
.stFileUploader label,
.stSlider label,
.stRadio label {
    color: #111827 !important;
}

.stTextInput input,
.stNumberInput input {
    background-color: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
}

[data-testid="stExpander"] {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
}

[data-testid="stExpander"] summary {
    color: #111827 !important;
}

[data-testid="stDataFrame"] {
    background-color: #ffffff;
}

.stAlert {
    color: #111827;
}

</style>
""", unsafe_allow_html=True)


MODEL_FILE = "random_forest_model.pkl"
FEATURE_FILE = "feature_names.pkl"
ROC_FILE = "roc_data.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_features():
    return joblib.load(FEATURE_FILE)


rf_model = load_model()
feature_names = load_features()


TOTAL_RECORDS = 743697
NORMAL_RECORDS = 611377
MALICIOUS_RECORDS = 132320


metrics_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        0.930180,
        0.998064,
        0.998023
    ],
    "Precision": [
        0.727265,
        0.994783,
        0.995606
    ],
    "Recall": [
        0.972151,
        0.994332,
        0.993274
    ],
    "F1 Score": [
        0.832064,
        0.994557,
        0.994439
    ],
    "ROC-AUC": [
        0.983860,
        0.996609,
        0.999693
    ]
})


confusion_matrices = {
    "Logistic Regression": np.array([
        [112628, 9648],
        [737, 25727]
    ]),
    "Decision Tree": np.array([
        [122138, 138],
        [150, 26314]
    ]),
    "Random Forest": np.array([
        [122160, 116],
        [178, 26286]
    ])
}


def metric_card(title, value, description=""):
    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">{title}</div>
            <div class="card-value">{value}</div>
            <div class="card-small">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.sidebar.markdown(
    '<div class="sidebar-title">🛡️ IDS</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="sidebar-subtitle">Intelligent Intrusion Detection</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "🏠 Dashboard",
        "🔎 Data Explorer",
        "🧠 Model Training",
        "📊 Model Performance",
        "🎯 Prediction",
        "⭐ Feature Importance"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="
        background:#1e293b;
        padding:15px;
        border-radius:12px;
        text-align:center;
    ">
    <b>Final Model</b><br>
    <span style="color:#93c5fd !important;">Random Forest</span><br><br>
    <small style="color:#cbd5e1 !important;">78 Network Features</small>
    </div>
    """,
    unsafe_allow_html=True
)


if page == "🏠 Dashboard":

    st.markdown(
        '<div class="page-title">Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">Intrusion Detection System • Network Traffic Analytics</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Total Records",
            f"{TOTAL_RECORDS:,}",
            "Dataset size"
        )

    with c2:
        metric_card(
            "Normal Traffic",
            f"{NORMAL_RECORDS:,}",
            "82.20% of records"
        )

    with c3:
        metric_card(
            "Malicious Traffic",
            f"{MALICIOUS_RECORDS:,}",
            "17.80% of records"
        )

    with c4:
        metric_card(
            "Input Features",
            "78",
            "Network traffic features"
        )

    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="info-card">
            <h3>🧹 Data Processing</h3>
            <p>
            Network traffic data was cleaned, converted to numeric
            values and prepared for machine learning.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="info-card">
            <h3>🤖 Model Comparison</h3>
            <p>
            Logistic Regression, Decision Tree and Random Forest
            were evaluated using multiple performance metrics.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="info-card">
            <h3>🛡️ Final Detection</h3>
            <p>
            Random Forest is used for the final prediction of
            normal and malicious network traffic.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">Traffic Distribution</div>',
        unsafe_allow_html=True
    )

    distribution = pd.DataFrame({
        "Traffic Type": [
            "Normal",
            "Malicious"
        ],
        "Records": [
            NORMAL_RECORDS,
            MALICIOUS_RECORDS
        ]
    })

    st.bar_chart(
        distribution.set_index("Traffic Type")
    )

    st.markdown(
        '<div class="section-title">Best Model</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="success-box">
        <h3>🏆 Random Forest</h3>
        <p>
        Random Forest achieved <b>99.80% accuracy</b>,
        <b>99.56% precision</b>, <b>99.33% recall</b>,
        <b>99.44% F1-score</b> and <b>99.97% ROC-AUC</b>.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )


elif page == "🔎 Data Explorer":

    st.markdown(
        '<div class="page-title">Data Explorer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">Explore dataset structure, classes and input features</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Records",
            f"{TOTAL_RECORDS:,}",
            "After preprocessing"
        )

    with c2:
        metric_card(
            "Features",
            "78",
            "Model input features"
        )

    with c3:
        metric_card(
            "Classes",
            "2",
            "Normal / Malicious"
        )

    st.markdown(
        '<div class="section-title">Class Distribution</div>',
        unsafe_allow_html=True
    )

    class_df = pd.DataFrame({
        "Class": [
            "Normal",
            "Malicious"
        ],
        "Records": [
            NORMAL_RECORDS,
            MALICIOUS_RECORDS
        ],
        "Percentage": [
            NORMAL_RECORDS / TOTAL_RECORDS * 100,
            MALICIOUS_RECORDS / TOTAL_RECORDS * 100
        ]
    })

    st.dataframe(
        class_df.style.format({
            "Percentage": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        class_df.set_index("Class")["Records"]
    )

    st.markdown(
        '<div class="section-title">Feature Explorer</div>',
        unsafe_allow_html=True
    )

    search_feature = st.text_input(
        "Search feature",
        placeholder="Type a feature name..."
    )

    filtered_features = [
        feature
        for feature in feature_names
        if search_feature.lower() in feature.lower()
    ]

    feature_df = pd.DataFrame({
        "No.": range(1, len(filtered_features) + 1),
        "Feature": filtered_features
    })

    st.dataframe(
        feature_df,
        use_container_width=True,
        height=450,
        hide_index=True
    )


elif page == "🧠 Model Training":

    st.markdown(
        '<div class="page-title">Model Training</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">Machine learning models and training configuration</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Training Split",
            "80%",
            "Training data"
        )

    with c2:
        metric_card(
            "Testing Split",
            "20%",
            "Testing data"
        )

    with c3:
        metric_card(
            "Random State",
            "42",
            "Reproducibility"
        )

    with c4:
        metric_card(
            "RF Trees",
            "100",
            "Random Forest"
        )

    st.markdown(
        '<div class="section-title">Models Used</div>',
        unsafe_allow_html=True
    )

    training_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "Type": [
            "Linear Classification",
            "Tree Based",
            "Ensemble Learning"
        ],
        "Scaling": [
            "StandardScaler",
            "Not Required",
            "Not Required"
        ],
        "Status": [
            "Evaluated",
            "Evaluated",
            "Final Model"
        ]
    })

    st.dataframe(
        training_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">Training Pipeline</div>',
        unsafe_allow_html=True
    )

    steps = st.columns(5)

    pipeline = [
        ("1", "Dataset"),
        ("2", "Cleaning"),
        ("3", "Train/Test Split"),
        ("4", "Model Training"),
        ("5", "Evaluation")
    ]

    for col, (number, name) in zip(steps, pipeline):

        with col:

            st.markdown(
                f"""
                <div class="info-card" style="text-align:center;">
                <h2>{number}</h2>
                <b>{name}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-title">Final Model Status</div>',
        unsafe_allow_html=True
    )

    st.success(
        "Random Forest model is trained and loaded successfully."
    )


elif page == "📊 Model Performance":

    st.markdown(
        '<div class="page-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">Compare classification results and model performance history</div>',
        unsafe_allow_html=True
    )

    display_df = metrics_df.copy()

    for col in display_df.columns[1:]:
        display_df[col] = display_df[col] * 100

    st.dataframe(
        display_df.style.format({
            "Accuracy": "{:.2f}%",
            "Precision": "{:.2f}%",
            "Recall": "{:.2f}%",
            "F1 Score": "{:.2f}%",
            "ROC-AUC": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">Metric Comparison</div>',
        unsafe_allow_html=True
    )

    selected_metric = st.selectbox(
        "Select performance metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    )

    chart_data = (
        metrics_df
        .set_index("Model")[[selected_metric]]
        * 100
    )

    st.bar_chart(chart_data)

    st.markdown(
        '<div class="section-title">Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    selected_model = st.selectbox(
        "Select model",
        list(confusion_matrices.keys())
    )

    cm = confusion_matrices[selected_model]

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Normal",
            "Actual Malicious"
        ],
        columns=[
            "Predicted Normal",
            "Predicted Malicious"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    tn, fp, fn, tp = cm.ravel()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("True Negative", f"{tn:,}")

    with c2:
        metric_card("False Positive", f"{fp:,}")

    with c3:
        metric_card("False Negative", f"{fn:,}")

    with c4:
        metric_card("True Positive", f"{tp:,}")

    st.markdown(
        '<div class="section-title">ROC Curve</div>',
        unsafe_allow_html=True
    )

    try:

        roc_data = joblib.load(ROC_FILE)

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.plot(
            roc_data["fpr_lr"],
            roc_data["tpr_lr"],
            label="Logistic Regression (AUC = 0.9839)"
        )

        ax.plot(
            roc_data["fpr_dt"],
            roc_data["tpr_dt"],
            label="Decision Tree (AUC = 0.9966)"
        )

        ax.plot(
            roc_data["fpr_rf"],
            roc_data["tpr_rf"],
            label="Random Forest (AUC = 0.9997)"
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC-AUC Comparison")
        ax.legend()

        st.pyplot(fig)

    except Exception:

        st.warning(
            "roc_data.pkl was not found. Add the file to display the ROC curve."
        )

    st.markdown(
        '<div class="section-title">📜 Model Performance History</div>',
        unsafe_allow_html=True
    )

    history_df = pd.DataFrame({
        "Experiment": [
            "Experiment 1",
            "Experiment 2",
            "Experiment 3"
        ],
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "Accuracy": [
            93.02,
            99.81,
            99.80
        ],
        "Precision": [
            72.73,
            99.48,
            99.56
        ],
        "Recall": [
            97.22,
            99.43,
            99.33
        ],
        "F1 Score": [
            83.21,
            99.46,
            99.44
        ],
        "ROC-AUC": [
            98.39,
            99.66,
            99.97
        ],
        "Status": [
            "Baseline",
            "Evaluated",
            "Selected"
        ]
    })

    st.dataframe(
        history_df.style.format({
            "Accuracy": "{:.2f}%",
            "Precision": "{:.2f}%",
            "Recall": "{:.2f}%",
            "F1 Score": "{:.2f}%",
            "ROC-AUC": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        """
        <div class="info-card">
        <h3>📌 Performance Summary</h3>
        <p>
        Logistic Regression provides the baseline result.
        Decision Tree and Random Forest provide substantially
        stronger classification performance. Random Forest has
        the highest ROC-AUC and precision and is therefore used
        as the final prediction model.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )


elif page == "🎯 Prediction":

    st.markdown(
        '<div class="page-title">Traffic Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">Detect whether network traffic is Normal or Malicious</div>',
        unsafe_allow_html=True
    )

    prediction_mode = st.radio(
        "Prediction Method",
        [
            "📁 CSV Upload",
            "✍️ Manual Input"
        ],
        horizontal=True
    )

    if prediction_mode == "📁 CSV Upload":

        st.markdown(
            '<div class="section-title">Upload Network Traffic</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card">
            Upload a CSV file containing the same 78 features used
            during Random Forest training.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"]
        )

        if uploaded_file:

            data = pd.read_csv(uploaded_file)

            st.markdown(
                '<div class="section-title">Uploaded Data</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                data.head(),
                use_container_width=True
            )

            missing = [
                feature
                for feature in feature_names
                if feature not in data.columns
            ]

            if missing:

                st.error(
                    f"{len(missing)} required features are missing."
                )

                with st.expander("View Missing Features"):

                    st.write(missing)

            else:

                input_data = data[feature_names].copy()

                input_data = input_data.apply(
                    pd.to_numeric,
                    errors="coerce"
                )

                if input_data.isnull().any().any():

                    st.error(
                        "The uploaded data contains missing or non-numeric values."
                    )

                else:

                    predictions = rf_model.predict(input_data)

                    probabilities = rf_model.predict_proba(
                        input_data
                    )

                    result = data.copy()

                    result["Prediction"] = predictions

                    result["Prediction Label"] = np.where(
                        predictions == 0,
                        "Normal",
                        "Malicious"
                    )

                    result["Normal Probability (%)"] = (
                        probabilities[:, 0] * 100
                    ).round(2)

                    result["Malicious Probability (%)"] = (
                        probabilities[:, 1] * 100
                    ).round(2)

                    normal_count = int(
                        (predictions == 0).sum()
                    )

                    malicious_count = int(
                        (predictions == 1).sum()
                    )

                    c1, c2 = st.columns(2)

                    with c1:

                        metric_card(
                            "Normal Traffic",
                            normal_count,
                            "Predicted records"
                        )

                    with c2:

                        metric_card(
                            "Malicious Traffic",
                            malicious_count,
                            "Predicted records"
                        )

                    st.markdown(
                        '<div class="section-title">Prediction Results</div>',
                        unsafe_allow_html=True
                    )

                    st.dataframe(
                        result,
                        use_container_width=True
                    )

                    csv_result = result.to_csv(
                        index=False
                    )

                    st.download_button(
                        "⬇️ Download Prediction Results",
                        csv_result,
                        "ids_prediction_results.csv",
                        "text/csv",
                        use_container_width=True
                    )

    else:

        st.markdown(
            '<div class="section-title">Manual Network Traffic Input</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="warning-box">
            <b>Important:</b> The Random Forest model requires all
            78 features. The fields below are grouped to make manual
            input easier. Default values are set to 0.
            </div>
            """,
            unsafe_allow_html=True
        )

        groups = {
            "Flow Information": feature_names[0:6],

            "Forward Packet Features": feature_names[6:10],

            "Backward Packet Features": feature_names[10:14],

            "Flow Statistics": feature_names[14:20],

            "Forward IAT": feature_names[20:25],

            "Backward IAT": feature_names[25:30],

            "TCP Flags": feature_names[30:34],

            "Header Information": feature_names[34:38],

            "Packet Length Features": feature_names[38:43],

            "Flag Counts": feature_names[43:51],

            "Packet Statistics": feature_names[51:55],

            "Bulk Features": feature_names[55:62],

            "Subflow Features": feature_names[62:66],

            "TCP Window Features": feature_names[66:68],

            "Packet Activity": feature_names[68:70],

            "Active Features": feature_names[70:74],

            "Idle Features": feature_names[74:78]
        }

        manual_data = {}

        for group_name, group_features in groups.items():

            with st.expander(group_name):

                cols = st.columns(2)

                for i, feature in enumerate(group_features):

                    with cols[i % 2]:

                        manual_data[feature] = st.number_input(
                            feature,
                            value=0.0,
                            key=f"input_{feature}"
                        )

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:

            predict_button = st.button(
                "🔍 Predict Traffic",
                type="primary",
                use_container_width=True
            )

        with c2:

            reset_button = st.button(
                "↻ Reset All Values",
                use_container_width=True
            )

        if reset_button:

            st.rerun()

        if predict_button:

            input_df = pd.DataFrame(
                [manual_data],
                columns=feature_names
            )

            input_df = input_df.apply(
                pd.to_numeric,
                errors="coerce"
            )

            prediction = rf_model.predict(
                input_df
            )[0]

            probability = rf_model.predict_proba(
                input_df
            )[0]

            normal_probability = probability[0] * 100

            malicious_probability = probability[1] * 100

            st.markdown(
                '<div class="section-title">Detection Result</div>',
                unsafe_allow_html=True
            )

            if prediction == 0:

                st.markdown(
                    '<div class="normal-result">🟢 NORMAL TRAFFIC</div>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<div class="malicious-result">🔴 MALICIOUS TRAFFIC</div>',
                    unsafe_allow_html=True
                )

            st.write("")

            c1, c2 = st.columns(2)

            with c1:

                metric_card(
                    "Normal Probability",
                    f"{normal_probability:.2f}%"
                )

            with c2:

                metric_card(
                    "Malicious Probability",
                    f"{malicious_probability:.2f}%"
                )


elif page == "⭐ Feature Importance":

    st.markdown(
        '<div class="page-title">Feature Importance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">Features contributing to Random Forest classification</div>',
        unsafe_allow_html=True
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        metric_card(
            "Total Features",
            len(feature_names)
        )

    with c2:

        metric_card(
            "Top Feature",
            importance_df.iloc[0]["Feature"]
        )

    with c3:

        metric_card(
            "Top Importance",
            f"{importance_df.iloc[0]['Importance']:.4f}"
        )

    st.markdown(
        '<div class="section-title">Top Important Features</div>',
        unsafe_allow_html=True
    )

    top_n = st.slider(
        "Number of features to display",
        min_value=5,
        max_value=30,
        value=15
    )

    top_features = importance_df.head(
        top_n
    )

    st.bar_chart(
        top_features.set_index("Feature")["Importance"]
    )

    st.markdown(
        '<div class="section-title">Feature Ranking</div>',
        unsafe_allow_html=True
    )

    ranking = top_features.reset_index(
        drop=True
    )

    ranking.insert(
        0,
        "Rank",
        range(1, len(ranking) + 1)
    )

    st.dataframe(
        ranking,
        use_container_width=True,
        hide_index=True
    )