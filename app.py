from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt


st.set_page_config(
    page_title="Learning Style Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)


MODEL_PATH = Path("models/learning_style_model.pkl")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")
TARGET_COLUMN = "Learning Style"


@st.cache_resource
def load_artifacts() -> tuple[Any | None, Any | None, list[str]]:
    errors: list[str] = []
    model = None
    preprocessor = None

    if not MODEL_PATH.exists():
        errors.append(f"Missing model file: `{MODEL_PATH}`")
    if not PREPROCESSOR_PATH.exists():
        errors.append(f"Missing preprocessor file: `{PREPROCESSOR_PATH}`")

    if errors:
        return model, preprocessor, errors

    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Failed to load artifacts: {exc}")

    return model, preprocessor, errors


def read_uploaded_dataset(uploaded_file: Any) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix == ".xlsx":
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file format. Please upload a .csv or .xlsx file.")


def get_training_metadata(preprocessor: Any) -> dict[str, Any]:
    return {
        "feature_columns": list(getattr(preprocessor, "feature_columns_", [])),
        "numeric_features": list(getattr(preprocessor, "numeric_features_", [])),
        "categorical_features": list(getattr(preprocessor, "categorical_features_", [])),
        "target_column": getattr(preprocessor, "target_column_", TARGET_COLUMN),
        "class_names": list(getattr(preprocessor, "class_names_", [])),
        "train_shape": getattr(preprocessor, "train_shape_", None),
        "test_shape": getattr(preprocessor, "test_shape_", None),
        "random_state": getattr(preprocessor, "random_state_", None),
        "model_name": getattr(preprocessor, "model_name_", "RandomForestClassifier"),
    }


def make_dataset_overview(df: pd.DataFrame, metadata: dict[str, Any]) -> None:
    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", int(df.shape[0]))
    c2.metric("Columns", int(df.shape[1]))
    c3.metric("Missing Values", int(df.isna().sum().sum()))

    with st.expander("Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("Column Information", expanded=True):
        info_df = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": df.dtypes.astype(str).values,
                "missing_values": df.isna().sum().values,
                "unique_values": df.nunique(dropna=False).values,
            }
        )
        st.dataframe(info_df, use_container_width=True)

    target_column = metadata["target_column"]
    if target_column in df.columns:
        st.subheader("Class Distribution")
        class_counts = (
            df[target_column]
            .astype("string")
            .fillna("Missing")
            .value_counts()
            .rename_axis("Learning Style")
            .reset_index(name="Count")
        )
        col_a, col_b = st.columns([1, 1])
        col_a.dataframe(class_counts, use_container_width=True)
        fig = px.bar(
            class_counts,
            x="Learning Style",
            y="Count",
            color="Learning Style",
            title="Learning Style Distribution",
        )
        fig.update_layout(showlegend=False)
        col_b.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Class distribution unavailable because `{target_column}` is not present in the uploaded dataset.")


def get_feature_importance_df(model: Any, preprocessor: Any) -> pd.DataFrame:
    feature_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return importance_df


def show_model_information(model: Any, preprocessor: Any, metadata: dict[str, Any]) -> None:
    st.subheader("Model Information")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estimator", metadata["model_name"])
    c2.metric("Classes", len(metadata["class_names"]))
    c3.metric("Numeric Features", len(metadata["numeric_features"]))
    c4.metric("Categorical Features", len(metadata["categorical_features"]))

    with st.expander("Metadata", expanded=True):
        st.json(
            {
                "target_column": metadata["target_column"],
                "feature_columns": metadata["feature_columns"],
                "numeric_features": metadata["numeric_features"],
                "categorical_features": metadata["categorical_features"],
                "class_names": metadata["class_names"],
                "train_shape": metadata["train_shape"],
                "test_shape": metadata["test_shape"],
                "random_state": metadata["random_state"],
                "model_parameters": model.get_params(),
            }
        )

    importance_df = get_feature_importance_df(model, preprocessor).head(15)

    col_a, col_b = st.columns([1, 1])
    col_a.dataframe(importance_df, use_container_width=True)

    fig = px.bar(
        importance_df.sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        orientation="h",
        title="Top 15 Feature Importances",
    )
    col_b.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance Plot")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=importance_df.sort_values("importance", ascending=True),
        x="importance",
        y="feature",
        palette="crest",
    )
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()


def coerce_value(raw_value: Any, column_name: str, metadata: dict[str, Any]) -> Any:
    if column_name in metadata["numeric_features"]:
        if raw_value in ("", None):
            return np.nan
        return float(raw_value)
    return None if raw_value == "" else raw_value


def build_single_prediction_form(
    reference_df: pd.DataFrame | None,
    metadata: dict[str, Any],
    preprocessor: Any,
) -> pd.DataFrame:
    feature_columns = metadata["feature_columns"]
    numeric_features = set(metadata["numeric_features"])
    input_data: dict[str, Any] = {}

    st.subheader("Interactive Prediction")
    with st.form("single_prediction_form"):
        cols = st.columns(2)
        for idx, column in enumerate(feature_columns):
            current_col = cols[idx % 2]
            series = reference_df[column] if reference_df is not None and column in reference_df.columns else None

            if column in numeric_features:
                default_value = 0.0
                if series is not None:
                    numeric_series = pd.to_numeric(series, errors="coerce")
                    if numeric_series.notna().any():
                        default_value = float(numeric_series.median())
                value = current_col.number_input(column, value=float(default_value))
            else:
                if series is not None:
                    categories = (
                        series.astype("string").dropna().drop_duplicates().sort_values().tolist()
                    )
                else:
                    categories = []

                known_categories = []
                for transformer_name, transformer, columns in preprocessor.transformers_:  # type: ignore[name-defined]
                    if transformer_name == "cat":
                        for feature_name, values in zip(columns, transformer.named_steps["encoder"].categories_):
                            if feature_name == column:
                                known_categories = [str(item) for item in values.tolist()]
                                break
                        if known_categories:
                            break

                options = sorted(set(categories) | set(known_categories))
                if not options:
                    value = current_col.text_input(column)
                else:
                    value = current_col.selectbox(column, options=options)

            input_data[column] = value

        submitted = st.form_submit_button("Predict Learning Style", use_container_width=True)

    if not submitted:
        return pd.DataFrame()

    parsed = {}
    for column, value in input_data.items():
        parsed[column] = coerce_value(value, column, metadata)

    return pd.DataFrame([parsed], columns=feature_columns)


def validate_prediction_dataframe(df: pd.DataFrame, metadata: dict[str, Any]) -> tuple[bool, str]:
    if df.empty:
        return False, "Uploaded dataset is empty."

    missing_columns = [column for column in metadata["feature_columns"] if column not in df.columns]
    if missing_columns:
        return False, f"Column mismatch. Missing required columns: {missing_columns}"

    numeric_columns = metadata["numeric_features"]
    for column in numeric_columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.isna().all():
            return False, f"Invalid datatype detected in numeric column `{column}`."

    return True, ""


def show_prediction_results(model: Any, preprocessor: Any, prediction_df: pd.DataFrame) -> None:
    transformed = preprocessor.transform(prediction_df)
    predicted_class = model.predict(transformed)[0]
    probabilities = model.predict_proba(transformed)[0]

    st.success(f"Predicted Learning Style: **{predicted_class}**")

    probability_df = pd.DataFrame(
        {
            "Learning Style": model.classes_,
            "Probability": probabilities,
        }
    ).sort_values("Probability", ascending=False)

    col_a, col_b = st.columns([1, 1])
    col_a.dataframe(probability_df, use_container_width=True)
    fig = px.bar(
        probability_df,
        x="Learning Style",
        y="Probability",
        color="Learning Style",
        title="Prediction Probability Distribution",
    )
    fig.update_layout(showlegend=False, yaxis_range=[0, 1])
    col_b.plotly_chart(fig, use_container_width=True)


def show_batch_prediction_section(model: Any, preprocessor: Any, uploaded_df: pd.DataFrame, metadata: dict[str, Any]) -> None:
    st.subheader("Batch Predictions")
    is_valid, message = validate_prediction_dataframe(uploaded_df, metadata)
    if not is_valid:
        st.warning(message)
        return

    feature_df = uploaded_df[metadata["feature_columns"]].copy()
    for column in metadata["numeric_features"]:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce")

    transformed = preprocessor.transform(feature_df)
    predictions = model.predict(transformed)
    probabilities = model.predict_proba(transformed)

    result_df = uploaded_df.copy()
    result_df["Predicted Learning Style"] = predictions
    result_df["Prediction Confidence"] = probabilities.max(axis=1)

    with st.expander("Prediction Output", expanded=True):
        st.dataframe(result_df.head(25), use_container_width=True)

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Predictions CSV",
        data=csv_bytes,
        file_name="learning_style_predictions.csv",
        mime="text/csv",
    )


def show_schema_reference(metadata: dict[str, Any]) -> None:
    st.subheader("Expected Input Schema")
    schema_df = pd.DataFrame(
        {
            "feature": metadata["feature_columns"],
            "type": [
                "numeric" if col in metadata["numeric_features"] else "categorical"
                for col in metadata["feature_columns"]
            ],
        }
    )
    st.dataframe(schema_df, use_container_width=True)


model, preprocessor, artifact_errors = load_artifacts()

st.title("Learning Style Classification")
st.caption("Predict student learning styles using saved machine learning artifacts.")

if artifact_errors:
    st.error("Model artifacts are unavailable.")
    for error in artifact_errors:
        st.write(f"- {error}")
    st.info("Run `model_training.ipynb` first to generate the required Joblib files.")
    st.stop()

metadata = get_training_metadata(preprocessor)

if not metadata["feature_columns"]:
    st.error("Loaded artifacts are incompatible with this app because feature metadata is missing.")
    st.info("Retrain the model using `model_training.ipynb` to regenerate compatible artifacts.")
    st.stop()

st.sidebar.title("Workflow")
page = st.sidebar.radio(
    "Go to",
    [
        "Upload Data",
        "Dataset Overview",
        "Model Information",
        "Make Predictions",
    ],
)

uploaded_file = st.sidebar.file_uploader("Upload dataset", type=["csv", "xlsx"])
uploaded_df: pd.DataFrame | None = None

if uploaded_file is not None:
    try:
        uploaded_df = read_uploaded_dataset(uploaded_file)
        if uploaded_df.empty:
            st.sidebar.warning("The uploaded file is empty.")
            uploaded_df = None
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Unable to read file: {exc}")
        uploaded_df = None

if page == "Upload Data":
    st.subheader("Upload Data")
    if uploaded_df is None:
        st.info("Upload a `.csv` or `.xlsx` dataset from the sidebar to begin.")
        show_schema_reference(metadata)
    else:
        st.success("Dataset uploaded successfully.")
        make_dataset_overview(uploaded_df, metadata)

elif page == "Dataset Overview":
    if uploaded_df is None:
        st.warning("Upload a dataset to view its overview.")
        show_schema_reference(metadata)
    else:
        make_dataset_overview(uploaded_df, metadata)

elif page == "Model Information":
    show_model_information(model, preprocessor, metadata)

elif page == "Make Predictions":
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    with tab1:
        reference_df = uploaded_df if uploaded_df is not None else None
        input_df = build_single_prediction_form(reference_df, metadata, preprocessor)
        if not input_df.empty:
            try:
                show_prediction_results(model, preprocessor, input_df)
            except ValueError as exc:
                st.error(f"Prediction failed due to invalid datatype or missing values: {exc}")
    with tab2:
        if uploaded_df is None:
            st.info("Upload a dataset to generate batch predictions.")
        else:
            show_batch_prediction_section(model, preprocessor, uploaded_df, metadata)
