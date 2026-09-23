from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ============================================================
# UBERPULSE AI - PRE-BOOKING OUTCOME PREDICTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "uberpulse_cleaned.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "ml"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = (
    MODEL_DIR / "prebooking_outcome_model.joblib"
)

METRICS_FILE = (
    REPORT_DIR / "prebooking_metrics.csv"
)

PREDICTIONS_FILE = (
    REPORT_DIR / "prebooking_predictions.csv"
)

FEATURE_IMPORTANCE_FILE = (
    REPORT_DIR / "prebooking_feature_importance.csv"
)

CONFUSION_MATRIX_FILE = (
    REPORT_DIR / "prebooking_confusion_matrix.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nCleaned dataset not found:\n{INPUT_FILE}\n\n"
            "Run first:\n"
            "python src/data_cleaning.py"
        )

    df = pd.read_csv(INPUT_FILE)

    print("\nDataset loaded:")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    # Restore numeric types after CSV loading.
    numeric_columns = [
        "Hour",
        "Day_of_Week",
        "Is_Weekend",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    df = df.copy()

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    df["Target_Completed"] = (
        df["Booking Status"] == "Completed"
    ).astype(int)

    # --------------------------------------------------------
    # PRE-BOOKING FEATURES ONLY
    # --------------------------------------------------------
    #
    # These are intended to represent information available
    # around the time a booking request is made.
    #
    # We intentionally exclude:
    #
    # Avg VTAT
    # Avg CTAT
    # Ride Distance
    # Driver Rating
    # Customer Rating
    # Revenue
    # Cancellation Reasons
    #
    # because these may become known only after the booking
    # process has progressed.
    # --------------------------------------------------------

    feature_columns = [
        "Vehicle Type",
        "Pickup Location",
        "Drop Location",
        "Hour",
        "Day_of_Week",
        "Is_Weekend",
        "Time_Period",
        "Payment Method",
    ]

    available_features = [
        column
        for column in feature_columns
        if column in df.columns
    ]

    X = df[
        available_features
    ].copy()

    y = df[
        "Target_Completed"
    ].copy()

    return (
        X,
        y,
        available_features,
    )


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor(X):

    categorical_features = [
        "Vehicle Type",
        "Pickup Location",
        "Drop Location",
        "Time_Period",
        "Payment Method",
    ]

    categorical_features = [
        column
        for column in categorical_features
        if column in X.columns
    ]

    numerical_features = [
        "Hour",
        "Day_of_Week",
        "Is_Weekend",
    ]

    numerical_features = [
        column
        for column in numerical_features
        if column in X.columns
    ]

    # --------------------------------------------------------
    # NUMERIC PIPELINE
    # --------------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    # --------------------------------------------------------
    # CATEGORICAL PIPELINE
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return (
        preprocessor,
        numerical_features,
        categorical_features,
    )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(X):

    (
        preprocessor,
        numerical_features,
        categorical_features,
    ) = create_preprocessor(X)

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )

    return (
        model,
        numerical_features,
        categorical_features,
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
):

    print(
        "\nTraining pre-booking Random Forest..."
    )

    (
        model,
        numerical_features,
        categorical_features,
    ) = create_model(
        X_train
    )

    print("\nNumeric features:")

    for feature in numerical_features:
        print(f"  - {feature}")

    print("\nCategorical features:")

    for feature in categorical_features:
        print(f"  - {feature}")

    model.fit(
        X_train,
        y_train,
    )

    print(
        "\nTraining completed."
    )

    return (
        model,
        numerical_features,
        categorical_features,
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    try:

        auc = roc_auc_score(
            y_test,
            probabilities,
        )

    except ValueError:

        auc = np.nan

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PRE-BOOKING MODEL PERFORMANCE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nAccuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {auc:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Not Completed",
                "Completed",
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = pd.DataFrame(
        [
            {
                "Metric": "Accuracy",
                "Score": accuracy,
            },
            {
                "Metric": "Precision",
                "Score": precision,
            },
            {
                "Metric": "Recall",
                "Score": recall,
            },
            {
                "Metric": "F1 Score",
                "Score": f1,
            },
            {
                "Metric": "ROC-AUC",
                "Score": auc,
            },
        ]
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=[
            "Actual_Not_Completed",
            "Actual_Completed",
        ],
        columns=[
            "Predicted_Not_Completed",
            "Predicted_Completed",
        ],
    )

    confusion_df.to_csv(
        CONFUSION_MATRIX_FILE
    )

    return (
        predictions,
        probabilities,
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def extract_feature_importance(
    model
):

    print(
        "\nGenerating feature importance..."
    )

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importance = (
        classifier
        .feature_importances_
    )

    feature_importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance,
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    feature_importance[
        "Importance"
    ] = (
        feature_importance[
            "Importance"
        ].round(6)
    )

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
    )

    print(
        "\nTop 20 Pre-Booking Predictive Features:"
    )

    print(
        feature_importance
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    df,
    test_indices,
    y_test,
    predictions,
    probabilities,
):

    result = df.iloc[
        test_indices
    ].copy()

    result[
        "Actual_Completed"
    ] = y_test.to_numpy()

    result[
        "Predicted_Completed"
    ] = predictions

    result[
        "Completion_Probability"
    ] = probabilities.round(4)

    result[
        "Prediction"
    ] = np.where(
        predictions == 1,
        "Completed",
        "Not Completed",
    )

    # Higher value = higher probability that
    # the booking will NOT complete.
    result[
        "Prebooking_Risk_Score"
    ] = (
        (1 - probabilities) * 100
    ).round(2)

    result = result.sort_values(
        "Prebooking_Risk_Score",
        ascending=False,
    )

    result.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        "\nPredictions saved to:"
    )

    print(
        PREDICTIONS_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "UBERPULSE AI"
    )

    print(
        "PRE-BOOKING OUTCOME PREDICTION"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    (
        X,
        y,
        feature_columns,
    ) = prepare_data(
        df
    )

    print(
        "\nPre-booking features used:"
    )

    for feature in feature_columns:
        print(
            f"  - {feature}"
        )

    print(
        "\nTarget distribution:"
    )

    print(
        y.value_counts()
        .rename(
            {
                0: "Not Completed",
                1: "Completed",
            }
        )
    )

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    indices = np.arange(
        len(df)
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_indices,
        test_indices,
    ) = train_test_split(
        X,
        y,
        indices,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(
        f"\nTraining rows : {len(X_train):,}"
    )

    print(
        f"Testing rows  : {len(X_test):,}"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    (
        model,
        numerical_features,
        categorical_features,
    ) = train_model(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    (
        predictions,
        probabilities,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    extract_feature_importance(
        model
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print(
        "\nModel saved to:"
    )

    print(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    save_predictions(
        df,
        test_indices,
        y_test,
        predictions,
        probabilities,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PRE-BOOKING AI MODEL COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"Model       : {MODEL_FILE}"
    )

    print(
        f"Metrics     : {METRICS_FILE}"
    )

    print(
        f"Predictions : {PREDICTIONS_FILE}"
    )

    print(
        f"Importance  : {FEATURE_IMPORTANCE_FILE}"
    )

    print(
        f"Confusion   : {CONFUSION_MATRIX_FILE}"
    )

    print(
        "\nML phase is now complete."
    )

    print(
        "Next step: Business Insights + Dashboard."
    )


if __name__ == "__main__":
    main()