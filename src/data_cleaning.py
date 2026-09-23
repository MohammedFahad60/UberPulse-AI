from pathlib import Path
import pandas as pd
import numpy as np
import re
import json

# ============================================================
# UBERPULSE AI - DATA CLEANING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "ncr_ride_bookings.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports" / "cleaning"

OUTPUT_FILE = OUTPUT_DIR / "uberpulse_cleaned.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

NUMERIC_COLUMNS = [
    "Avg VTAT",
    "Avg CTAT",
    "Cancelled Rides by Customer",
    "Cancelled Rides by Driver",
    "Incomplete Rides",
    "Booking Value",
    "Ride Distance",
    "Driver Ratings",
    "Customer Rating",
]

RATING_COLUMNS = [
    "Driver Ratings",
    "Customer Rating",
]

NON_NEGATIVE_COLUMNS = [
    "Avg VTAT",
    "Avg CTAT",
    "Booking Value",
    "Ride Distance",
]

INDICATOR_COLUMNS = [
    "Cancelled Rides by Customer",
    "Cancelled Rides by Driver",
    "Incomplete Rides",
]

CATEGORICAL_COLUMNS = [
    "Booking Status",
    "Vehicle Type",
    "Pickup Location",
    "Drop Location",
    "Reason for cancelling by Customer",
    "Driver Cancellation Reason",
    "Incomplete Rides Reason",
    "Payment Method",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_string(value):
    """Clean whitespace and unnecessary quotation marks."""
    if pd.isna(value):
        return value

    value = str(value).strip()
    value = value.strip('"').strip("'")
    value = re.sub(r"\s+", " ", value)

    return value


def clean_text_columns(df):
    """Clean all text/categorical columns."""
    for column in df.columns:
        if (
            df[column].dtype == "object"
            or pd.api.types.is_string_dtype(df[column])
        ):
            df[column] = df[column].apply(clean_string)

    return df


def convert_numeric_columns(df):
    """Convert numeric fields safely."""
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def validate_numeric_values(df):
    """Detect and null invalid numeric values."""
    validation_report = []

    # Non-negative fields
    for column in NON_NEGATIVE_COLUMNS:
        if column not in df.columns:
            continue

        invalid_mask = df[column].notna() & (df[column] < 0)
        invalid_count = int(invalid_mask.sum())

        if invalid_count > 0:
            df.loc[invalid_mask, column] = np.nan

        validation_report.append(
            {
                "column": column,
                "rule": "Must be >= 0",
                "invalid_values": invalid_count,
            }
        )

    # Ratings
    for column in RATING_COLUMNS:
        if column not in df.columns:
            continue

        invalid_mask = df[column].notna() & (
            (df[column] < 0) | (df[column] > 5)
        )

        invalid_count = int(invalid_mask.sum())

        if invalid_count > 0:
            df.loc[invalid_mask, column] = np.nan

        validation_report.append(
            {
                "column": column,
                "rule": "Must be between 0 and 5",
                "invalid_values": invalid_count,
            }
        )

    # Binary indicator fields
    for column in INDICATOR_COLUMNS:
        if column not in df.columns:
            continue

        invalid_mask = df[column].notna() & (
            ~df[column].isin([0, 1])
        )

        invalid_count = int(invalid_mask.sum())

        if invalid_count > 0:
            df.loc[invalid_mask, column] = np.nan

        validation_report.append(
            {
                "column": column,
                "rule": "Must be 0 or 1",
                "invalid_values": invalid_count,
            }
        )

    return df, pd.DataFrame(validation_report)


def create_datetime_features(df):
    """Create temporal features from Date and Time."""
    if "Date" not in df.columns or "Time" not in df.columns:
        raise ValueError("Date and Time columns are required.")

    df["Date"] = df["Date"].astype("string").str.strip()
    df["Time"] = df["Time"].astype("string").str.strip()

    df["DateTime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        errors="coerce"
    )

    invalid_datetime_count = int(df["DateTime"].isna().sum())

    df = df.dropna(subset=["DateTime"]).copy()

    df["Year"] = df["DateTime"].dt.year
    df["Month"] = df["DateTime"].dt.month
    df["Month_Name"] = df["DateTime"].dt.month_name()
    df["Day"] = df["DateTime"].dt.day
    df["Day_of_Week"] = df["DateTime"].dt.dayofweek
    df["Day_Name"] = df["DateTime"].dt.day_name()
    df["Hour"] = df["DateTime"].dt.hour
    df["Is_Weekend"] = df["Day_of_Week"].isin([5, 6]).astype(int)

    def time_period(hour):
        if 0 <= hour <= 5:
            return "Night"
        elif 6 <= hour <= 11:
            return "Morning"
        elif 12 <= hour <= 16:
            return "Afternoon"
        elif 17 <= hour <= 20:
            return "Evening"
        else:
            return "Late Evening"

    df["Time_Period"] = df["Hour"].apply(time_period)

    return df, invalid_datetime_count


def create_business_features(df):
    """Create analytical features without inventing business values."""

    status = df["Booking Status"].fillna("").astype(str).str.strip()

    df["Is_Completed"] = (
        status == "Completed"
    ).astype(int)

    df["Is_Cancelled"] = (
        status.str.contains("Cancelled", case=False, na=False)
    ).astype(int)

    df["Is_Incomplete"] = (
        status == "Incomplete"
    ).astype(int)

    df["Is_No_Driver_Found"] = (
        status == "No Driver Found"
    ).astype(int)

    df["Booking_Outcome"] = np.select(
        [
            status == "Completed",
            status == "Cancelled by Customer",
            status == "Cancelled by Driver",
            status == "No Driver Found",
            status == "Incomplete",
        ],
        [
            "Completed",
            "Customer Cancelled",
            "Driver Cancelled",
            "No Driver Found",
            "Incomplete",
        ],
        default="Other",
    )

    # Revenue is counted only for completed rides.
    df["Revenue"] = np.where(
        df["Is_Completed"] == 1,
        df["Booking Value"],
        0,
    )

    # Fare per kilometre is meaningful only for completed rides
    # with a positive ride distance.
    df["Fare_Per_KM"] = np.where(
        (df["Is_Completed"] == 1)
        & df["Ride Distance"].notna()
        & (df["Ride Distance"] > 0)
        & df["Booking Value"].notna(),
        df["Booking Value"] / df["Ride Distance"],
        np.nan,
    )

    # Data availability flags
    availability_columns = {
        "Booking Value": "Has_Booking_Value",
        "Ride Distance": "Has_Ride_Distance",
        "Driver Ratings": "Has_Driver_Rating",
        "Customer Rating": "Has_Customer_Rating",
        "Payment Method": "Has_Payment_Method",
        "Avg VTAT": "Has_VTAT",
        "Avg CTAT": "Has_CTAT",
    }

    for source_column, flag_column in availability_columns.items():
        if source_column in df.columns:
            df[flag_column] = df[source_column].notna().astype(int)

    return df


# ============================================================
# MAIN CLEANING PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("UBERPULSE AI - DATA CLEANING")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{INPUT_FILE}\n\n"
            "Place ncr_ride_bookings.csv inside:\n"
            "data/raw/"
        )

    print(f"\nInput file: {INPUT_FILE}")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    original_rows = len(df)
    original_columns = len(df.columns)

    print(f"Original rows    : {original_rows:,}")
    print(f"Original columns : {original_columns}")

    # --------------------------------------------------------
    # CLEAN TEXT
    # --------------------------------------------------------

    print("\n[1/8] Cleaning text columns...")

    df = clean_text_columns(df)

    # --------------------------------------------------------
    # REMOVE EXACT DUPLICATES
    # --------------------------------------------------------

    print("[2/8] Removing exact duplicate rows...")

    duplicate_rows = int(df.duplicated().sum())

    df = df.drop_duplicates().reset_index(drop=True)

    # --------------------------------------------------------
    # IDENTIFIER AUDIT
    # --------------------------------------------------------

    print("[3/8] Auditing booking identifiers...")

    duplicate_booking_ids = 0
    duplicate_booking_id_rows = 0

    if "Booking ID" in df.columns:

        duplicate_mask = df["Booking ID"].duplicated(keep=False)

        duplicate_booking_id_rows = int(duplicate_mask.sum())

        duplicate_booking_ids = int(
            df.loc[duplicate_mask, "Booking ID"].nunique()
        )

    # Do NOT automatically delete duplicate Booking IDs.
    # They are retained so potentially meaningful records
    # are not silently lost.

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    print("[4/8] Converting numeric columns...")

    df = convert_numeric_columns(df)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("[5/8] Validating numeric values...")

    df, validation_report = validate_numeric_values(df)

    validation_report.to_csv(
        REPORT_DIR / "numeric_validation_report.csv",
        index=False
    )

    # --------------------------------------------------------
    # DATETIME
    # --------------------------------------------------------

    print("[6/8] Creating datetime features...")

    df, invalid_datetime_count = create_datetime_features(df)

    # --------------------------------------------------------
    # BUSINESS FEATURES
    # --------------------------------------------------------

    print("[7/8] Creating business analytics features...")

    df = create_business_features(df)

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    if "DateTime" in df.columns:
        df = df.sort_values(
            by="DateTime"
        ).reset_index(drop=True)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print("[8/8] Saving cleaned dataset...")

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # REPORTS
    # ========================================================

    # Missing-value report
    missing_report = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing_Count": df.isna().sum().values,
            "Missing_Percentage": (
                df.isna().mean().values * 100
            ).round(2),
            "Data_Type": [
                str(dtype)
                for dtype in df.dtypes
            ],
        }
    )

    missing_report = missing_report.sort_values(
        by="Missing_Percentage",
        ascending=False
    )

    missing_report.to_csv(
        REPORT_DIR / "missing_values_after_cleaning.csv",
        index=False
    )

    # Booking status report
    if "Booking Status" in df.columns:
        status_report = (
            df["Booking Status"]
            .value_counts(dropna=False)
            .rename_axis("Booking Status")
            .reset_index(name="Count")
        )

        status_report["Percentage"] = (
            status_report["Count"]
            / len(df)
            * 100
        ).round(2)

        status_report.to_csv(
            REPORT_DIR / "booking_status_after_cleaning.csv",
            index=False
        )

    # Vehicle report
    if "Vehicle Type" in df.columns:
        vehicle_report = (
            df["Vehicle Type"]
            .value_counts(dropna=False)
            .rename_axis("Vehicle Type")
            .reset_index(name="Count")
        )

        vehicle_report["Percentage"] = (
            vehicle_report["Count"]
            / len(df)
            * 100
        ).round(2)

        vehicle_report.to_csv(
            REPORT_DIR / "vehicle_type_after_cleaning.csv",
            index=False
        )

    # Duplicate Booking ID report
    if "Booking ID" in df.columns:

        duplicate_id_report = (
            df[df["Booking ID"].duplicated(keep=False)]
            .sort_values("Booking ID")
        )

        duplicate_id_report.to_csv(
            REPORT_DIR / "duplicate_booking_ids.csv",
            index=False
        )

    # Feature/data dictionary
    data_dictionary = []

    for column in df.columns:

        data_dictionary.append(
            {
                "Column": column,
                "Data_Type": str(df[column].dtype),
                "Missing_Count": int(df[column].isna().sum()),
                "Missing_Percentage": round(
                    df[column].isna().mean() * 100,
                    2
                ),
                "Unique_Values": int(
                    df[column].nunique(dropna=True)
                ),
            }
        )

    pd.DataFrame(data_dictionary).to_csv(
        REPORT_DIR / "cleaned_data_dictionary.csv",
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "original_rows": original_rows,
        "final_rows": int(len(df)),
        "original_columns": original_columns,
        "final_columns": int(len(df.columns)),
        "exact_duplicate_rows_removed": duplicate_rows,
        "duplicate_booking_ids_found": duplicate_booking_ids,
        "rows_with_duplicate_booking_ids": duplicate_booking_id_rows,
        "invalid_datetime_rows_removed": invalid_datetime_count,
        "missing_values_remaining": int(df.isna().sum().sum()),
        "completed_bookings": int(
            df["Is_Completed"].sum()
        ),
        "cancelled_bookings": int(
            df["Is_Cancelled"].sum()
        ),
        "incomplete_bookings": int(
            df["Is_Incomplete"].sum()
        ),
        "no_driver_found_bookings": int(
            df["Is_No_Driver_Found"].sum()
        ),
        "total_revenue_completed_rides": round(
            float(df["Revenue"].sum()),
            2
        ),
    }

    with open(
        REPORT_DIR / "cleaning_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4
        )

    print("\n" + "=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)

    print(f"\nRows before cleaning : {original_rows:,}")
    print(f"Rows after cleaning  : {len(df):,}")
    print(f"Columns before       : {original_columns}")
    print(f"Columns after        : {len(df.columns)}")

    print(
        f"\nExact duplicates removed : {duplicate_rows:,}"
    )

    print(
        f"Invalid datetime rows    : {invalid_datetime_count:,}"
    )

    print(
        f"Duplicate Booking IDs    : {duplicate_booking_ids:,}"
    )

    print(
        f"Completed bookings       : "
        f"{df['Is_Completed'].sum():,}"
    )

    print(
        f"Cancelled bookings       : "
        f"{df['Is_Cancelled'].sum():,}"
    )

    print(
        f"Total completed revenue  : "
        f"{df['Revenue'].sum():,.2f}"
    )

    print(f"\nCleaned dataset saved to:")
    print(OUTPUT_FILE)

    print("\nReports saved to:")
    print(REPORT_DIR)

    print("\nNext step: EDA + KPI analysis.")


if __name__ == "__main__":
    main()