from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "ncr_ride_bookings.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "audit"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def save_report(dataframe, filename):
    path = REPORT_DIR / filename
    dataframe.to_csv(path)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------
if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{RAW_FILE}\n\n"
        "Put ncr_ride_bookings.csv inside data/raw/."
    )

df = pd.read_csv(RAW_FILE)

section("UBERPULSE AI — DATA AUDIT")
print(f"File: {RAW_FILE}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ---------------------------------------------------------------------
# Basic inspection
# ---------------------------------------------------------------------
section("1. DATASET PREVIEW")
print(df.head().to_string(index=False))

section("2. DATASET INFO")
df.info()

section("3. COLUMN NAMES")
for i, column in enumerate(df.columns, 1):
    print(f"{i:02d}. {column}")


# ---------------------------------------------------------------------
# Normalize text for the audit
# Removes surrounding spaces and quotes from string columns.
# This does NOT remove records.
# ---------------------------------------------------------------------
string_columns = df.select_dtypes(include=["object", "string"]).columns

for column in string_columns:
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.strip('"')
        .str.strip("'")
    )


# ---------------------------------------------------------------------
# Missing-value audit
# ---------------------------------------------------------------------
section("4. MISSING VALUES")

missing = pd.DataFrame({
    "Missing Count": df.isna().sum(),
    "Missing %": (df.isna().mean() * 100).round(2),
})

missing = missing.sort_values("Missing %", ascending=False)

print(missing.to_string())
save_report(missing, "missing_values.csv")


# ---------------------------------------------------------------------
# Duplicate audit
# ---------------------------------------------------------------------
section("5. DUPLICATES")

duplicate_rows = int(df.duplicated().sum())
print(f"Duplicate complete rows: {duplicate_rows:,}")

duplicate_booking_ids = None

if "Booking ID" in df.columns:
    duplicate_booking_ids = int(df["Booking ID"].duplicated().sum())
    print(f"Duplicate Booking IDs: {duplicate_booking_ids:,}")

duplicate_summary = pd.DataFrame({
    "Metric": [
        "Duplicate complete rows",
        "Duplicate Booking IDs",
    ],
    "Count": [
        duplicate_rows,
        duplicate_booking_ids if duplicate_booking_ids is not None else np.nan,
    ],
})

save_report(duplicate_summary, "duplicate_summary.csv")


# ---------------------------------------------------------------------
# Booking status
# ---------------------------------------------------------------------
section("6. BOOKING STATUS")

if "Booking Status" in df.columns:
    status_counts = df["Booking Status"].value_counts(dropna=False)
    status_percent = (
        df["Booking Status"]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
        .round(2)
    )

    status_summary = pd.DataFrame({
        "Count": status_counts,
        "Percentage": status_percent,
    })

    print(status_summary.to_string())
    save_report(status_summary, "booking_status_summary.csv")


# ---------------------------------------------------------------------
# Categorical audit
# ---------------------------------------------------------------------
section("7. CATEGORICAL COLUMNS")

categorical_columns = df.select_dtypes(
    include=["object", "string", "category"]
).columns

categorical_summary = []

for column in categorical_columns:
    unique_count = df[column].nunique(dropna=True)
    missing_count = int(df[column].isna().sum())

    categorical_summary.append({
        "Column": column,
        "Unique Values": unique_count,
        "Missing": missing_count,
    })

    print(f"\n--- {column} ---")
    print(f"Unique values: {unique_count:,}")
    print(df[column].value_counts(dropna=False).head(20).to_string())

categorical_summary = pd.DataFrame(categorical_summary)
save_report(categorical_summary, "categorical_summary.csv")


# ---------------------------------------------------------------------
# Numeric audit
# ---------------------------------------------------------------------
section("8. NUMERICAL SUMMARY")

numeric_columns = df.select_dtypes(include=np.number).columns
numeric_summary = df[numeric_columns].describe().T

print(numeric_summary.to_string())
save_report(numeric_summary, "numeric_summary.csv")


# ---------------------------------------------------------------------
# Negative and zero-value checks
# ---------------------------------------------------------------------
section("9. NUMERIC DATA QUALITY")

quality_rows = []

for column in numeric_columns:
    negative = int((df[column] < 0).sum())
    zero = int((df[column] == 0).sum())
    missing_count = int(df[column].isna().sum())

    quality_rows.append({
        "Column": column,
        "Negative Count": negative,
        "Zero Count": zero,
        "Missing Count": missing_count,
    })

    print(
        f"{column:<35} "
        f"negative={negative:>7,} | "
        f"zero={zero:>7,} | "
        f"missing={missing_count:>7,}"
    )

quality_summary = pd.DataFrame(quality_rows)
save_report(quality_summary, "numeric_quality_summary.csv")


# ---------------------------------------------------------------------
# Date/time audit
# ---------------------------------------------------------------------
section("10. DATETIME AUDIT")

if {"Date", "Time"}.issubset(df.columns):
    df["DateTime"] = pd.to_datetime(
        df["Date"].astype("string") + " " + df["Time"].astype("string"),
        errors="coerce",
    )

    invalid_datetime = int(df["DateTime"].isna().sum())

    print(f"Invalid DateTime values: {invalid_datetime:,}")

    if df["DateTime"].notna().any():
        print(f"Minimum DateTime: {df['DateTime'].min()}")
        print(f"Maximum DateTime: {df['DateTime'].max()}")

    print("\nDateTime preview:")
    print(
        df[["Date", "Time", "DateTime"]]
        .head(10)
        .to_string(index=False)
    )
else:
    invalid_datetime = None
    print("Date or Time column not found.")


# ---------------------------------------------------------------------
# Rating audit
# ---------------------------------------------------------------------
section("11. RATING VALIDATION")

rating_columns = [
    column for column in df.columns
    if "rating" in column.lower()
]

rating_rows = []

for column in rating_columns:
    invalid = int(
        ((df[column] < 0) | (df[column] > 5)).sum()
    )

    rating_rows.append({
        "Column": column,
        "Min": df[column].min(),
        "Max": df[column].max(),
        "Outside 0-5": invalid,
        "Missing": int(df[column].isna().sum()),
    })

    print(
        f"{column}: "
        f"min={df[column].min()}, "
        f"max={df[column].max()}, "
        f"outside 0-5={invalid:,}"
    )

if rating_rows:
    save_report(pd.DataFrame(rating_rows), "rating_validation.csv")


# ---------------------------------------------------------------------
# Identifier audit
# ---------------------------------------------------------------------
section("12. IDENTIFIER AUDIT")

identifier_rows = []

for column in ["Booking ID", "Customer ID"]:
    if column in df.columns:
        row = {
            "Column": column,
            "Missing": int(df[column].isna().sum()),
            "Unique": int(df[column].nunique(dropna=True)),
            "Duplicate Values": int(df[column].duplicated().sum()),
        }

        identifier_rows.append(row)

        print(
            f"{column}: "
            f"missing={row['Missing']:,}, "
            f"unique={row['Unique']:,}, "
            f"duplicates={row['Duplicate Values']:,}"
        )

if identifier_rows:
    save_report(pd.DataFrame(identifier_rows), "identifier_audit.csv")


# ---------------------------------------------------------------------
# Missingness by booking status
# This is important because missing values are often status-dependent.
# ---------------------------------------------------------------------
section("13. MISSINGNESS BY BOOKING STATUS")

if "Booking Status" in df.columns:

    important_columns = [
        column for column in [
            "Avg VTAT",
            "Avg CTAT",
            "Cancelled Rides by Customer",
            "Reason for cancelling by Customer",
            "Cancelled Rides by Driver",
            "Driver Cancellation Reason",
            "Incomplete Rides",
            "Incomplete Rides Reason",
            "Booking Value",
            "Ride Distance",
            "Driver Ratings",
            "Customer Rating",
            "Payment Method",
        ]
        if column in df.columns
    ]

    if important_columns:
        status_missing = (
            df.groupby("Booking Status")[important_columns]
            .apply(lambda group: group.isna().mean() * 100)
            .round(2)
        )

        print(status_missing.to_string())

        status_missing.to_csv(
            REPORT_DIR / "missingness_by_booking_status.csv"
        )

        print(
            f"Saved: {REPORT_DIR / 'missingness_by_booking_status.csv'}"
        )


# ---------------------------------------------------------------------
# Business-feature preview
# These are NOT the final cleaned data.
# ---------------------------------------------------------------------
section("14. ANALYTICAL FEATURE PREVIEW")

if "DateTime" in df.columns:
    df["Year"] = df["DateTime"].dt.year
    df["Month"] = df["DateTime"].dt.month
    df["Month Name"] = df["DateTime"].dt.month_name()
    df["Day"] = df["DateTime"].dt.day
    df["Day Name"] = df["DateTime"].dt.day_name()
    df["Hour"] = df["DateTime"].dt.hour
    df["Day of Week"] = df["DateTime"].dt.dayofweek
    df["Is Weekend"] = (df["Day of Week"] >= 5).astype(int)

if "Booking Status" in df.columns:
    df["Is Completed"] = (
        df["Booking Status"].eq("Completed")
    ).astype(int)

    df["Is Cancelled"] = (
        df["Booking Status"]
        .astype("string")
        .str.contains("cancelled", case=False, na=False)
    ).astype(int)

if {"Booking Status", "Booking Value"}.issubset(df.columns):
    df["Revenue"] = np.where(
        df["Booking Status"].eq("Completed"),
        df["Booking Value"],
        0,
    )

if {"Booking Value", "Ride Distance"}.issubset(df.columns):
    df["Fare Per KM"] = np.where(
        (df["Ride Distance"] > 0)
        & df["Booking Value"].notna(),
        df["Booking Value"] / df["Ride Distance"],
        np.nan,
    )

preview_columns = [
    column for column in [
        "Booking ID",
        "Booking Status",
        "DateTime",
        "Hour",
        "Day Name",
        "Is Weekend",
        "Booking Value",
        "Ride Distance",
        "Revenue",
        "Fare Per KM",
    ]
    if column in df.columns
]

print(df[preview_columns].head(10).to_string(index=False))


# ---------------------------------------------------------------------
# Overall audit report
# ---------------------------------------------------------------------
section("15. AUDIT SUMMARY")

audit_summary = pd.DataFrame([{
    "Rows": len(df),
    "Columns": len(df.columns),
    "Duplicate Rows": duplicate_rows,
    "Duplicate Booking IDs": (
        duplicate_booking_ids
        if duplicate_booking_ids is not None
        else np.nan
    ),
    "Total Missing Cells": int(df.isna().sum().sum()),
    "Invalid DateTime": (
        invalid_datetime
        if invalid_datetime is not None
        else np.nan
    ),
}])

print(audit_summary.to_string(index=False))
save_report(audit_summary, "audit_overview.csv")

section("AUDIT COMPLETE")

print(
    f"All audit reports have been saved to:\n{REPORT_DIR}\n\n"
    "Important: this audit does NOT delete rows and does NOT fill missing values."
)
