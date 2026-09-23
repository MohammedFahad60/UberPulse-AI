import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "uberpulse_cleaned.csv"
EDA_DIR = BASE_DIR / "reports" / "eda"
ML_DIR = BASE_DIR / "reports" / "ml"
OUTPUT_DIR = BASE_DIR / "reports" / "business_insights"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def clean_number(value):
    try:
        return round(float(value), 2)
    except Exception:
        return None


def safe_percentage(numerator, denominator):
    if denominator == 0:
        return 0
    return round((numerator / denominator) * 100, 2)


def top_value(df, column):
    if df.empty or column not in df.columns:
        return None
    return df[column].value_counts().index[0]


def load_csv(filename):
    path = EDA_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("UBERPULSE AI - BUSINESS INSIGHTS")
print("=" * 70)

print("\nLoading cleaned dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Rows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns)}")


# ============================================================
# BASIC NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "Booking Value",
    "Ride Distance",
    "Driver Ratings",
    "Customer Rating",
    "Avg VTAT",
    "Avg CTAT",
    "Hour",
    "Is_Weekend",
    "Is_Completed",
    "Is_Cancelled",
    "Is_Incomplete",
    "Is_No_Driver_Found",
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")


# ============================================================
# EXECUTIVE KPIs
# ============================================================

total_bookings = len(df)

completed = int(df["Is_Completed"].sum())
cancelled = int(df["Is_Cancelled"].sum())
incomplete = int(df["Is_Incomplete"].sum())
no_driver = int(df["Is_No_Driver_Found"].sum())

completion_rate = safe_percentage(completed, total_bookings)
cancellation_rate = safe_percentage(cancelled, total_bookings)
incomplete_rate = safe_percentage(incomplete, total_bookings)
no_driver_rate = safe_percentage(no_driver, total_bookings)

completed_df = df[df["Is_Completed"] == 1].copy()

total_revenue = (
    completed_df["Booking Value"].sum()
    if "Booking Value" in completed_df.columns
    else 0
)

avg_booking_value = (
    completed_df["Booking Value"].mean()
    if not completed_df.empty
    else 0
)

avg_distance = (
    completed_df["Ride Distance"].mean()
    if not completed_df.empty
    else 0
)

avg_driver_rating = (
    completed_df["Driver Ratings"].mean()
    if not completed_df.empty
    else 0
)

avg_customer_rating = (
    completed_df["Customer Rating"].mean()
    if not completed_df.empty
    else 0
)


# ============================================================
# VEHICLE INSIGHTS
# ============================================================

vehicle_insights = {}

if "Vehicle Type" in df.columns:

    vehicle_summary = (
        df.groupby("Vehicle Type")
        .agg(
            bookings=("Booking ID", "count"),
            completed=("Is_Completed", "sum"),
            cancelled=("Is_Cancelled", "sum"),
            revenue=("Booking Value", "sum"),
        )
        .reset_index()
    )

    vehicle_summary["completion_rate"] = (
        vehicle_summary["completed"]
        / vehicle_summary["bookings"]
        * 100
    )

    vehicle_summary["cancellation_rate"] = (
        vehicle_summary["cancelled"]
        / vehicle_summary["bookings"]
        * 100
    )

    vehicle_summary = vehicle_summary.sort_values(
        "bookings",
        ascending=False
    )

    vehicle_insights = {
        "highest_demand_vehicle": vehicle_summary.iloc[0]["Vehicle Type"],
        "highest_revenue_vehicle": vehicle_summary.sort_values(
            "revenue",
            ascending=False
        ).iloc[0]["Vehicle Type"],
        "highest_completion_vehicle": vehicle_summary.sort_values(
            "completion_rate",
            ascending=False
        ).iloc[0]["Vehicle Type"],
        "vehicle_summary": vehicle_summary.to_dict(orient="records"),
    }


# ============================================================
# HOURLY DEMAND INSIGHTS
# ============================================================

hourly_insights = {}

if "Hour" in df.columns:

    hourly = (
        df.groupby("Hour")
        .agg(
            bookings=("Booking ID", "count"),
            completed=("Is_Completed", "sum"),
            cancelled=("Is_Cancelled", "sum"),
            revenue=("Booking Value", "sum"),
        )
        .reset_index()
    )

    hourly["completion_rate"] = (
        hourly["completed"]
        / hourly["bookings"]
        * 100
    )

    hourly["cancellation_rate"] = (
        hourly["cancelled"]
        / hourly["bookings"]
        * 100
    )

    peak_hour = hourly.sort_values(
        "bookings",
        ascending=False
    ).iloc[0]

    highest_revenue_hour = hourly.sort_values(
        "revenue",
        ascending=False
    ).iloc[0]

    hourly_insights = {
        "peak_hour": int(peak_hour["Hour"]),
        "peak_hour_bookings": int(peak_hour["bookings"]),
        "highest_revenue_hour": int(highest_revenue_hour["Hour"]),
        "highest_hourly_revenue": clean_number(
            highest_revenue_hour["revenue"]
        ),
        "hourly_summary": hourly.to_dict(orient="records"),
    }


# ============================================================
# TIME PERIOD INSIGHTS
# ============================================================

time_period_insights = {}

if "Time_Period" in df.columns:

    period = (
        df.groupby("Time_Period")
        .agg(
            bookings=("Booking ID", "count"),
            completed=("Is_Completed", "sum"),
            cancelled=("Is_Cancelled", "sum"),
            revenue=("Booking Value", "sum"),
        )
        .reset_index()
    )

    period["completion_rate"] = (
        period["completed"]
        / period["bookings"]
        * 100
    )

    period["cancellation_rate"] = (
        period["cancelled"]
        / period["bookings"]
        * 100
    )

    peak_period = period.sort_values(
        "bookings",
        ascending=False
    ).iloc[0]

    time_period_insights = {
        "highest_demand_period": peak_period["Time_Period"],
        "period_summary": period.to_dict(orient="records"),
    }


# ============================================================
# LOCATION INSIGHTS
# ============================================================

location_insights = {}

if "Pickup Location" in df.columns:

    pickup = (
        df.groupby("Pickup Location")
        .agg(
            bookings=("Booking ID", "count"),
            completed=("Is_Completed", "sum"),
            cancelled=("Is_Cancelled", "sum"),
            revenue=("Booking Value", "sum"),
        )
        .reset_index()
    )

    pickup["completion_rate"] = (
        pickup["completed"]
        / pickup["bookings"]
        * 100
    )

    pickup["cancellation_rate"] = (
        pickup["cancelled"]
        / pickup["bookings"]
        * 100
    )

    top_pickup = pickup.sort_values(
        "bookings",
        ascending=False
    ).iloc[0]

    location_insights = {
        "highest_demand_pickup": top_pickup["Pickup Location"],
        "highest_demand_pickup_bookings": int(top_pickup["bookings"]),
        "top_pickup_locations": pickup.sort_values(
            "bookings",
            ascending=False
        ).head(10).to_dict(orient="records"),
    }


# ============================================================
# ROUTE INSIGHTS
# ============================================================

route_insights = {}

if "Pickup Location" in df.columns and "Drop Location" in df.columns:

    route_df = df.copy()

    route_df["Route"] = (
        route_df["Pickup Location"].astype(str)
        + " → "
        + route_df["Drop Location"].astype(str)
    )

    routes = (
        route_df.groupby("Route")
        .agg(
            bookings=("Booking ID", "count"),
            completed=("Is_Completed", "sum"),
            revenue=("Booking Value", "sum"),
        )
        .reset_index()
    )

    routes["completion_rate"] = (
        routes["completed"]
        / routes["bookings"]
        * 100
    )

    top_route = routes.sort_values(
        "bookings",
        ascending=False
    ).iloc[0]

    route_insights = {
        "highest_volume_route": top_route["Route"],
        "highest_volume_route_bookings": int(top_route["bookings"]),
        "top_routes": routes.sort_values(
            "bookings",
            ascending=False
        ).head(10).to_dict(orient="records"),
    }


# ============================================================
# PAYMENT INSIGHTS
# ============================================================

payment_insights = {}

if "Payment Method" in df.columns:

    payment = (
        df.groupby("Payment Method")
        .agg(
            bookings=("Booking ID", "count"),
            completed=("Is_Completed", "sum"),
            revenue=("Booking Value", "sum"),
        )
        .reset_index()
    )

    payment["completion_rate"] = (
        payment["completed"]
        / payment["bookings"]
        * 100
    )

    top_payment = payment.sort_values(
        "bookings",
        ascending=False
    ).iloc[0]

    payment_insights = {
        "most_used_payment_method": top_payment["Payment Method"],
        "most_used_payment_bookings": int(top_payment["bookings"]),
        "payment_summary": payment.to_dict(orient="records"),
    }


# ============================================================
# CANCELLATION INSIGHTS
# ============================================================

cancellation_insights = {}

if "Booking Status" in df.columns:

    cancellation_df = df[
        df["Booking Status"].isin(
            ["Cancelled by Driver", "Cancelled by Customer"]
        )
    ].copy()

    driver_cancel_reason = {}

    if "Driver Cancellation Reason" in cancellation_df.columns:

        driver_reasons = (
            cancellation_df[
                cancellation_df["Booking Status"]
                == "Cancelled by Driver"
            ]["Driver Cancellation Reason"]
            .dropna()
            .value_counts()
        )

        driver_cancel_reason = driver_reasons.head(10).to_dict()

    customer_cancel_reason = {}

    if "Reason for cancelling by Customer" in cancellation_df.columns:

        customer_reasons = (
            cancellation_df[
                cancellation_df["Booking Status"]
                == "Cancelled by Customer"
            ]["Reason for cancelling by Customer"]
            .dropna()
            .value_counts()
        )

        customer_cancel_reason = customer_reasons.head(10).to_dict()

    cancellation_insights = {
        "total_cancelled": int(len(cancellation_df)),
        "driver_cancellation_count": int(
            (
                cancellation_df["Booking Status"]
                == "Cancelled by Driver"
            ).sum()
        ),
        "customer_cancellation_count": int(
            (
                cancellation_df["Booking Status"]
                == "Cancelled by Customer"
            ).sum()
        ),
        "top_driver_cancellation_reasons": driver_cancel_reason,
        "top_customer_cancellation_reasons": customer_cancel_reason,
    }


# ============================================================
# INCOMPLETE RIDE INSIGHTS
# ============================================================

incomplete_insights = {}

if "Incomplete Rides Reason" in df.columns:

    incomplete_reasons = (
        df[df["Is_Incomplete"] == 1]["Incomplete Rides Reason"]
        .dropna()
        .value_counts()
    )

    incomplete_insights = {
        "total_incomplete": incomplete,
        "top_incomplete_reasons": incomplete_reasons.head(
            10
        ).to_dict(),
    }


# ============================================================
# RATING INSIGHTS
# ============================================================

rating_insights = {}

if not completed_df.empty:

    rating_insights = {
        "average_driver_rating": clean_number(
            avg_driver_rating
        ),
        "average_customer_rating": clean_number(
            avg_customer_rating
        ),
        "driver_rating_min": clean_number(
            completed_df["Driver Ratings"].min()
        ),
        "driver_rating_max": clean_number(
            completed_df["Driver Ratings"].max()
        ),
        "customer_rating_min": clean_number(
            completed_df["Customer Rating"].min()
        ),
        "customer_rating_max": clean_number(
            completed_df["Customer Rating"].max()
        ),
    }


# ============================================================
# REVENUE INSIGHTS
# ============================================================

revenue_insights = {
    "completed_ride_revenue": clean_number(total_revenue),
    "average_completed_booking_value": clean_number(
        avg_booking_value
    ),
    "average_completed_ride_distance": clean_number(
        avg_distance
    ),
}

if "Booking Value" in completed_df.columns:

    revenue_by_vehicle = (
        completed_df.groupby("Vehicle Type")["Booking Value"]
        .sum()
        .sort_values(ascending=False)
    )

    revenue_insights["top_revenue_vehicle"] = (
        revenue_by_vehicle.index[0]
        if not revenue_by_vehicle.empty
        else None
    )


# ============================================================
# AI MODEL INSIGHTS
# ============================================================

ai_insights = {}

metrics_file = ML_DIR / "prebooking_metrics.csv"
importance_file = ML_DIR / "prebooking_feature_importance.csv"

if metrics_file.exists():

    metrics_df = pd.read_csv(metrics_file)

    if not metrics_df.empty:

        ai_insights["model_metrics"] = (
            metrics_df.iloc[0].to_dict()
        )

if importance_file.exists():

    importance_df = pd.read_csv(importance_file)

    if not importance_df.empty:

        # Normalize column names
        importance_df.columns = [
            str(col).strip().lower()
            for col in importance_df.columns
        ]

        # Detect feature column
        feature_column = None

        for candidate in [
            "feature",
            "features",
            "feature_name",
            "variable"
        ]:
            if candidate in importance_df.columns:
                feature_column = candidate
                break

        # Detect importance column
        importance_column = None

        for candidate in [
            "importance",
            "feature_importance",
            "importance_score",
            "score",
            "value"
        ]:
            if candidate in importance_df.columns:
                importance_column = candidate
                break

        if importance_column is not None:

            importance_df[importance_column] = pd.to_numeric(
                importance_df[importance_column],
                errors="coerce"
            )

            importance_df = importance_df.dropna(
                subset=[importance_column]
            )

            importance_df = importance_df.sort_values(
                importance_column,
                ascending=False
            )

            top_features = importance_df.head(10).to_dict(
                orient="records"
            )

            ai_insights["top_predictive_features"] = top_features

        else:

            print(
                "\nWARNING: Could not identify the feature "
                "importance column."
            )

            print(
                "Available columns:",
                list(importance_df.columns)
            )

            ai_insights["top_predictive_features"] = []

# ============================================================
# BUSINESS FINDINGS
# ============================================================

findings = []

findings.append(
    f"The dataset contains {total_bookings:,} bookings, "
    f"with {completed:,} completed rides and a "
    f"{completion_rate:.2f}% completion rate."
)

findings.append(
    f"{cancelled:,} bookings were cancelled, representing "
    f"{cancellation_rate:.2f}% of all bookings."
)

findings.append(
    f"{no_driver:,} bookings had no driver found, "
    f"representing {no_driver_rate:.2f}% of total bookings."
)

findings.append(
    f"Completed rides generated approximately "
    f"{total_revenue:,.0f} in booking value."
)

if hourly_insights:

    findings.append(
        f"The highest booking volume occurs at hour "
        f"{hourly_insights['peak_hour']}:00."
    )

if vehicle_insights:

    findings.append(
        f"The highest-demand vehicle category is "
        f"{vehicle_insights['highest_demand_vehicle']}."
    )

if location_insights:

    findings.append(
        f"The highest-volume pickup location is "
        f"{location_insights['highest_demand_pickup']}."
    )

if payment_insights:

    findings.append(
        f"The most frequently used payment method is "
        f"{payment_insights['most_used_payment_method']}."
    )


# ============================================================
# ACTIONABLE RECOMMENDATIONS
# ============================================================

recommendations = []

recommendations.append(
    "Use peak-hour demand patterns to plan driver availability "
    "and reduce unserved booking requests."
)

recommendations.append(
    "Monitor locations with high booking volumes and elevated "
    "cancellation or no-driver-found rates for targeted supply allocation."
)

recommendations.append(
    "Investigate the most frequent driver and customer "
    "cancellation reasons separately because their operational causes differ."
)

recommendations.append(
    "Use vehicle-level completion and cancellation metrics "
    "to understand where different vehicle categories perform differently."
)

recommendations.append(
    "Use payment-method patterns as an input to pre-booking "
    "outcome prediction and operational monitoring."
)

recommendations.append(
    "Use the pre-booking AI model as a decision-support tool "
    "rather than treating predictions as guaranteed outcomes."
)

recommendations.append(
    "Continuously evaluate model performance on newer booking "
    "data before using predictions for operational decisions."
)


# ============================================================
# FINAL INSIGHTS OBJECT
# ============================================================

business_insights = {

    "project": "UberPulse AI",

    "title": "AI-Powered Ride-Hailing Analytics & Demand Intelligence",

    "dataset": {
        "rows": int(total_bookings),
        "source_file": "ncr_ride_bookings.csv"
    },

    "executive_kpis": {
        "total_bookings": total_bookings,
        "completed_bookings": completed,
        "cancelled_bookings": cancelled,
        "incomplete_bookings": incomplete,
        "no_driver_found": no_driver,
        "completion_rate": clean_number(completion_rate),
        "cancellation_rate": clean_number(cancellation_rate),
        "incomplete_rate": clean_number(incomplete_rate),
        "no_driver_found_rate": clean_number(no_driver_rate),
        "completed_ride_revenue": clean_number(total_revenue),
        "average_booking_value": clean_number(avg_booking_value),
        "average_ride_distance": clean_number(avg_distance),
        "average_driver_rating": clean_number(avg_driver_rating),
        "average_customer_rating": clean_number(avg_customer_rating),
    },

    "vehicle_insights": vehicle_insights,

    "hourly_insights": hourly_insights,

    "time_period_insights": time_period_insights,

    "location_insights": location_insights,

    "route_insights": route_insights,

    "payment_insights": payment_insights,

    "cancellation_insights": cancellation_insights,

    "incomplete_insights": incomplete_insights,

    "rating_insights": rating_insights,

    "revenue_insights": revenue_insights,

    "ai_insights": ai_insights,

    "key_findings": findings,

    "recommendations": recommendations,
}


# ============================================================
# SAVE JSON
# ============================================================

json_file = OUTPUT_DIR / "business_insights.json"

with open(json_file, "w", encoding="utf-8") as file:
    json.dump(
        business_insights,
        file,
        indent=4,
        ensure_ascii=False,
        default=str
    )


# ============================================================
# SAVE MARKDOWN REPORT
# ============================================================

md_file = OUTPUT_DIR / "business_insights.md"

with open(md_file, "w", encoding="utf-8") as file:

    file.write("# UberPulse AI — Business Insights\n\n")

    file.write(
        "## Executive KPIs\n\n"
    )

    file.write(
        f"- Total Bookings: **{total_bookings:,}**\n"
        f"- Completed Bookings: **{completed:,}**\n"
        f"- Cancelled Bookings: **{cancelled:,}**\n"
        f"- Incomplete Bookings: **{incomplete:,}**\n"
        f"- No Driver Found: **{no_driver:,}**\n"
        f"- Completion Rate: **{completion_rate:.2f}%**\n"
        f"- Cancellation Rate: **{cancellation_rate:.2f}%**\n"
        f"- Incomplete Rate: **{incomplete_rate:.2f}%**\n"
        f"- No Driver Found Rate: **{no_driver_rate:.2f}%**\n"
        f"- Completed Ride Revenue: **{total_revenue:,.0f}**\n"
        f"- Average Booking Value: **{avg_booking_value:.2f}**\n"
        f"- Average Ride Distance: **{avg_distance:.2f}**\n"
        f"- Average Driver Rating: **{avg_driver_rating:.2f}**\n"
        f"- Average Customer Rating: **{avg_customer_rating:.2f}**\n\n"
    )

    file.write("## Key Findings\n\n")

    for finding in findings:
        file.write(f"- {finding}\n")

    file.write("\n## Operational Recommendations\n\n")

    for recommendation in recommendations:
        file.write(f"- {recommendation}\n")

    file.write("\n## AI Model\n\n")

    if ai_insights.get("model_metrics"):

        metrics = ai_insights["model_metrics"]

        for key, value in metrics.items():
            file.write(f"- {key}: **{value}**\n")

    if ai_insights.get("top_predictive_features"):

        file.write("\n### Top Predictive Features\n\n")

        for feature in ai_insights["top_predictive_features"]:

            feature_name = (
                feature.get("feature")
                or feature.get("features")
                or feature.get("feature_name")
                or feature.get("variable")
                or "Unknown"
            )

            importance_value = (
                feature.get("importance")
                or feature.get("feature_importance")
                or feature.get("importance_score")
                or feature.get("score")
                or feature.get("value")
                or 0
            )

            file.write(
                f"- {feature_name}: {importance_value}\n"
            )

        file.write(
            "\n---\n\n"
            "Generated by **UberPulse AI — AI-Powered Ride-Hailing "
            "Analytics & Demand Intelligence**.\n"
        )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS INSIGHTS GENERATED")
print("=" * 70)

print(f"\nTotal bookings       : {total_bookings:,}")
print(f"Completed bookings   : {completed:,}")
print(f"Cancellation rate    : {cancellation_rate:.2f}%")
print(f"Completion rate      : {completion_rate:.2f}%")
print(f"No-driver rate       : {no_driver_rate:.2f}%")
print(f"Completed revenue    : {total_revenue:,.0f}")

if hourly_insights:
    print(
        f"Peak booking hour   : "
        f"{hourly_insights['peak_hour']}:00"
    )

if vehicle_insights:
    print(
        f"Top vehicle         : "
        f"{vehicle_insights['highest_demand_vehicle']}"
    )

if location_insights:
    print(
        f"Top pickup location : "
        f"{location_insights['highest_demand_pickup']}"
    )

if payment_insights:
    print(
        f"Top payment method  : "
        f"{payment_insights['most_used_payment_method']}"
    )

print("\nFiles generated:")

print(f"  {json_file}")
print(f"  {md_file}")

print("\n" + "=" * 70)