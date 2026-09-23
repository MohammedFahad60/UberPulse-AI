from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# UBERPULSE AI - EDA & KPI ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "uberpulse_cleaned.csv"

REPORT_DIR = PROJECT_ROOT / "reports" / "eda"
CHART_DIR = REPORT_DIR / "charts"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_csv(df, filename):
    path = REPORT_DIR / filename
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def save_chart(filename):
    path = CHART_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart: {path}")


def percentage(part, total):
    if total == 0:
        return 0
    return round((part / total) * 100, 2)


def safe_mean(series):
    return round(float(series.mean()), 2) if series.notna().any() else 0


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

    print(f"\nLoading dataset:")
    print(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns)}")

    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(
            df["DateTime"],
            errors="coerce"
        )

    return df


# ============================================================
# 1. EXECUTIVE KPI ANALYSIS
# ============================================================

def executive_kpis(df):

    print("\n" + "=" * 70)
    print("1. EXECUTIVE KPIs")
    print("=" * 70)

    total_bookings = len(df)

    completed = int(df["Is_Completed"].sum())
    cancelled = int(df["Is_Cancelled"].sum())
    incomplete = int(df["Is_Incomplete"].sum())
    no_driver = int(df["Is_No_Driver_Found"].sum())

    total_revenue = float(df["Revenue"].sum())

    avg_booking_value = (
        df.loc[
            df["Is_Completed"] == 1,
            "Booking Value"
        ].mean()
    )

    avg_ride_distance = (
        df.loc[
            df["Is_Completed"] == 1,
            "Ride Distance"
        ].mean()
    )

    avg_driver_rating = (
        df.loc[
            df["Is_Completed"] == 1,
            "Driver Ratings"
        ].mean()
    )

    avg_customer_rating = (
        df.loc[
            df["Is_Completed"] == 1,
            "Customer Rating"
        ].mean()
    )

    completion_rate = percentage(
        completed,
        total_bookings
    )

    cancellation_rate = percentage(
        cancelled,
        total_bookings
    )

    no_driver_rate = percentage(
        no_driver,
        total_bookings
    )

    incomplete_rate = percentage(
        incomplete,
        total_bookings
    )

    kpis = pd.DataFrame(
        [
            {
                "Metric": "Total Bookings",
                "Value": total_bookings,
            },
            {
                "Metric": "Completed Bookings",
                "Value": completed,
            },
            {
                "Metric": "Cancelled Bookings",
                "Value": cancelled,
            },
            {
                "Metric": "Incomplete Bookings",
                "Value": incomplete,
            },
            {
                "Metric": "No Driver Found",
                "Value": no_driver,
            },
            {
                "Metric": "Completion Rate (%)",
                "Value": completion_rate,
            },
            {
                "Metric": "Cancellation Rate (%)",
                "Value": cancellation_rate,
            },
            {
                "Metric": "No Driver Found Rate (%)",
                "Value": no_driver_rate,
            },
            {
                "Metric": "Incomplete Rate (%)",
                "Value": incomplete_rate,
            },
            {
                "Metric": "Completed Ride Revenue",
                "Value": round(total_revenue, 2),
            },
            {
                "Metric": "Average Booking Value",
                "Value": round(avg_booking_value, 2),
            },
            {
                "Metric": "Average Ride Distance",
                "Value": round(avg_ride_distance, 2),
            },
            {
                "Metric": "Average Driver Rating",
                "Value": round(avg_driver_rating, 2),
            },
            {
                "Metric": "Average Customer Rating",
                "Value": round(avg_customer_rating, 2),
            },
        ]
    )

    save_csv(kpis, "executive_kpis.csv")

    print("\nExecutive KPIs:")
    print(kpis.to_string(index=False))


# ============================================================
# 2. BOOKING STATUS ANALYSIS
# ============================================================

def booking_status_analysis(df):

    print("\n" + "=" * 70)
    print("2. BOOKING STATUS ANALYSIS")
    print("=" * 70)

    status = (
        df.groupby("Booking Status")
        .agg(
            Bookings=("Booking ID", "count"),
            Revenue=("Revenue", "sum"),
            Avg_Booking_Value=("Booking Value", "mean"),
            Avg_Ride_Distance=("Ride Distance", "mean"),
        )
        .reset_index()
    )

    status["Booking_Percentage"] = (
        status["Bookings"]
        / len(df)
        * 100
    ).round(2)

    status["Revenue"] = status["Revenue"].round(2)
    status["Avg_Booking_Value"] = status[
        "Avg_Booking_Value"
    ].round(2)
    status["Avg_Ride_Distance"] = status[
        "Avg_Ride_Distance"
    ].round(2)

    save_csv(
        status,
        "booking_status_analysis.csv"
    )

    # Chart
    plt.figure(figsize=(10, 6))

    plt.bar(
        status["Booking Status"],
        status["Bookings"]
    )

    plt.title("Bookings by Status")
    plt.xlabel("Booking Status")
    plt.ylabel("Number of Bookings")
    plt.xticks(rotation=25)

    save_chart("booking_status.png")


# ============================================================
# 3. VEHICLE PERFORMANCE
# ============================================================

def vehicle_analysis(df):

    print("\n" + "=" * 70)
    print("3. VEHICLE PERFORMANCE")
    print("=" * 70)

    vehicle = (
        df.groupby("Vehicle Type")
        .agg(
            Total_Bookings=("Booking ID", "count"),
            Completed_Bookings=("Is_Completed", "sum"),
            Cancelled_Bookings=("Is_Cancelled", "sum"),
            Revenue=("Revenue", "sum"),
            Avg_Booking_Value=("Booking Value", "mean"),
            Avg_Ride_Distance=("Ride Distance", "mean"),
            Avg_Driver_Rating=("Driver Ratings", "mean"),
            Avg_Customer_Rating=("Customer Rating", "mean"),
        )
        .reset_index()
    )

    vehicle["Completion_Rate"] = (
        vehicle["Completed_Bookings"]
        / vehicle["Total_Bookings"]
        * 100
    ).round(2)

    vehicle["Cancellation_Rate"] = (
        vehicle["Cancelled_Bookings"]
        / vehicle["Total_Bookings"]
        * 100
    ).round(2)

    numeric_columns = [
        "Revenue",
        "Avg_Booking_Value",
        "Avg_Ride_Distance",
        "Avg_Driver_Rating",
        "Avg_Customer_Rating",
    ]

    vehicle[numeric_columns] = vehicle[
        numeric_columns
    ].round(2)

    save_csv(
        vehicle.sort_values(
            "Revenue",
            ascending=False
        ),
        "vehicle_performance.csv"
    )

    # Bookings
    plt.figure(figsize=(10, 6))

    plt.bar(
        vehicle["Vehicle Type"],
        vehicle["Total_Bookings"]
    )

    plt.title("Bookings by Vehicle Type")
    plt.xlabel("Vehicle Type")
    plt.ylabel("Bookings")
    plt.xticks(rotation=30)

    save_chart("vehicle_bookings.png")

    # Revenue
    plt.figure(figsize=(10, 6))

    revenue_sorted = vehicle.sort_values(
        "Revenue",
        ascending=False
    )

    plt.bar(
        revenue_sorted["Vehicle Type"],
        revenue_sorted["Revenue"]
    )

    plt.title("Revenue by Vehicle Type")
    plt.xlabel("Vehicle Type")
    plt.ylabel("Revenue")
    plt.xticks(rotation=30)

    save_chart("vehicle_revenue.png")


# ============================================================
# 4. TIME ANALYSIS
# ============================================================

def time_analysis(df):

    print("\n" + "=" * 70)
    print("4. TIME & DEMAND ANALYSIS")
    print("=" * 70)

    hourly = (
        df.groupby("Hour")
        .agg(
            Bookings=("Booking ID", "count"),
            Completed=("Is_Completed", "sum"),
            Cancelled=("Is_Cancelled", "sum"),
            Revenue=("Revenue", "sum"),
        )
        .reset_index()
    )

    hourly["Completion_Rate"] = (
        hourly["Completed"]
        / hourly["Bookings"]
        * 100
    ).round(2)

    hourly["Cancellation_Rate"] = (
        hourly["Cancelled"]
        / hourly["Bookings"]
        * 100
    ).round(2)

    save_csv(
        hourly,
        "hourly_demand_analysis.csv"
    )

    # Hourly bookings
    plt.figure(figsize=(11, 6))

    plt.plot(
        hourly["Hour"],
        hourly["Bookings"],
        marker="o"
    )

    plt.title("Hourly Booking Demand")
    plt.xlabel("Hour of Day")
    plt.ylabel("Bookings")
    plt.xticks(range(24))

    save_chart("hourly_booking_demand.png")

    # Day of week
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    daily = (
        df.groupby("Day_Name")
        .agg(
            Bookings=("Booking ID", "count"),
            Completed=("Is_Completed", "sum"),
            Cancelled=("Is_Cancelled", "sum"),
            Revenue=("Revenue", "sum"),
        )
        .reindex(day_order)
        .reset_index()
    )

    daily["Completion_Rate"] = (
        daily["Completed"]
        / daily["Bookings"]
        * 100
    ).round(2)

    daily["Cancellation_Rate"] = (
        daily["Cancelled"]
        / daily["Bookings"]
        * 100
    ).round(2)

    save_csv(
        daily,
        "day_of_week_analysis.csv"
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        daily["Day_Name"],
        daily["Bookings"]
    )

    plt.title("Bookings by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Bookings")
    plt.xticks(rotation=30)

    save_chart("day_of_week_bookings.png")

    # Time period
    period = (
        df.groupby("Time_Period")
        .agg(
            Bookings=("Booking ID", "count"),
            Completed=("Is_Completed", "sum"),
            Cancelled=("Is_Cancelled", "sum"),
            Revenue=("Revenue", "sum"),
        )
        .reset_index()
    )

    period["Completion_Rate"] = (
        period["Completed"]
        / period["Bookings"]
        * 100
    ).round(2)

    period["Cancellation_Rate"] = (
        period["Cancelled"]
        / period["Bookings"]
        * 100
    ).round(2)

    save_csv(
        period,
        "time_period_analysis.csv"
    )


# ============================================================
# 5. REVENUE ANALYSIS
# ============================================================

def revenue_analysis(df):

    print("\n" + "=" * 70)
    print("5. REVENUE ANALYSIS")
    print("=" * 70)

    completed = df[
        df["Is_Completed"] == 1
    ].copy()

    revenue_summary = pd.DataFrame(
        [
            {
                "Metric": "Total Revenue",
                "Value": completed["Booking Value"].sum(),
            },
            {
                "Metric": "Average Revenue per Completed Ride",
                "Value": completed["Booking Value"].mean(),
            },
            {
                "Metric": "Median Revenue per Completed Ride",
                "Value": completed["Booking Value"].median(),
            },
            {
                "Metric": "Minimum Booking Value",
                "Value": completed["Booking Value"].min(),
            },
            {
                "Metric": "Maximum Booking Value",
                "Value": completed["Booking Value"].max(),
            },
            {
                "Metric": "Average Ride Distance",
                "Value": completed["Ride Distance"].mean(),
            },
            {
                "Metric": "Average Fare per KM",
                "Value": completed["Fare_Per_KM"].mean(),
            },
        ]
    )

    revenue_summary["Value"] = revenue_summary[
        "Value"
    ].round(2)

    save_csv(
        revenue_summary,
        "revenue_summary.csv"
    )

    # Revenue by month
    monthly = (
        completed.groupby(
            ["Year", "Month"]
        )
        .agg(
            Completed_Bookings=("Booking ID", "count"),
            Revenue=("Booking Value", "sum"),
            Avg_Booking_Value=("Booking Value", "mean"),
        )
        .reset_index()
    )

    monthly["Revenue"] = monthly["Revenue"].round(2)
    monthly["Avg_Booking_Value"] = monthly[
        "Avg_Booking_Value"
    ].round(2)

    save_csv(
        monthly,
        "monthly_revenue.csv"
    )

    if len(monthly) > 1:

        labels = (
            monthly["Year"].astype(str)
            + "-"
            + monthly["Month"].astype(str).str.zfill(2)
        )

        plt.figure(figsize=(10, 6))

        plt.plot(
            labels,
            monthly["Revenue"],
            marker="o"
        )

        plt.title("Monthly Revenue")
        plt.xlabel("Month")
        plt.ylabel("Revenue")
        plt.xticks(rotation=30)

        save_chart("monthly_revenue.png")


# ============================================================
# 6. CANCELLATION ANALYSIS
# ============================================================

def cancellation_analysis(df):

    print("\n" + "=" * 70)
    print("6. CANCELLATION INTELLIGENCE")
    print("=" * 70)

    # Customer cancellation reasons
    if "Reason for cancelling by Customer" in df.columns:

        customer_cancel = (
            df[
                df["Booking Status"]
                == "Cancelled by Customer"
            ]
            .groupby(
                "Reason for cancelling by Customer"
            )
            .agg(
                Cancelled_Bookings=("Booking ID", "count")
            )
            .reset_index()
            .sort_values(
                "Cancelled_Bookings",
                ascending=False
            )
        )

        customer_cancel["Percentage"] = (
            customer_cancel["Cancelled_Bookings"]
            / customer_cancel["Cancelled_Bookings"].sum()
            * 100
        ).round(2)

        save_csv(
            customer_cancel,
            "customer_cancellation_reasons.csv"
        )

    # Driver cancellation reasons
    if "Driver Cancellation Reason" in df.columns:

        driver_cancel = (
            df[
                df["Booking Status"]
                == "Cancelled by Driver"
            ]
            .groupby(
                "Driver Cancellation Reason"
            )
            .agg(
                Cancelled_Bookings=("Booking ID", "count")
            )
            .reset_index()
            .sort_values(
                "Cancelled_Bookings",
                ascending=False
            )
        )

        driver_cancel["Percentage"] = (
            driver_cancel["Cancelled_Bookings"]
            / driver_cancel["Cancelled_Bookings"].sum()
            * 100
        ).round(2)

        save_csv(
            driver_cancel,
            "driver_cancellation_reasons.csv"
        )

    # Customer cancellation chart
    if "Reason for cancelling by Customer" in df.columns:

        chart_data = (
            df[
                df["Booking Status"]
                == "Cancelled by Customer"
            ]
            ["Reason for cancelling by Customer"]
            .value_counts()
            .head(10)
        )

        if not chart_data.empty:

            plt.figure(figsize=(11, 6))

            plt.barh(
                chart_data.index.astype(str),
                chart_data.values
            )

            plt.title(
                "Top Customer Cancellation Reasons"
            )
            plt.xlabel("Cancelled Bookings")
            plt.ylabel("Reason")

            plt.gca().invert_yaxis()

            save_chart(
                "customer_cancellation_reasons.png"
            )

    # Driver cancellation chart
    if "Driver Cancellation Reason" in df.columns:

        chart_data = (
            df[
                df["Booking Status"]
                == "Cancelled by Driver"
            ]
            ["Driver Cancellation Reason"]
            .value_counts()
            .head(10)
        )

        if not chart_data.empty:

            plt.figure(figsize=(11, 6))

            plt.barh(
                chart_data.index.astype(str),
                chart_data.values
            )

            plt.title(
                "Top Driver Cancellation Reasons"
            )
            plt.xlabel("Cancelled Bookings")
            plt.ylabel("Reason")

            plt.gca().invert_yaxis()

            save_chart(
                "driver_cancellation_reasons.png"
            )


# ============================================================
# 7. LOCATION ANALYSIS
# ============================================================

def location_analysis(df):

    print("\n" + "=" * 70)
    print("7. LOCATION & ROUTE ANALYSIS")
    print("=" * 70)

    # Pickup locations
    pickup = (
        df.groupby("Pickup Location")
        .agg(
            Total_Bookings=("Booking ID", "count"),
            Completed_Bookings=("Is_Completed", "sum"),
            Cancelled_Bookings=("Is_Cancelled", "sum"),
            Revenue=("Revenue", "sum"),
            Avg_Ride_Distance=("Ride Distance", "mean"),
        )
        .reset_index()
    )

    pickup["Completion_Rate"] = (
        pickup["Completed_Bookings"]
        / pickup["Total_Bookings"]
        * 100
    ).round(2)

    pickup["Cancellation_Rate"] = (
        pickup["Cancelled_Bookings"]
        / pickup["Total_Bookings"]
        * 100
    ).round(2)

    pickup["Revenue"] = pickup["Revenue"].round(2)
    pickup["Avg_Ride_Distance"] = pickup[
        "Avg_Ride_Distance"
    ].round(2)

    save_csv(
        pickup.sort_values(
            "Total_Bookings",
            ascending=False
        ),
        "pickup_location_analysis.csv"
    )

    # Drop locations
    drop = (
        df.groupby("Drop Location")
        .agg(
            Total_Bookings=("Booking ID", "count"),
            Completed_Bookings=("Is_Completed", "sum"),
            Cancelled_Bookings=("Is_Cancelled", "sum"),
            Revenue=("Revenue", "sum"),
        )
        .reset_index()
    )

    drop["Completion_Rate"] = (
        drop["Completed_Bookings"]
        / drop["Total_Bookings"]
        * 100
    ).round(2)

    drop["Cancellation_Rate"] = (
        drop["Cancelled_Bookings"]
        / drop["Total_Bookings"]
        * 100
    ).round(2)

    drop["Revenue"] = drop["Revenue"].round(2)

    save_csv(
        drop.sort_values(
            "Total_Bookings",
            ascending=False
        ),
        "drop_location_analysis.csv"
    )

    # Route analysis
    route = (
        df.groupby(
            [
                "Pickup Location",
                "Drop Location",
            ]
        )
        .agg(
            Total_Bookings=("Booking ID", "count"),
            Completed_Bookings=("Is_Completed", "sum"),
            Cancelled_Bookings=("Is_Cancelled", "sum"),
            Revenue=("Revenue", "sum"),
            Avg_Ride_Distance=("Ride Distance", "mean"),
            Avg_Booking_Value=("Booking Value", "mean"),
        )
        .reset_index()
    )

    route["Completion_Rate"] = (
        route["Completed_Bookings"]
        / route["Total_Bookings"]
        * 100
    ).round(2)

    route["Cancellation_Rate"] = (
        route["Cancelled_Bookings"]
        / route["Total_Bookings"]
        * 100
    ).round(2)

    route["Revenue"] = route["Revenue"].round(2)
    route["Avg_Ride_Distance"] = route[
        "Avg_Ride_Distance"
    ].round(2)
    route["Avg_Booking_Value"] = route[
        "Avg_Booking_Value"
    ].round(2)

    save_csv(
        route.sort_values(
            "Total_Bookings",
            ascending=False
        ),
        "route_analysis.csv"
    )

    # Top pickup locations chart
    top_pickups = (
        pickup.sort_values(
            "Total_Bookings",
            ascending=False
        )
        .head(15)
    )

    plt.figure(figsize=(11, 7))

    plt.barh(
        top_pickups["Pickup Location"].astype(str),
        top_pickups["Total_Bookings"]
    )

    plt.title("Top Pickup Locations by Booking Volume")
    plt.xlabel("Bookings")
    plt.ylabel("Pickup Location")

    plt.gca().invert_yaxis()

    save_chart("top_pickup_locations.png")

    # Top routes
    top_routes = (
        route.sort_values(
            "Total_Bookings",
            ascending=False
        )
        .head(15)
        .copy()
    )

    top_routes["Route"] = (
        top_routes["Pickup Location"].astype(str)
        + " → "
        + top_routes["Drop Location"].astype(str)
    )

    plt.figure(figsize=(12, 8))

    plt.barh(
        top_routes["Route"],
        top_routes["Total_Bookings"]
    )

    plt.title("Top Routes by Booking Volume")
    plt.xlabel("Bookings")
    plt.ylabel("Route")

    plt.gca().invert_yaxis()

    save_chart("top_routes.png")


# ============================================================
# 8. PAYMENT ANALYSIS
# ============================================================

def payment_analysis(df):

    print("\n" + "=" * 70)
    print("8. PAYMENT METHOD ANALYSIS")
    print("=" * 70)

    payment = (
        df.groupby("Payment Method")
        .agg(
            Bookings=("Booking ID", "count"),
            Completed_Bookings=("Is_Completed", "sum"),
            Revenue=("Revenue", "sum"),
            Avg_Booking_Value=("Booking Value", "mean"),
        )
        .reset_index()
    )

    payment["Booking_Percentage"] = (
        payment["Bookings"]
        / payment["Bookings"].sum()
        * 100
    ).round(2)

    payment["Completion_Rate"] = (
        payment["Completed_Bookings"]
        / payment["Bookings"]
        * 100
    ).round(2)

    payment["Revenue"] = payment[
        "Revenue"
    ].round(2)

    payment["Avg_Booking_Value"] = payment[
        "Avg_Booking_Value"
    ].round(2)

    save_csv(
        payment.sort_values(
            "Bookings",
            ascending=False
        ),
        "payment_method_analysis.csv"
    )

    chart_data = payment.sort_values(
        "Bookings",
        ascending=False
    )

    if not chart_data.empty:

        plt.figure(figsize=(10, 6))

        plt.bar(
            chart_data["Payment Method"].astype(str),
            chart_data["Bookings"]
        )

        plt.title("Bookings by Payment Method")
        plt.xlabel("Payment Method")
        plt.ylabel("Bookings")
        plt.xticks(rotation=25)

        save_chart("payment_methods.png")


# ============================================================
# 9. RATING ANALYSIS
# ============================================================

def rating_analysis(df):

    print("\n" + "=" * 70)
    print("9. RATING ANALYSIS")
    print("=" * 70)

    rating_summary = pd.DataFrame(
        [
            {
                "Metric": "Average Driver Rating",
                "Value": safe_mean(df["Driver Ratings"]),
            },
            {
                "Metric": "Average Customer Rating",
                "Value": safe_mean(df["Customer Rating"]),
            },
            {
                "Metric": "Median Driver Rating",
                "Value": round(
                    float(df["Driver Ratings"].median()),
                    2
                ),
            },
            {
                "Metric": "Median Customer Rating",
                "Value": round(
                    float(df["Customer Rating"].median()),
                    2
                ),
            },
        ]
    )

    save_csv(
        rating_summary,
        "rating_summary.csv"
    )

    # Driver rating distribution
    driver_ratings = (
        df["Driver Ratings"]
        .dropna()
        .value_counts()
        .sort_index()
        .reset_index()
    )

    driver_ratings.columns = [
        "Driver_Rating",
        "Count",
    ]

    save_csv(
        driver_ratings,
        "driver_rating_distribution.csv"
    )

    # Customer rating distribution
    customer_ratings = (
        df["Customer Rating"]
        .dropna()
        .value_counts()
        .sort_index()
        .reset_index()
    )

    customer_ratings.columns = [
        "Customer_Rating",
        "Count",
    ]

    save_csv(
        customer_ratings,
        "customer_rating_distribution.csv"
    )


# ============================================================
# 10. WAIT TIME ANALYSIS
# ============================================================

def wait_time_analysis(df):

    print("\n" + "=" * 70)
    print("10. WAIT TIME ANALYSIS")
    print("=" * 70)

    wait = (
        df.groupby("Booking Status")
        .agg(
            Avg_VTAT=("Avg VTAT", "mean"),
            Avg_CTAT=("Avg CTAT", "mean"),
            Bookings=("Booking ID", "count"),
        )
        .reset_index()
    )

    wait["Avg_VTAT"] = wait[
        "Avg_VTAT"
    ].round(2)

    wait["Avg_CTAT"] = wait[
        "Avg_CTAT"
    ].round(2)

    save_csv(
        wait,
        "wait_time_by_status.csv"
    )

    # VTAT by vehicle
    vehicle_wait = (
        df.groupby("Vehicle Type")
        .agg(
            Avg_VTAT=("Avg VTAT", "mean"),
            Avg_CTAT=("Avg CTAT", "mean"),
            Bookings=("Booking ID", "count"),
        )
        .reset_index()
    )

    vehicle_wait["Avg_VTAT"] = vehicle_wait[
        "Avg_VTAT"
    ].round(2)

    vehicle_wait["Avg_CTAT"] = vehicle_wait[
        "Avg_CTAT"
    ].round(2)

    save_csv(
        vehicle_wait,
        "wait_time_by_vehicle.csv"
    )


# ============================================================
# 11. RIDE DISTANCE ANALYSIS
# ============================================================

def distance_analysis(df):

    print("\n" + "=" * 70)
    print("11. RIDE DISTANCE ANALYSIS")
    print("=" * 70)

    completed = df[
        df["Is_Completed"] == 1
    ].copy()

    distance_summary = pd.DataFrame(
        [
            {
                "Metric": "Average Distance",
                "Value": completed["Ride Distance"].mean(),
            },
            {
                "Metric": "Median Distance",
                "Value": completed["Ride Distance"].median(),
            },
            {
                "Metric": "Minimum Distance",
                "Value": completed["Ride Distance"].min(),
            },
            {
                "Metric": "Maximum Distance",
                "Value": completed["Ride Distance"].max(),
            },
        ]
    )

    distance_summary["Value"] = distance_summary[
        "Value"
    ].round(2)

    save_csv(
        distance_summary,
        "distance_summary.csv"
    )

    # Distance buckets
    bins = [
        0,
        5,
        10,
        20,
        30,
        40,
        50,
        np.inf,
    ]

    labels = [
        "0-5 km",
        "5-10 km",
        "10-20 km",
        "20-30 km",
        "30-40 km",
        "40-50 km",
        "50+ km",
    ]

    completed["Distance_Bucket"] = pd.cut(
        completed["Ride Distance"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    distance_buckets = (
        completed["Distance_Bucket"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    distance_buckets.columns = [
        "Distance_Bucket",
        "Completed_Rides",
    ]

    save_csv(
        distance_buckets,
        "distance_buckets.csv"
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        completed["Ride Distance"].dropna(),
        bins=25
    )

    plt.title(
        "Ride Distance Distribution"
    )
    plt.xlabel("Ride Distance (km)")
    plt.ylabel("Completed Rides")

    save_chart("ride_distance_distribution.png")


# ============================================================
# 12. CORRELATION ANALYSIS
# ============================================================

def correlation_analysis(df):

    print("\n" + "=" * 70)
    print("12. NUMERIC CORRELATION ANALYSIS")
    print("=" * 70)

    numeric_columns = [
        "Avg VTAT",
        "Avg CTAT",
        "Booking Value",
        "Ride Distance",
        "Driver Ratings",
        "Customer Rating",
        "Revenue",
        "Fare_Per_KM",
        "Hour",
        "Is_Completed",
        "Is_Cancelled",
    ]

    available = [
        column
        for column in numeric_columns
        if column in df.columns
    ]

    correlation = df[available].corr(
        numeric_only=True
    )

    correlation.round(3).to_csv(
        REPORT_DIR / "numeric_correlation_matrix.csv"
    )

    print(
        "\nCorrelation matrix saved."
    )


# ============================================================
# 13. TOP BUSINESS INSIGHTS DATA
# ============================================================

def generate_insight_tables(df):

    print("\n" + "=" * 70)
    print("13. BUSINESS INSIGHT TABLES")
    print("=" * 70)

    # Top vehicles by revenue
    vehicle = (
        df.groupby("Vehicle Type")
        .agg(
            Bookings=("Booking ID", "count"),
            Completed=("Is_Completed", "sum"),
            Revenue=("Revenue", "sum"),
            Cancellation_Rate=("Is_Cancelled", "mean"),
        )
        .reset_index()
    )

    vehicle["Cancellation_Rate"] *= 100

    vehicle["Cancellation_Rate"] = vehicle[
        "Cancellation_Rate"
    ].round(2)

    vehicle["Revenue"] = vehicle[
        "Revenue"
    ].round(2)

    save_csv(
        vehicle.sort_values(
            "Revenue",
            ascending=False
        ).head(10),
        "top_vehicles_by_revenue.csv"
    )

    # Top pickup locations
    pickup = (
        df.groupby("Pickup Location")
        .agg(
            Bookings=("Booking ID", "count"),
            Completed=("Is_Completed", "sum"),
            Revenue=("Revenue", "sum"),
        )
        .reset_index()
    )

    save_csv(
        pickup.sort_values(
            "Bookings",
            ascending=False
        ).head(20),
        "top_pickup_locations.csv"
    )

    # Top routes
    routes = (
        df.groupby(
            [
                "Pickup Location",
                "Drop Location",
            ]
        )
        .agg(
            Bookings=("Booking ID", "count"),
            Completed=("Is_Completed", "sum"),
            Revenue=("Revenue", "sum"),
        )
        .reset_index()
    )

    routes["Route"] = (
        routes["Pickup Location"].astype(str)
        + " → "
        + routes["Drop Location"].astype(str)
    )

    save_csv(
        routes.sort_values(
            "Bookings",
            ascending=False
        ).head(20),
        "top_routes.csv"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("UBERPULSE AI")
    print("EXPLORATORY DATA ANALYSIS & KPI ENGINE")
    print("=" * 70)

    df = load_data()

    executive_kpis(df)

    booking_status_analysis(df)

    vehicle_analysis(df)

    time_analysis(df)

    revenue_analysis(df)

    cancellation_analysis(df)

    location_analysis(df)

    payment_analysis(df)

    rating_analysis(df)

    wait_time_analysis(df)

    distance_analysis(df)

    correlation_analysis(df)

    generate_insight_tables(df)

    print("\n" + "=" * 70)
    print("EDA & KPI ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nReports:")
    print(REPORT_DIR)

    print("\nCharts:")
    print(CHART_DIR)

    print("\nNext step:")
    print("AI/ML modeling")


if __name__ == "__main__":
    main()