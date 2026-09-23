from pathlib import Path
import json

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "processed" / "uberpulse_cleaned.csv"
INSIGHTS_FILE = (
    BASE_DIR
    / "reports"
    / "business_insights"
    / "business_insights.json"
)
ML_METRICS_FILE = BASE_DIR / "reports" / "ml" / "prebooking_metrics.csv"
ML_IMPORTANCE_FILE = (
    BASE_DIR / "reports" / "ml" / "prebooking_feature_importance.csv"
)


st.set_page_config(
    page_title="UberPulse AI",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.5rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(128,128,128,0.2);
        background: linear-gradient(
            135deg,
            rgba(255,193,7,0.14),
            rgba(0,0,0,0.04)
        );
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.75;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }

    .insight-box {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.22);
        margin-bottom: 0.7rem;
    }

    .metric-label {
        font-size: 0.85rem;
        opacity: 0.7;
    }

    .metric-value {
        font-size: 1.55rem;
        font-weight: 750;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not DATA_FILE.exists():
        return pd.DataFrame()

    data = pd.read_csv(DATA_FILE)

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
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    if "DateTime" in data.columns:
        data["DateTime"] = pd.to_datetime(
            data["DateTime"],
            errors="coerce"
        )

    return data


@st.cache_data
def load_insights():

    if not INSIGHTS_FILE.exists():
        return {}

    with open(
        INSIGHTS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


@st.cache_data
def load_ml_metrics():

    if not ML_METRICS_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(ML_METRICS_FILE)


@st.cache_data
def load_ml_importance():

    if not ML_IMPORTANCE_FILE.exists():
        return pd.DataFrame()

    return pd.read_csv(ML_IMPORTANCE_FILE)


df = load_data()
insights = load_insights()
ml_metrics = load_ml_metrics()
ml_importance = load_ml_importance()


# ============================================================
# ERROR HANDLING
# ============================================================

if df.empty:

    st.error(
        "Cleaned dataset not found.\n\n"
        "Run:\n"
        "`python src\\data_cleaning.py`"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🚕 UberPulse AI
        </div>

        <div class="hero-subtitle">
            AI-Powered Ride-Hailing Analytics & Demand Intelligence
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


st.caption(
    "Interactive analytics dashboard built from the processed NCR "
    "ride-bookings dataset."
)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("🎛️ Dashboard Filters")

st.sidebar.markdown("---")


# Vehicle filter

if "Vehicle Type" in df.columns:

    vehicle_options = sorted(
        df["Vehicle Type"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_vehicles = st.sidebar.multiselect(
        "Vehicle Type",
        vehicle_options,
        default=vehicle_options,
    )

else:

    selected_vehicles = []


# Booking status

if "Booking Status" in df.columns:

    status_options = sorted(
        df["Booking Status"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_statuses = st.sidebar.multiselect(
        "Booking Status",
        status_options,
        default=status_options,
    )

else:

    selected_statuses = []


# Payment

if "Payment Method" in df.columns:

    payment_options = sorted(
        df["Payment Method"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_payments = st.sidebar.multiselect(
        "Payment Method",
        payment_options,
        default=payment_options,
    )

else:

    selected_payments = []


# Day type

day_type = st.sidebar.radio(
    "Day Type",
    ["All Days", "Weekdays", "Weekends"],
)


# Hour range

if "Hour" in df.columns:

    min_hour = int(df["Hour"].min())
    max_hour = int(df["Hour"].max())

    selected_hours = st.sidebar.slider(
        "Hour Range",
        min_value=min_hour,
        max_value=max_hour,
        value=(min_hour, max_hour),
    )

else:

    selected_hours = (0, 23)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()


if selected_vehicles and "Vehicle Type" in filtered_df.columns:

    filtered_df = filtered_df[
        filtered_df["Vehicle Type"].isin(selected_vehicles)
    ]


if selected_statuses and "Booking Status" in filtered_df.columns:

    filtered_df = filtered_df[
        filtered_df["Booking Status"].isin(selected_statuses)
    ]


if selected_payments and "Payment Method" in filtered_df.columns:

    filtered_df = filtered_df[
        filtered_df["Payment Method"].isin(selected_payments)
    ]


if day_type == "Weekdays" and "Is_Weekend" in filtered_df.columns:

    filtered_df = filtered_df[
        filtered_df["Is_Weekend"] == 0
    ]


elif day_type == "Weekends" and "Is_Weekend" in filtered_df.columns:

    filtered_df = filtered_df[
        filtered_df["Is_Weekend"] == 1
    ]


if "Hour" in filtered_df.columns:

    filtered_df = filtered_df[
        filtered_df["Hour"].between(
            selected_hours[0],
            selected_hours[1]
        )
    ]


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_bookings = len(filtered_df)

completed = int(
    filtered_df["Is_Completed"].sum()
)

cancelled = int(
    filtered_df["Is_Cancelled"].sum()
)

no_driver = int(
    filtered_df["Is_No_Driver_Found"].sum()
)

completion_rate = (
    completed / total_bookings * 100
    if total_bookings
    else 0
)

cancellation_rate = (
    cancelled / total_bookings * 100
    if total_bookings
    else 0
)

no_driver_rate = (
    no_driver / total_bookings * 100
    if total_bookings
    else 0
)

completed_df = filtered_df[
    filtered_df["Is_Completed"] == 1
]


revenue = (
    completed_df["Booking Value"].sum()
    if "Booking Value" in completed_df.columns
    else 0
)

average_booking_value = (
    completed_df["Booking Value"].mean()
    if not completed_df.empty
    else 0
)

average_distance = (
    completed_df["Ride Distance"].mean()
    if not completed_df.empty
    else 0
)


# ============================================================
# KPI CARDS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Executive Overview</div>',
    unsafe_allow_html=True,
)


kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)


with kpi1:
    st.metric(
        "Total Bookings",
        f"{total_bookings:,}"
    )

with kpi2:
    st.metric(
        "Completed",
        f"{completed:,}"
    )

with kpi3:
    st.metric(
        "Completion Rate",
        f"{completion_rate:.1f}%"
    )

with kpi4:
    st.metric(
        "Cancellation Rate",
        f"{cancellation_rate:.1f}%"
    )

with kpi5:
    st.metric(
        "No Driver Found",
        f"{no_driver_rate:.1f}%"
    )

with kpi6:
    st.metric(
        "Revenue",
        f"{revenue:,.0f}"
    )


st.markdown("---")


# ============================================================
# BOOKING STATUS + HOURLY DEMAND
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.markdown(
        '<div class="section-title">Booking Outcomes</div>',
        unsafe_allow_html=True
    )

    status_counts = (
        filtered_df["Booking Status"]
        .value_counts()
        .reset_index()
    )

    status_counts.columns = [
        "Booking Status",
        "Bookings"
    ]

    fig = px.pie(
        status_counts,
        names="Booking Status",
        values="Bookings",
        hole=0.55,
        title="Booking Status Distribution",
    )

    fig.update_layout(
        margin=dict(t=60, b=20, l=20, r=20),
        legend_title_text="",
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


with col2:

    st.markdown(
        '<div class="section-title">Demand by Hour</div>',
        unsafe_allow_html=True
    )

    hourly = (
        filtered_df
        .groupby("Hour")
        .size()
        .reset_index(name="Bookings")
    )

    fig = px.line(
        hourly,
        x="Hour",
        y="Bookings",
        markers=True,
        title="Hourly Booking Demand",
    )

    fig.update_xaxes(
        dtick=1
    )

    fig.update_layout(
        margin=dict(t=60, b=20, l=20, r=20)
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# ============================================================
# VEHICLE PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">🚗 Vehicle Performance</div>',
    unsafe_allow_html=True
)


vehicle = (
    filtered_df
    .groupby("Vehicle Type")
    .agg(
        Bookings=("Booking ID", "count"),
        Completed=("Is_Completed", "sum"),
        Cancelled=("Is_Cancelled", "sum"),
        Revenue=("Booking Value", "sum"),
    )
    .reset_index()
)

vehicle["Completion Rate"] = (
    vehicle["Completed"]
    / vehicle["Bookings"]
    * 100
)

vehicle["Cancellation Rate"] = (
    vehicle["Cancelled"]
    / vehicle["Bookings"]
    * 100
)


vcol1, vcol2 = st.columns(2)


with vcol1:

    fig = px.bar(
        vehicle.sort_values(
            "Bookings",
            ascending=False
        ),
        x="Vehicle Type",
        y="Bookings",
        title="Bookings by Vehicle Type",
        text_auto=True,
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


with vcol2:

    fig = px.bar(
        vehicle.sort_values(
            "Revenue",
            ascending=False
        ),
        x="Vehicle Type",
        y="Revenue",
        title="Revenue by Vehicle Type",
        text_auto=".2s",
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


st.dataframe(
    vehicle.sort_values(
        "Bookings",
        ascending=False
    ),
    width="stretch",
    hide_index=True,
)


# ============================================================
# TIME ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">⏰ Time Intelligence</div>',
    unsafe_allow_html=True
)


tcol1, tcol2 = st.columns(2)


with tcol1:

    if "Time_Period" in filtered_df.columns:

        period = (
            filtered_df
            .groupby("Time_Period")
            .size()
            .reset_index(name="Bookings")
        )

        fig = px.bar(
            period,
            x="Time_Period",
            y="Bookings",
            title="Bookings by Time Period",
            text_auto=True,
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


with tcol2:

    if "Day_Name" in filtered_df.columns:

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
            filtered_df
            .groupby("Day_Name")
            .size()
            .reindex(day_order)
            .reset_index(name="Bookings")
        )

        fig = px.bar(
            daily,
            x="Day_Name",
            y="Bookings",
            title="Bookings by Day of Week",
            text_auto=True,
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


# ============================================================
# REVENUE ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">💰 Revenue Intelligence</div>',
    unsafe_allow_html=True
)


rcol1, rcol2 = st.columns(2)


with rcol1:

    if "DateTime" in completed_df.columns:

        revenue_trend = (
            completed_df
            .set_index("DateTime")
            .resample("D")["Booking Value"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            revenue_trend,
            x="DateTime",
            y="Booking Value",
            title="Daily Revenue",
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


with rcol2:

    if "Payment Method" in filtered_df.columns:

        payment = (
            filtered_df
            .groupby("Payment Method")
            .agg(
                Bookings=("Booking ID", "count"),
                Revenue=("Booking Value", "sum"),
            )
            .reset_index()
        )

        fig = px.bar(
            payment.sort_values(
                "Bookings",
                ascending=False
            ),
            x="Payment Method",
            y="Bookings",
            title="Bookings by Payment Method",
            text_auto=True,
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


# ============================================================
# LOCATION ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">📍 Location Intelligence</div>',
    unsafe_allow_html=True
)


lcol1, lcol2 = st.columns(2)


with lcol1:

    pickup = (
        filtered_df
        .groupby("Pickup Location")
        .size()
        .reset_index(name="Bookings")
        .sort_values(
            "Bookings",
            ascending=False
        )
        .head(15)
    )

    fig = px.bar(
        pickup.sort_values("Bookings"),
        x="Bookings",
        y="Pickup Location",
        orientation="h",
        title="Top Pickup Locations",
        text_auto=True,
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


with lcol2:

    drop = (
        filtered_df
        .groupby("Drop Location")
        .size()
        .reset_index(name="Bookings")
        .sort_values(
            "Bookings",
            ascending=False
        )
        .head(15)
    )

    fig = px.bar(
        drop.sort_values("Bookings"),
        x="Bookings",
        y="Drop Location",
        orientation="h",
        title="Top Drop Locations",
        text_auto=True,
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# ============================================================
# ROUTE ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">🛣️ Route Intelligence</div>',
    unsafe_allow_html=True
)


route_df = filtered_df.copy()

route_df["Route"] = (
    route_df["Pickup Location"].astype(str)
    + " → "
    + route_df["Drop Location"].astype(str)
)


routes = (
    route_df
    .groupby("Route")
    .agg(
        Bookings=("Booking ID", "count"),
        Completed=("Is_Completed", "sum"),
        Revenue=("Booking Value", "sum"),
    )
    .reset_index()
)

routes["Completion Rate"] = (
    routes["Completed"]
    / routes["Bookings"]
    * 100
)

routes = routes.sort_values(
    "Bookings",
    ascending=False
).head(15)


st.dataframe(
    routes,
    width="stretch",
    hide_index=True,
)


# ============================================================
# CANCELLATION ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">❌ Cancellation Intelligence</div>',
    unsafe_allow_html=True
)


ccol1, ccol2 = st.columns(2)


with ccol1:

    if "Driver Cancellation Reason" in filtered_df.columns:

        driver_cancel = (
            filtered_df[
                filtered_df["Booking Status"]
                == "Cancelled by Driver"
            ]["Driver Cancellation Reason"]
            .dropna()
            .value_counts()
            .head(10)
            .reset_index()
        )

        driver_cancel.columns = [
            "Reason",
            "Bookings"
        ]

        fig = px.bar(
            driver_cancel.sort_values("Bookings"),
            x="Bookings",
            y="Reason",
            orientation="h",
            title="Top Driver Cancellation Reasons",
            text_auto=True,
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


with ccol2:

    if "Reason for cancelling by Customer" in filtered_df.columns:

        customer_cancel = (
            filtered_df[
                filtered_df["Booking Status"]
                == "Cancelled by Customer"
            ]["Reason for cancelling by Customer"]
            .dropna()
            .value_counts()
            .head(10)
            .reset_index()
        )

        customer_cancel.columns = [
            "Reason",
            "Bookings"
        ]

        fig = px.bar(
            customer_cancel.sort_values("Bookings"),
            x="Bookings",
            y="Reason",
            orientation="h",
            title="Top Customer Cancellation Reasons",
            text_auto=True,
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


# ============================================================
# RATINGS + DISTANCE
# ============================================================

st.markdown(
    '<div class="section-title">⭐ Ride Quality & Ratings</div>',
    unsafe_allow_html=True
)


qcol1, qcol2, qcol3 = st.columns(3)


with qcol1:

    avg_driver_rating = (
        completed_df["Driver Ratings"].mean()
        if "Driver Ratings" in completed_df.columns
        else 0
    )

    st.metric(
        "Average Driver Rating",
        f"{avg_driver_rating:.2f}"
    )


with qcol2:

    avg_customer_rating = (
        completed_df["Customer Rating"].mean()
        if "Customer Rating" in completed_df.columns
        else 0
    )

    st.metric(
        "Average Customer Rating",
        f"{avg_customer_rating:.2f}"
    )


with qcol3:

    st.metric(
        "Average Ride Distance",
        f"{average_distance:.2f} km"
    )


qcol4, qcol5 = st.columns(2)


with qcol4:

    if "Driver Ratings" in completed_df.columns:

        fig = px.histogram(
            completed_df,
            x="Driver Ratings",
            nbins=10,
            title="Driver Rating Distribution",
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


with qcol5:

    if (
        "Ride Distance" in completed_df.columns
        and "Booking Value" in completed_df.columns
    ):

        sample = completed_df[
            [
                "Ride Distance",
                "Booking Value"
            ]
        ].dropna()

        fig = px.scatter(
            sample.sample(
                min(5000, len(sample)),
                random_state=42
            ),
            x="Ride Distance",
            y="Booking Value",
            opacity=0.55,
            title="Ride Distance vs Booking Value",
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


# ============================================================
# AI / ML SECTION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🤖 AI Pre-Booking Prediction</div>',
    unsafe_allow_html=True
)


st.info(
    "The AI model predicts whether a booking is likely to be completed "
    "using request-time features such as vehicle type, locations, hour, "
    "day of week, time period, and payment method."
)


if not ml_metrics.empty:

    metric_columns = st.columns(
        min(5, len(ml_metrics.columns))
    )

    row = ml_metrics.iloc[0]

    displayed = 0

    for column in ml_metrics.columns:

        if displayed >= len(metric_columns):
            break

        value = row[column]

        if isinstance(value, (int, float)):

            if "accuracy" in column.lower():
                display_value = f"{value:.2%}"

            elif any(
                x in column.lower()
                for x in [
                    "precision",
                    "recall",
                    "f1",
                    "auc"
                ]
            ):
                display_value = f"{value:.2%}"

            else:
                display_value = f"{value:.4f}"

        else:

            display_value = str(value)

        with metric_columns[displayed]:

            st.metric(
                column.replace("_", " ").title(),
                display_value
            )

        displayed += 1


if not ml_importance.empty:

    st.markdown(
        "### Feature Importance"
    )

    importance = ml_importance.copy()

    importance.columns = [
        str(column).strip()
        for column in importance.columns
    ]

    feature_column = next(
        (
            column
            for column in importance.columns
            if column.lower()
            in [
                "feature",
                "features",
                "feature_name",
                "variable"
            ]
        ),
        importance.columns[0]
    )

    importance_column = next(
        (
            column
            for column in importance.columns
            if column.lower()
            in [
                "importance",
                "feature_importance",
                "importance_score",
                "score",
                "value"
            ]
        ),
        None
    )

    if importance_column:

        importance[importance_column] = pd.to_numeric(
            importance[importance_column],
            errors="coerce"
        )

        importance = (
            importance
            .dropna(subset=[importance_column])
            .sort_values(
                importance_column,
                ascending=False
            )
            .head(15)
        )

        fig = px.bar(
            importance.sort_values(
                importance_column
            ),
            x=importance_column,
            y=feature_column,
            orientation="h",
            title="Top Predictive Features",
            text_auto=".3f",
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">💡 Business Insights</div>',
    unsafe_allow_html=True
)


key_findings = insights.get(
    "key_findings",
    []
)

recommendations = insights.get(
    "recommendations",
    []
)


if key_findings:

    st.markdown("### Key Findings")

    for finding in key_findings:

        st.markdown(
            f"""
            <div class="insight-box">
                🔎 {finding}
            </div>
            """,
            unsafe_allow_html=True
        )


if recommendations:

    st.markdown("### Recommended Actions")

    for recommendation in recommendations:

        st.markdown(
            f"""
            <div class="insight-box">
                💡 {recommendation}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# DATA TABLE
# ============================================================

with st.expander("🔍 Explore Filtered Dataset"):

    st.write(
        f"Showing {len(filtered_df):,} rows"
    )

    st.dataframe(
        filtered_df.head(5000),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# DOWNLOAD
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📥 Export</div>',
    unsafe_allow_html=True
)


download_col1, download_col2 = st.columns(2)


with download_col1:

    csv_data = filtered_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Filtered Dataset",
        data=csv_data,
        file_name="uberpulse_filtered_data.csv",
        mime="text/csv",
        width="stretch",
    )


with download_col2:

    if INSIGHTS_FILE.exists():

        with open(
            INSIGHTS_FILE,
            "rb"
        ) as file:

            st.download_button(
                label="⬇️ Download Business Insights",
                data=file,
                file_name="uberpulse_business_insights.json",
                mime="application/json",
                width="stretch",
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "UberPulse AI • AI-Powered Ride-Hailing Analytics & Demand Intelligence"
)

st.caption(
    "Analytics and predictions are intended for decision support "
    "and should be validated before operational deployment."
)