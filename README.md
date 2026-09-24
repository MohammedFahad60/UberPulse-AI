# 🚕 UberPulse AI

## AI-Powered Ride-Hailing Analytics & Demand Intelligence

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-purple)

---

## 📌 Project Overview

**UberPulse AI** is an end-to-end Data Analytics and Artificial Intelligence project developed to analyze ride-hailing booking data and transform raw transactional records into meaningful business insights.

The project analyzes:

- Booking outcomes
- Vehicle performance
- Pickup and drop locations
- Demand patterns
- Revenue
- Ride distance
- Driver and customer ratings
- Payment methods
- Cancellation behavior
- Waiting times
- Time-based booking trends

The complete workflow combines:

**Data Audit → Data Cleaning → Feature Engineering → Exploratory Data Analysis → Business Intelligence → Machine Learning → Interactive Dashboard**

UberPulse AI also includes a **pre-booking machine learning model** that predicts whether a booking is likely to be completed using booking-context information.

> **Dataset note:** The dataset used in this project is a public NCR ride-bookings dataset. It is not proprietary or internal Uber data.

---

# 🎯 Objectives

The main objectives of UberPulse AI are:

1. Clean and validate the raw ride-booking dataset.
2. Identify missing values, duplicates, and data-quality issues.
3. Perform exploratory data analysis.
4. Analyze booking completion and cancellation patterns.
5. Analyze demand by hour, day, vehicle type, and location.
6. Analyze revenue and payment behavior.
7. Identify operational problem areas.
8. Build a leakage-aware machine learning model.
9. Predict booking completion using booking-context information.
10. Evaluate the model using standard classification metrics.
11. Generate business-oriented insights.
12. Present the results through an interactive Streamlit dashboard.

---

# 📊 Dataset

The project uses a **public NCR ride-bookings dataset** containing:

- **150,000 booking records**
- **21 original attributes**
- **176 pickup locations**
- **176 drop locations**

## Dataset Columns

| Column | Description |
|---|---|
| Date | Booking date |
| Time | Booking time |
| Booking ID | Booking identifier |
| Booking Status | Final booking outcome |
| Customer ID | Customer identifier |
| Vehicle Type | Type of vehicle booked |
| Pickup Location | Pickup location |
| Drop Location | Destination |
| Avg VTAT | Average vehicle arrival time |
| Avg CTAT | Average customer trip arrival time |
| Cancelled Rides by Customer | Customer cancellation indicator |
| Reason for cancelling by Customer | Customer cancellation reason |
| Cancelled Rides by Driver | Driver cancellation indicator |
| Driver Cancellation Reason | Driver cancellation reason |
| Incomplete Rides | Incomplete ride indicator |
| Incomplete Rides Reason | Reason for incomplete ride |
| Booking Value | Recorded booking value |
| Ride Distance | Distance of the ride |
| Driver Ratings | Driver rating |
| Customer Rating | Customer rating |
| Payment Method | Payment method |

## Booking Statuses

- Completed
- Cancelled by Driver
- Cancelled by Customer
- Incomplete
- No Driver Found

## Vehicle Types

- eBike
- Go Sedan
- Auto
- Premier Sedan
- Bike
- Go Mini
- Uber XL

## Payment Methods

- UPI
- Debit Card
- Cash
- Uber Wallet
- Credit Card

---

# 🔗 Dataset Source

The project uses the public NCR ride-bookings dataset supplied for the project analysis.

**Dataset filename used in the project:**

```text
data/raw/ncr_ride_bookings.csv
```

> The exact public dataset URL should be retained with the dataset/source information used when the project was obtained. This README does not invent a URL where the source URL is not available in the project materials.

---

# 🧩 Project Architecture

```text
                     ┌─────────────────────┐
                     │   Raw Ride Data     │
                     │     CSV Dataset     │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │    Data Audit       │
                     │ Missing Values      │
                     │ Duplicates          │
                     │ Data Validation     │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │  Data Cleaning      │
                     │ Feature Engineering │
                     └──────────┬──────────┘
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
         ┌─────────────────┐       ┌────────────────────┐
         │      EDA        │       │ Machine Learning   │
         │ KPIs & Trends   │       │ Pre-booking Model  │
         └────────┬────────┘       └──────────┬─────────┘
                  │                           │
                  └─────────────┬─────────────┘
                                ▼
                     ┌─────────────────────┐
                     │ Business Insights   │
                     │ Recommendations     │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Streamlit Dashboard │
                     └─────────────────────┘
```

---

# 🧹 Data Audit

Before cleaning, the dataset is audited for:

- Dataset shape
- Column data types
- Missing values
- Missing-value percentages
- Exact duplicate rows
- Duplicate Booking IDs
- Booking status distribution
- Numeric ranges
- Negative values
- Rating validity
- Date and time validity
- Status-dependent missingness

### Important Data Quality Findings

The raw dataset contains meaningful status-dependent missingness.

Examples:

| Field | Available Records | Approx. Missing |
|---|---:|---:|
| Avg VTAT | 139,500 | 7% |
| Avg CTAT | 102,000 | 32% |
| Booking Value | 102,000 | 32% |
| Ride Distance | 102,000 | 32% |
| Driver Ratings | 93,000 | 38% |
| Customer Rating | 93,000 | 38% |
| Driver Cancellation Reason | 27,000 | 82% |
| Customer Cancellation Reason | 10,500 | 93% |

This missingness is not blindly imputed because many fields are naturally unavailable for particular booking outcomes.

---

# 🧹 Data Cleaning

The cleaning pipeline performs:

- String standardization
- Removal of surrounding quotes from identifiers/text
- Numeric type conversion
- Date and time parsing
- Duplicate detection
- Exact duplicate removal
- Rating validation
- Numeric range validation
- Booking outcome encoding
- Temporal feature creation
- Business feature creation

### Duplicate Handling

The dataset contained:

```text
Exact duplicate rows: 0
Duplicate Booking IDs: 1,224
```

The duplicate Booking IDs were **reported and retained** because duplicate identifiers do not automatically prove that the complete analytical records are duplicates.

After cleaning and feature engineering:

```text
Rows: 150,000
Columns: 45
```

---

# ⚙️ Feature Engineering

UberPulse AI generates additional analytical features including:

```text
DateTime
Year
Month
Month_Name
Day
Day_of_Week
Day_Name
Hour
Is_Weekend
Time_Period
Is_Completed
Is_Cancelled
Is_Incomplete
Is_No_Driver_Found
Booking_Outcome
Revenue
Fare_Per_KM
```

Availability indicators are also generated for selected operational fields.

---

# 📈 Exploratory Data Analysis

The EDA workflow analyzes several dimensions of the ride-booking dataset.

## Booking Performance

- Total bookings
- Completed bookings
- Cancelled bookings
- Incomplete bookings
- No-driver-found bookings
- Completion rate
- Cancellation rate

## Demand Analysis

- Hourly demand
- Daily demand
- Day-of-week patterns
- Weekend vs weekday
- Time-period demand

## Vehicle Analysis

- Vehicle booking volume
- Completion rate
- Completed revenue
- Vehicle-level performance

## Location Analysis

- Top pickup locations
- Top drop locations
- Popular routes

## Financial Analysis

- Booking value
- Completed-ride revenue
- Revenue by vehicle
- Payment method distribution

## Customer and Driver Analysis

- Driver ratings
- Customer ratings
- Ride distance
- Waiting times

## Cancellation Analysis

- Driver cancellation reasons
- Customer cancellation reasons
- Cancellation distribution

---

# 📊 Key Business KPIs

| KPI | Value |
|---|---:|
| Total Bookings | 150,000 |
| Completed Bookings | 93,000 |
| Cancelled Bookings | 37,500 |
| Incomplete Bookings | 9,000 |
| No Driver Found | 10,500 |
| Completion Rate | 62.00% |
| Cancellation Rate | 25.00% |
| No Driver Found Rate | 7.00% |
| Incomplete Rate | 6.00% |
| Completed Ride Revenue | 47,260,574 |
| Average Booking Value | 508.18 |
| Average Ride Distance | 26.00 |
| Average Driver Rating | 4.23 |
| Average Customer Rating | 4.40 |

---

# 📌 Key Analytical Findings

The analysis identified the following descriptive patterns:

- The dataset contains **150,000 bookings**.
- **93,000 bookings** were completed.
- The overall completion rate is **62%**.
- **37,500 bookings** were cancelled.
- The cancellation rate is **25%**.
- **10,500 bookings** had a No Driver Found outcome.
- The No Driver Found rate is **7%**.
- Completed rides generated recorded revenue of **47,260,574**.
- The peak booking hour identified by the analysis is **18:00**.
- **Auto** is the top vehicle category in the generated business-insight summary.
- **Khandsa** is the top pickup location in the generated business-insight summary.
- **UPI** is the top payment method in the generated business-insight summary.

These are descriptive findings from the analyzed dataset and should not automatically be generalized to other ride-hailing markets.

---

# 🤖 Artificial Intelligence / Machine Learning

## Pre-booking Booking Outcome Prediction

UberPulse AI includes a machine learning model designed to predict whether a booking will be completed.

### Target

```text
Completed
      vs
Not Completed
```

## Features Used

The final model uses booking-context information:

```text
Vehicle Type
Pickup Location
Drop Location
Hour
Day of Week
Is Weekend
Time Period
Payment Method
```

## Algorithm

**Random Forest Classifier**

The machine learning pipeline uses:

- Categorical feature preprocessing
- One-hot encoding
- Numerical features
- Random Forest classification
- Train/test evaluation
- Classification metrics
- Confusion matrix
- Feature importance

---

# 🔐 Leakage-Aware Modeling

An initial wider model was tested using additional operational variables.

However, features such as:

- Ride Distance
- Driver Rating
- Customer Rating
- Avg CTAT
- Avg VTAT

may become available after the booking has progressed.

Using such variables for a pre-booking prediction task can introduce **data leakage**.

Therefore, the final model intentionally excludes these post-outcome or potentially post-booking variables and focuses on information that is more appropriate for the booking/request stage.

This makes the final model a more defensible baseline for pre-booking prediction.

---

# 📊 Machine Learning Results

| Metric | Result |
|---|---:|
| Accuracy | 68.48% |
| Precision | 91.17% |
| Recall | 54.42% |
| F1 Score | 68.16% |
| ROC-AUC | 72.78% |

### Classification Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Not Completed | 0.55 | 0.91 | 0.69 |
| Completed | 0.91 | 0.54 | 0.68 |

The model provides an analytical baseline for booking-completion prediction. It should be treated as a decision-support component rather than a guaranteed prediction of an individual booking outcome.

---

# 🔎 Feature Importance

The Random Forest feature-importance analysis identified payment-method categories as major contributors to the model's predictions.

Important model features included:

- UPI
- Cash
- Credit Card
- Uber Wallet
- Debit Card
- Hour
- Day of Week
- Time Period
- Vehicle Type
- Location features

> Feature importance describes patterns learned by the fitted model. It does not establish that a feature causally produces a booking outcome.

---

# 💡 Business Insights

UberPulse AI transforms the analytical results into business-oriented insights.

### Booking Operations

The booking outcome distribution helps distinguish:

- Successful rides
- Driver cancellations
- Customer cancellations
- Incomplete rides
- No-driver-found outcomes

### Demand Planning

Hourly and day-level analysis can help identify periods of higher booking activity.

### Vehicle Planning

Vehicle-level booking and revenue analysis can help understand demand across vehicle categories.

### Location Intelligence

Pickup and drop-location analysis can identify high-volume areas and commonly observed movement patterns.

### Cancellation Intelligence

Driver and customer cancellation reasons are analyzed separately because the two categories represent different operational contexts.

### Revenue Intelligence

Completed-ride booking value provides a dataset-level view of recorded revenue.

---

# 🖥️ Interactive Streamlit Dashboard

UberPulse AI includes an interactive dashboard built using **Streamlit**.

## Dashboard Features

### Executive KPIs

- Total Bookings
- Completed Bookings
- Completion Rate
- Cancellation Rate
- No Driver Found Rate
- Revenue

### Demand Intelligence

- Hourly demand
- Daily demand
- Time-period analysis

### Vehicle Analytics

- Vehicle booking volume
- Completion performance
- Revenue by vehicle

### Location Analytics

- Top pickup locations
- Top drop locations

### Cancellation Intelligence

- Top driver cancellation reasons
- Top customer cancellation reasons

### Revenue Analytics

- Booking value
- Completed revenue
- Vehicle-level revenue

### AI Analytics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion matrix
- Feature importance

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Joblib | Model serialization |
| Plotly | Interactive visualization |
| Matplotlib | Analytical visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive dashboard |
| Jupyter Notebook | Reproducible analytics |
| Git | Version control |
| GitHub | Repository hosting |

---

# 📂 Project Structure

```text
UberPulse-AI/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── ncr_ride_bookings.csv
│   │
│   └── processed/
│       └── uberpulse_cleaned.csv
│
├── models/
│   └── prebooking_outcome_model.joblib
│
├── notebooks/
│   └── MohammedFahad_UberPulseAI.ipynb
│
├── src/
│   ├── data_audit.py
│   ├── data_cleaning.py
│   ├── eda_analysis.py
│   ├── ml_booking_prediction.py
│   ├── ml_prebooking_prediction.py
│   └── business_insights.py
│
└── reports/
    ├── cleaning/
    ├── eda/
    ├── ml/
    └── business_insights/
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd UberPulse-AI
```

## 2. Create a Virtual Environment

```bash
python -m venv myenv
```

### Windows

```bash
myenv\Scripts\activate
```

### Linux / macOS

```bash
source myenv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Data Audit

```bash
python src/data_audit.py
```

## Data Cleaning

```bash
python src/data_cleaning.py
```

## Exploratory Data Analysis

```bash
python src/eda_analysis.py
```

## Train the Pre-booking Model

```bash
python src/ml_prebooking_prediction.py
```

## Generate Business Insights

```bash
python src/business_insights.py
```

## Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard runs locally at:

```text
http://localhost:8501
```

---

# 📦 Requirements

The project uses the following Python libraries:

```text
streamlit
pandas
numpy
scikit-learn
joblib
plotly
matplotlib
seaborn
```

---

# 📁 Generated Outputs

The project generates the following important outputs:

```text
data/processed/uberpulse_cleaned.csv

models/prebooking_outcome_model.joblib

reports/ml/prebooking_metrics.csv
reports/ml/prebooking_predictions.csv
reports/ml/prebooking_feature_importance.csv
reports/ml/prebooking_confusion_matrix.csv

reports/business_insights/business_insights.json
reports/business_insights/business_insights.md
```

---

# 🚀 Deployment

The Streamlit dashboard can be deployed using **Streamlit Community Cloud**.

### Deployment Steps

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect the GitHub repository.
4. Select the `main` branch.
5. Set the main application file to:

```text
app.py
```

6. Deploy the application.

The project repository should contain the dependencies in:

```text
requirements.txt
```

---

# ⚠️ Limitations

- The dataset is a public NCR ride-bookings dataset and is not proprietary internal Uber data.
- The analysis represents the dataset's available geography and period.
- The machine learning model is an analytical baseline and not a guaranteed prediction system.
- Feature importance does not establish causality.
- The dataset does not contain a driver identifier, so individual driver ranking is not supported.
- Recorded booking value is not equivalent to company profit.
- Duplicate Booking IDs were detected and retained because they do not automatically establish duplicate analytical records.
- Real-world deployment would require regularly refreshed operational data and additional model validation.
- The model's current recall indicates that some completed bookings are not identified by the classifier.

---

# 🔮 Future Scope

Future versions of UberPulse AI can include:

### Demand Forecasting

- Time-series demand forecasting
- Location-level demand prediction
- Vehicle-specific demand forecasting
- Peak-period forecasting

### Advanced AI

- Gradient boosting models
- Model probability calibration
- Ensemble learning
- Model drift monitoring

### Operational Intelligence

- Real-time booking monitoring
- Driver availability forecasting
- Automated cancellation alerts
- Demand-supply imbalance detection
- Operational scenario simulation

### Dashboard Enhancements

- Real-time data refresh
- Advanced geographic maps
- Forecasting panels
- Automated alerts
- Interactive what-if analysis

---

# 🎓 Internship Context

This project was developed as part of:

**AICTE | IBM SkillsBuild Data Analytics with AI Internship Program 2026**

The project demonstrates practical application of:

- Data Fundamentals
- Data Cleaning and Preparation
- Exploratory Data Analysis
- Artificial Intelligence for Data Analytics
- Machine Learning
- Business Intelligence
- Data Visualization
- Interactive Dashboard Development

---

# 👨‍💻 Author

**Mohammed Fahad**

Data Analytics with AI

---

# 🔗 Project Links

## GitHub Repository

The complete source code, dashboard, analytical scripts, model, and documentation are available in the project repository.

```text
https://github.com/MohammedFahad60/UberPulse-AI
```

## Live Dashboard

```text
https://uberpulse-ai-data.streamlit.app/
```

## Dataset

```text
Public NCR ride-bookings dataset
```

---

# 📜 License

This project is developed for academic and educational purposes as part of the **AICTE | IBM SkillsBuild Data Analytics with AI Internship Program 2026**.

---

# 🙏 Acknowledgement

This project was developed as part of the **AICTE | IBM SkillsBuild Data Analytics with AI Internship Program 2026**.

The project applies the concepts of data analysis, data cleaning, artificial intelligence, machine learning, business intelligence, and visualization learned during the internship program.

---

## ⭐ UberPulse AI

**Turning ride-booking data into actionable intelligence.**
