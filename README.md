# Predicting Power Outage Duration in the U.S.
*Author: Spence Lowery – lospence@umich.edu*

## Introduction

This project investigates the characteristics and predictors of major power outages in the United States from 2000 to 2016. We focus specifically on forecasting how long a power outage will last, using factors such as the number of customers affected, weather anomalies, and the root cause of the outage. Understanding these patterns can help utilities allocate restoration resources more effectively and improve infrastructure resilience.

My central question is:

**Can we predict the duration of a power outage, using only information available at the time the outage begins?**

We use data provided in the Power Outages dataset, which contains 1534 outages across all 50 states. Key features include:

| Column Name             | Description |
|-------------------------|-------------|
| OUTAGE.DURATION         | Duration of the outage (minutes) |
| CUSTOMERS.AFFECTED      | Number of people affected |
| CAUSE.CATEGORY          | Root cause of the outage (weather, equipment failure, etc.) |
| ANOMALY.LEVEL           | Temperature deviation from average |
| RES.CUSTOMERS, COM.CUSTOMERS, IND.CUSTOMERS | Breakdown of customers in each sector |

---

## Data Cleaning and Exploratory Data Analysis

To prepare the dataset for analysis:
- Converted `OUTAGE.DURATION` from minutes to hours.
- Converted 35+ object columns into numerical types (e.g., prices, percentages).
- Dropped high-null columns like `HURRICANE.NAMES` and timestamp strings.
- Imputed missing values in `CUSTOMERS.AFFECTED` using the median.

### Distribution of Outage Duration
<iframe src="assets/duration_hist.html" width="800" height="600" frameborder="0"></iframe>

The raw outage duration distribution is highly skewed. Most outages last under 100 hours, but some extreme events last over 3000 hours. We apply a log transformation for modeling.

### Duration vs Cause Category
<iframe src="assets/cause_vs_duration.html" width="800" height="600" frameborder="0"></iframe>

Outages caused by equipment failure or fuel supply issues tend to last much longer than those caused by weather or intentional attacks.

### Duration vs Customers Affected
<iframe src="assets/duration_vs_customers.html" width="800" height="600" frameborder="0"></iframe>

There is no strong correlation between how many people are affected and how long an outage lasts. This suggests other features, like the cause or infrastructure factors, may be more important.

---

## Framing a Prediction Problem

We aim to predict the **duration** of a power outage, in hours, using features available at the start of the outage. Since duration is a continuous variable, this is a **regression** problem.

- **Target (y):** `log(1 + OUTAGE.DURATION.HOURS)`
- **Features (X):**
  - `CUSTOMERS.AFFECTED` (numeric)
  - `CAUSE.CATEGORY` (categorical)
- We selected RMSE (Root Mean Squared Error) as our metric to penalize large errors and keep results interpretable in original units.

---

## Baseline Model

We built a baseline model using a simple **linear regression** pipeline with:
- StandardScaler on numerical features
- OneHotEncoder on categorical features

Train/Test split: 80/20  
Evaluation Metric: RMSE on log-transformed target

```python
Baseline RMSE (log-hours): 1.3754
```

This corresponds to an average prediction error of ~2.4 hours on the original scale. A solid starting point.

---

## Final Model

We engineered 3 new features:
- `TOTAL.ESTIMATED.CUSTOMERS`: sum of RES, COM, and IND customers
- `PCT_CUSTOMERS_AFFECTED`: ratio of affected customers to total
- `AFFECTED_X_ANOMALY`: interaction between anomaly level and customer impact

We used a `RandomForestRegressor` with a tuned pipeline:

**Best Parameters (via GridSearchCV):**
- `n_estimators`: 200
- `max_depth`: 10
- `max_features`: "sqrt"

```python
Final Test RMSE (log-hours): 1.2209
```

This represents an ~11% improvement over baseline.

### Actual vs Predicted Duration
<iframe src="assets/pred_vs_actual.html" width="800" height="600" frameborder="0"></iframe>

The model is reasonably accurate for short to medium outages. For extreme outliers, predictions tend to under-shoot — common in imbalanced regression problems.
