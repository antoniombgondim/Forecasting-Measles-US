import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------------------------------
# Load and prepare the data
# ----------------------------------------------------
data = pd.read_csv("/Users/cleolaurindo/Downloads/Fitting Code/Measle_US/data-table (1).csv")

# Drop the first column if it is an unnecessary index column
data = data.drop(data.columns[0], axis=1)

# Convert columns
data["year"] = data["year"].astype(int)
data["cases"] = data["cases"].astype(int)

# Keep relevant columns
data = data[["year", "cases"]].dropna()
data = data[data["year"] <= 2026]

# The 2026 value is held out completely and forecasted, not trained or tested.
forecast_count_year = 2026
partial_2026_observed_date = pd.Timestamp("2026-05-01")
full_year_2026_forecast_date = pd.Timestamp("2026-12-31")
observed_2026_cases = data.loc[
    data["year"] == forecast_count_year,
    "cases",
]
observed_2026_cases = (
    float(observed_2026_cases.iloc[0]) if not observed_2026_cases.empty else np.nan
)
forecast_indices = pd.DatetimeIndex(
    [
        partial_2026_observed_date,
        full_year_2026_forecast_date,
    ],
    name="mm_yr",
)
data = data[data["year"] < forecast_count_year].copy()

# Sort by the original reporting year before computing the cumulative total.
data = data.sort_values("year").reset_index(drop=True)
data["cumulative_cases"] = data["cases"].cumsum()

# Counts are reported at January of the following year.
data["month_year"] = pd.to_datetime(
    {
        "year": data["year"] + 1,
        "month": 1,
        "day": 1,
    }
)

data = (
    data.rename(columns={"year": "count_year"})
    .sort_values("month_year")
    .set_index("month_year")
)
data.index.name = "mm_yr"

print("Prepared cumulative data:")
print(data.head())
print(data.tail())

# ----------------------------------------------------
# Training data starts at the 1996 reporting year and ends at 2025
# ----------------------------------------------------
train = data[data["count_year"] >= 1996].copy()

# ----------------------------------------------------
# Create exogenous intervention variables
# ----------------------------------------------------
def create_features(df, base_year=None):
    df = df.copy()

    years = df["count_year"]
    if base_year is None:
        base_year = years.min()

    def after_year(start_year):
        return ((years >= start_year)&(years<=2011)).astype(int)

    def outbreak_year(outbreak):
        return (years >= outbreak).astype(int)

    # Linear time trend. This is essential for cumulative counts because
    # the total should continue increasing between intervention periods.
    df["time"] = years - base_year

    # Step indicators: 1 during/after the intervention period, 0 before.
    # df["two_dose"] = after_year(1990)
    # df["vfc"] = after_year(1995)
    df["post_elimination"] = after_year(2000)

    # COVID disruption period
    df["covid_period"] = ((years >= 2020)).astype(int)

    # Recent resurgence/risk period
    # df["recent_risk"] = after_year(2020)

    # Major outbreak years: 1 only in the outbreak year, 0 otherwise.
    df["outbreak_2014"] = outbreak_year(2014)
    df["outbreak_2019"] = outbreak_year(2019)
    df["outbreak_2025"] = outbreak_year(2025)

    return df


train_base_year = train["count_year"].min()
train_features = create_features(train, base_year=train_base_year)

# ----------------------------------------------------
# Define cumulative outcome and predictors
# ----------------------------------------------------
y_train = train_features["cumulative_cases"]

exog_cols = [
    "time",
    # "two_dose",
    # "vfc",
    #"post_elimination",
    #"covid_period",
    # "recent_risk",
    "outbreak_2014",
    "outbreak_2019",
    "outbreak_2025",
]

X_train = train_features[exog_cols]
X_train = sm.add_constant(X_train, has_constant="add")

# ----------------------------------------------------
# Plot cumulative observed data
# ----------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(
    train.index,
    train["cumulative_cases"],
    marker="o",
    label="Observed cumulative cases",
)
plt.xlabel("Month-Year")
plt.ylabel("Cumulative measles cases")
plt.title("U.S. Cumulative Measles Cases")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("poisson_cumulative_measles_cases.png", dpi=300)
plt.show()

# ----------------------------------------------------
# Fit Poisson regression model
# ----------------------------------------------------
poisson_model = sm.GLM(
    y_train,
    X_train,
    family=sm.families.Poisson(),
)

poisson_results = poisson_model.fit()

# ----------------------------------------------------
# Forecast the held-out 2026 point with 95% confidence interval
# ----------------------------------------------------
forecast = pd.DataFrame(
    {
        "count_year": [forecast_count_year] * len(forecast_indices),
        "cases": [np.nan] * len(forecast_indices),
        "cumulative_cases": [np.nan] * len(forecast_indices),
    },
    index=forecast_indices,
)
forecast_features = create_features(forecast, base_year=train_base_year)
X_forecast = sm.add_constant(
    forecast_features[exog_cols],
    has_constant="add",
)
X_forecast = X_forecast[X_train.columns]

forecast_pred = poisson_results.get_prediction(X_forecast)
forecast_summary = forecast_pred.summary_frame(alpha=0.05)
forecast_features["forecast_cumulative_cases"] = forecast_summary["mean"]
forecast_features["ci_lower_95"] = forecast_summary["mean_ci_lower"]
forecast_features["ci_upper_95"] = forecast_summary["mean_ci_upper"]

forecast_output = forecast_features[
    [
        "count_year",
        "forecast_cumulative_cases",
        "ci_lower_95",
        "ci_upper_95",
    ]
].copy()
forecast_output.to_csv("poisson_cumulative_2026_forecast.csv")

print("\nHeld-out 2026 cumulative forecast:")
print(forecast_output)

# ----------------------------------------------------
# Parameter summary / uncertainty quantification
# ----------------------------------------------------
print(poisson_results.summary())

param_summary = pd.DataFrame(
    {
        "coef": poisson_results.params,
        "std_error": poisson_results.bse,
        "z_value": poisson_results.tvalues,
        "p_value": poisson_results.pvalues,
        "ci_lower_95": poisson_results.conf_int()[0],
        "ci_upper_95": poisson_results.conf_int()[1],
    }
)

# Incidence Rate Ratios (IRR)
param_summary["IRR"] = np.exp(param_summary["coef"])
param_summary["IRR_ci_lower_95"] = np.exp(param_summary["ci_lower_95"])
param_summary["IRR_ci_upper_95"] = np.exp(param_summary["ci_upper_95"])

print("\nPoisson parameter uncertainty summary:")
print(param_summary)

param_summary.to_csv("poisson_cumulative_parameter_uncertainty.csv")

# ----------------------------------------------------
# In-sample fitted values with 95% confidence intervals
# ----------------------------------------------------
pred_train = poisson_results.get_prediction(X_train)
pred_summary = pred_train.summary_frame(alpha=0.05)

train_features["fitted_cumulative_cases"] = pred_summary["mean"]
train_features["ci_lower_95"] = pred_summary["mean_ci_lower"]
train_features["ci_upper_95"] = pred_summary["mean_ci_upper"]

prediction_plot = pd.concat(
    [
        train_features[
            [
                "fitted_cumulative_cases",
                "ci_lower_95",
                "ci_upper_95",
            ]
        ],
        forecast_features.rename(
            columns={"forecast_cumulative_cases": "fitted_cumulative_cases"}
        )[
            [
                "fitted_cumulative_cases",
                "ci_lower_95",
                "ci_upper_95",
            ]
        ],
    ]
).sort_index()

# ----------------------------------------------------
# Training accuracy
# ----------------------------------------------------
mae = mean_absolute_error(
    train_features["cumulative_cases"],
    train_features["fitted_cumulative_cases"],
)

rmse = np.sqrt(
    mean_squared_error(
        train_features["cumulative_cases"],
        train_features["fitted_cumulative_cases"],
    )
)

print(f"\nTraining MAE: {mae:.2f}")
print(f"Training RMSE: {rmse:.2f}")

# ----------------------------------------------------
# Model summary table for manuscript
# ----------------------------------------------------
cumulative_through_2025 = float(data["cumulative_cases"].iloc[-1])
partial_2026_observed_cumulative_cases = (
    cumulative_through_2025 + observed_2026_cases
    if np.isfinite(observed_2026_cases)
    else np.nan
)

full_year_forecast = forecast_features.loc[full_year_2026_forecast_date]
raw_forecast_cumulative_cases = float(
    full_year_forecast["forecast_cumulative_cases"]
)
raw_forecast_2026_cases_implied = (
    raw_forecast_cumulative_cases - cumulative_through_2025
)
adjusted_forecast_cumulative_cases = (
    max(raw_forecast_cumulative_cases, partial_2026_observed_cumulative_cases)
    if np.isfinite(partial_2026_observed_cumulative_cases)
    else raw_forecast_cumulative_cases
)
adjusted_forecast_2026_cases_implied = (
    adjusted_forecast_cumulative_cases - cumulative_through_2025
)
ci_lower_95 = float(full_year_forecast["ci_lower_95"])
ci_upper_95 = float(
    max(full_year_forecast["ci_upper_95"], adjusted_forecast_cumulative_cases)
)

model_summary = pd.DataFrame(
    [
        {
            "model": "Poisson cumulative",
            "forecast_count_year": forecast_count_year,
            "partial_2026_observed_date": partial_2026_observed_date.date().isoformat(),
            "partial_2026_observed_cumulative_cases": partial_2026_observed_cumulative_cases,
            "full_year_2026_forecast_date": full_year_2026_forecast_date.date().isoformat(),
            "raw_forecast_cumulative_cases": raw_forecast_cumulative_cases,
            "forecast_cumulative_cases": adjusted_forecast_cumulative_cases,
            "raw_forecast_2026_cases_implied": raw_forecast_2026_cases_implied,
            "forecast_2026_cases_implied": adjusted_forecast_2026_cases_implied,
            "ci_lower_95": ci_lower_95,
            "ci_upper_95": ci_upper_95,
            "aic": float(poisson_results.aic),
            "mae": mae,
            "rmse": rmse,
        }
    ]
)
model_summary.to_csv("poisson_cumulative_model_summary.csv", index=False)


def latex_value(value):
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, str):
        return value
    if np.isfinite(value):
        return f"{value:,.2f}"
    return "NA"


latex_rows = [
    ("AIC", poisson_results.aic),
    ("Training MAE", mae),
    ("Training RMSE", rmse),
    ("Partial 2026 observed date", partial_2026_observed_date),
    (
        "Partial 2026 observed cumulative cases",
        partial_2026_observed_cumulative_cases,
    ),
    ("Full-year 2026 forecast date", full_year_2026_forecast_date),
    ("Raw forecast cumulative cases", raw_forecast_cumulative_cases),
    ("Adjusted forecast cumulative cases", adjusted_forecast_cumulative_cases),
    ("Raw forecast 2026 cases implied", raw_forecast_2026_cases_implied),
    ("Adjusted forecast 2026 cases implied", adjusted_forecast_2026_cases_implied),
    ("95\\% CI lower", ci_lower_95),
    ("95\\% CI upper", ci_upper_95),
]

latex_table = "\n".join(
    [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Poisson model summary and 2026 forecast.}",
        "\\label{tab:poisson_summary}",
        "\\small",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Quantity & Value \\\\",
        "\\midrule",
        *[f"{quantity} & {latex_value(value)} \\\\" for quantity, value in latex_rows],
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
)

with open("poisson_cumulative_model_summary.tex", "w") as file:
    file.write(latex_table + "\n")

print("\nPoisson model summary:")
print(model_summary)
print("\nLaTeX table:")
print(latex_table)
print("\nSaved poisson_cumulative_model_summary.csv")
print("Saved poisson_cumulative_model_summary.tex")

# ----------------------------------------------------
# Plot observed, fitted, and 95% CI
# ----------------------------------------------------
plt.figure(figsize=(12, 6))

plt.plot(
    train_features.index,
    train_features["cumulative_cases"],
    marker="o",
    label="Observed cumulative cases",
)

plt.plot(
    prediction_plot.index,
    prediction_plot["fitted_cumulative_cases"],
    marker="o",
    linestyle="--",
    label="Poisson fitted and 2026 forecast",
)

plt.fill_between(
    prediction_plot.index,
    prediction_plot["ci_lower_95"],
    prediction_plot["ci_upper_95"],
    alpha=0.25,
    label="95% CI",
)

plt.xlabel("Month-Year")
plt.ylabel("Cumulative measles cases")
plt.title("U.S. Cumulative Measles Cases - Poisson Forecast with 95% CI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("poisson_cumulative_measles_fit_ci.png", dpi=300)
plt.show()

# ----------------------------------------------------
# Save fitted values
# ----------------------------------------------------
fit_output = train_features[
    [
        "count_year",
        "cases",
        "cumulative_cases",
        "fitted_cumulative_cases",
        "ci_lower_95",
        "ci_upper_95",
    ]
].copy()
fit_output.to_csv("poisson_cumulative_fitted_values.csv")
