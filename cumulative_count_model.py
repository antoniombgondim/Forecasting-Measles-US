from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data-table (1).csv"
FORECAST_COUNT_YEAR = 2026
PARTIAL_2026_OBSERVED_DATE = pd.Timestamp("2026-05-01")
FULL_YEAR_2026_FORECAST_DATE = pd.Timestamp("2026-12-31")
EXOG_COLS = [
    "time",
    "outbreak_2014",
    "outbreak_2019",
    "outbreak_2025",
]
VALIDATION_NOTE = (
    "Out-of-sample validation is limited by reliance on a single held-out "
    "2026 observation; model performance should be interpreted cautiously "
    "and supplemented with leave-one-out cross-validation or additional "
    "hold-out exercises for outbreak years such as 2019 and 2025."
)


def create_features(df, base_year):
    df = df.copy()
    years = df["count_year"]
    df["time"] = years - base_year
    # Match poisson_cumulative_measle.py exactly: these are step indicators.
    df["outbreak_2014"] = (years >= 2014).astype(int)
    df["outbreak_2019"] = (years >= 2019).astype(int)
    df["outbreak_2025"] = (years >= 2025).astype(int)
    return df


def mean_absolute_error(observed, predicted):
    return float(np.mean(np.abs(np.asarray(observed) - np.asarray(predicted))))


def root_mean_squared_error(observed, predicted):
    error = np.asarray(observed) - np.asarray(predicted)
    return float(np.sqrt(np.mean(error**2)))


def prepare_data():
    data = pd.read_csv(DATA_PATH)
    if data.columns[0].lower().startswith("unnamed") or data.columns[0] not in {
        "year",
        "cases",
    }:
        data = data.drop(columns=data.columns[0])

    data = data[["year", "cases"]].dropna()
    data["year"] = data["year"].astype(int)
    data["cases"] = data["cases"].astype(int)
    data = data[data["year"] <= FORECAST_COUNT_YEAR].sort_values("year")

    observed_2026 = data.loc[data["year"] == FORECAST_COUNT_YEAR, "cases"]
    observed_2026_cases = (
        float(observed_2026.iloc[0]) if not observed_2026.empty else np.nan
    )

    history = data[data["year"] < FORECAST_COUNT_YEAR].copy()
    history["cumulative_cases"] = history["cases"].cumsum()
    history["month_year"] = pd.to_datetime(
        {
            "year": history["year"] + 1,
            "month": 1,
            "day": 1,
        }
    )
    history = (
        history.rename(columns={"year": "count_year"})
        .set_index("month_year")
        .rename_axis("mm_yr")
    )
    train = history[history["count_year"] >= 1996].copy()
    return history, train, observed_2026_cases


def prediction_frame(results, exog, result_kind):
    summary = results.get_prediction(exog).summary_frame(alpha=0.05)
    if result_kind == "glm":
        return summary.rename(
            columns={
                "mean": "prediction",
                "mean_ci_lower": "ci_lower_95",
                "mean_ci_upper": "ci_upper_95",
            }
        )[["prediction", "ci_lower_95", "ci_upper_95"]]
    return summary.rename(
        columns={
            "predicted": "prediction",
            "ci_lower": "ci_lower_95",
            "ci_upper": "ci_upper_95",
        }
    )[["prediction", "ci_lower_95", "ci_upper_95"]]


def parameter_summary(results, result_kind):
    params = pd.Series(results.params)
    conf = pd.DataFrame(
        np.asarray(results.conf_int()),
        index=params.index,
        columns=["ci_lower_95", "ci_upper_95"],
    )
    summary = pd.DataFrame(
        {
            "coef": params,
            "std_error": pd.Series(results.bse, index=params.index),
            "z_value": pd.Series(results.tvalues, index=params.index),
            "p_value": pd.Series(results.pvalues, index=params.index),
        }
    ).join(conf)

    summary["IRR"] = np.nan
    summary["IRR_ci_lower_95"] = np.nan
    summary["IRR_ci_upper_95"] = np.nan
    coefficient_mask = (
        summary.index != "alpha"
        if result_kind == "discrete_nb"
        else np.ones(len(summary), dtype=bool)
    )
    summary.loc[coefficient_mask, "IRR"] = np.exp(
        summary.loc[coefficient_mask, "coef"]
    )
    summary.loc[coefficient_mask, "IRR_ci_lower_95"] = np.exp(
        summary.loc[coefficient_mask, "ci_lower_95"]
    )
    summary.loc[coefficient_mask, "IRR_ci_upper_95"] = np.exp(
        summary.loc[coefficient_mask, "ci_upper_95"]
    )
    return summary


def latex_value(value):
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, str):
        return value
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:,.2f}"


def run_cumulative_count_model(model_key):
    settings = {
        "quasi_poisson": {
            "label": "Quasi-Poisson cumulative",
            "slug": "quasi_poisson_cumulative",
            "caption": "Quasi-Poisson model summary and 2026 forecast.",
            "table_label": "tab:quasi_poisson_summary",
            "result_kind": "glm",
        },
        "negative_binomial": {
            "label": "Negative binomial cumulative",
            "slug": "negative_binomial_cumulative",
            "caption": "Negative binomial model summary and 2026 forecast.",
            "table_label": "tab:negative_binomial_summary",
            "result_kind": "discrete_nb",
        },
    }[model_key]

    history, train, observed_2026_cases = prepare_data()
    base_year = int(train["count_year"].min())
    train_features = create_features(train, base_year)
    y_train = train_features["cumulative_cases"].astype(float)
    x_train = sm.add_constant(
        train_features[EXOG_COLS].astype(float),
        has_constant="add",
    )

    if model_key == "quasi_poisson":
        model = sm.GLM(y_train, x_train, family=sm.families.Poisson())
        results = model.fit(scale="X2")
        aic = np.nan
        dispersion = float(results.scale)
        converged = bool(results.converged)
    else:
        model = sm.NegativeBinomial(
            y_train,
            x_train,
            loglike_method="nb2",
        )
        # Nelder-Mead is stable when alpha is estimated near its Poisson boundary.
        results = model.fit(method="nm", maxiter=10000, disp=False)
        aic = float(results.aic)
        dispersion = float(results.params["alpha"])
        converged = bool(results.mle_retvals.get("converged", False))

    forecast_index = pd.DatetimeIndex(
        [PARTIAL_2026_OBSERVED_DATE, FULL_YEAR_2026_FORECAST_DATE],
        name="mm_yr",
    )
    forecast = pd.DataFrame(
        {
            "count_year": [FORECAST_COUNT_YEAR] * len(forecast_index),
            "cases": [np.nan] * len(forecast_index),
            "cumulative_cases": [np.nan] * len(forecast_index),
        },
        index=forecast_index,
    )
    forecast_features = create_features(forecast, base_year)
    x_forecast = sm.add_constant(
        forecast_features[EXOG_COLS].astype(float),
        has_constant="add",
    )[x_train.columns]

    forecast_predictions = prediction_frame(
        results,
        x_forecast,
        settings["result_kind"],
    )
    forecast_features["forecast_cumulative_cases"] = np.asarray(
        forecast_predictions["prediction"]
    )
    forecast_features["ci_lower_95"] = np.asarray(
        forecast_predictions["ci_lower_95"]
    )
    forecast_features["ci_upper_95"] = np.asarray(
        forecast_predictions["ci_upper_95"]
    )

    train_predictions = prediction_frame(
        results,
        x_train,
        settings["result_kind"],
    )
    train_features["fitted_cumulative_cases"] = np.asarray(
        train_predictions["prediction"]
    )
    train_features["ci_lower_95"] = np.asarray(train_predictions["ci_lower_95"])
    train_features["ci_upper_95"] = np.asarray(train_predictions["ci_upper_95"])

    mae = mean_absolute_error(
        train_features["cumulative_cases"],
        train_features["fitted_cumulative_cases"],
    )
    rmse = root_mean_squared_error(
        train_features["cumulative_cases"],
        train_features["fitted_cumulative_cases"],
    )

    cumulative_through_2025 = float(history["cumulative_cases"].iloc[-1])
    partial_2026_cumulative = (
        cumulative_through_2025 + observed_2026_cases
        if np.isfinite(observed_2026_cases)
        else np.nan
    )
    full_year_forecast = forecast_features.loc[FULL_YEAR_2026_FORECAST_DATE]
    raw_forecast_cumulative = float(
        full_year_forecast["forecast_cumulative_cases"]
    )
    adjusted_forecast_cumulative = (
        max(raw_forecast_cumulative, partial_2026_cumulative)
        if np.isfinite(partial_2026_cumulative)
        else raw_forecast_cumulative
    )
    raw_forecast_cases = raw_forecast_cumulative - cumulative_through_2025
    adjusted_forecast_cases = (
        adjusted_forecast_cumulative - cumulative_through_2025
    )
    ci_lower = float(full_year_forecast["ci_lower_95"])
    ci_upper = float(
        max(full_year_forecast["ci_upper_95"], adjusted_forecast_cumulative)
    )

    model_summary = pd.DataFrame(
        [
            {
                "model": settings["label"],
                "exogenous_variables": ", ".join(EXOG_COLS),
                "forecast_count_year": FORECAST_COUNT_YEAR,
                "partial_2026_observed_date": (
                    PARTIAL_2026_OBSERVED_DATE.date().isoformat()
                ),
                "partial_2026_observed_cumulative_cases": partial_2026_cumulative,
                "full_year_2026_forecast_date": (
                    FULL_YEAR_2026_FORECAST_DATE.date().isoformat()
                ),
                "raw_forecast_cumulative_cases": raw_forecast_cumulative,
                "forecast_cumulative_cases": adjusted_forecast_cumulative,
                "raw_forecast_2026_cases_implied": raw_forecast_cases,
                "forecast_2026_cases_implied": adjusted_forecast_cases,
                "ci_lower_95": ci_lower,
                "ci_upper_95": ci_upper,
                "dispersion": dispersion,
                "aic": aic,
                "converged": converged,
                "mae": mae,
                "rmse": rmse,
                "validation_note": VALIDATION_NOTE,
            }
        ]
    )

    param_summary = parameter_summary(results, settings["result_kind"])
    forecast_output = forecast_features[
        [
            "count_year",
            "forecast_cumulative_cases",
            "ci_lower_95",
            "ci_upper_95",
        ]
    ].copy()
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

    prediction_plot = pd.concat(
        [
            train_features[
                ["fitted_cumulative_cases", "ci_lower_95", "ci_upper_95"]
            ],
            forecast_features.rename(
                columns={"forecast_cumulative_cases": "fitted_cumulative_cases"}
            )[["fitted_cumulative_cases", "ci_lower_95", "ci_upper_95"]],
        ]
    ).sort_index()

    slug = settings["slug"]
    model_summary.to_csv(ROOT / f"{slug}_model_summary.csv", index=False)
    param_summary.to_csv(ROOT / f"{slug}_parameter_uncertainty.csv")
    forecast_output.to_csv(ROOT / f"{slug}_2026_forecast.csv")
    fit_output.to_csv(ROOT / f"{slug}_fitted_values.csv")

    latex_rows = [
        ("AIC", aic),
        ("Dispersion", dispersion),
        ("Converged", str(converged)),
        ("Training MAE", mae),
        ("Training RMSE", rmse),
        ("Validation note", VALIDATION_NOTE),
        ("Partial 2026 observed date", PARTIAL_2026_OBSERVED_DATE),
        ("Partial 2026 observed cumulative cases", partial_2026_cumulative),
        ("Full-year 2026 forecast date", FULL_YEAR_2026_FORECAST_DATE),
        ("Raw forecast cumulative cases", raw_forecast_cumulative),
        ("Adjusted forecast cumulative cases", adjusted_forecast_cumulative),
        ("Raw forecast 2026 cases implied", raw_forecast_cases),
        ("Adjusted forecast 2026 cases implied", adjusted_forecast_cases),
        ("95\\% CI lower", ci_lower),
        ("95\\% CI upper", ci_upper),
    ]
    latex_table = "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            f"\\caption{{{settings['caption']}}}",
            f"\\label{{{settings['table_label']}}}",
            "\\small",
            "\\begin{tabular}{p{0.34\\linewidth}p{0.56\\linewidth}}",
            "\\toprule",
            "Quantity & Value \\\\",
            "\\midrule",
            *[
                f"{quantity} & {latex_value(value)} \\\\"
                for quantity, value in latex_rows
            ],
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    (ROOT / f"{slug}_model_summary.tex").write_text(
        latex_table + "\n",
        encoding="ascii",
    )

    plt.figure(figsize=(12, 6))
    plt.plot(
        train_features.index,
        train_features["cumulative_cases"],
        marker="o",
        color="black",
        label="Observed cumulative cases",
    )
    plt.plot(
        prediction_plot.index,
        prediction_plot["fitted_cumulative_cases"],
        marker="o",
        linestyle="--",
        label=f"{settings['label']} fitted and 2026 forecast",
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
    plt.title(f"U.S. Cumulative Measles Cases - {settings['label']}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ROOT / f"{slug}_measles_fit_ci.png", dpi=300)
    plt.close()

    print(results.summary())
    print("\nModel summary:")
    print(model_summary)
    print("\nParameter uncertainty:")
    print(param_summary)
    print(f"\nSaved {slug} outputs.")
