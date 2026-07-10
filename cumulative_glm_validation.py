from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data-table (1).csv"
FORECAST_COUNT_YEAR = 2026
TRAIN_START_YEAR = 1996
ROLLING_START_YEAR = 2016
EXOG_COLS = [
    "time",
    "outbreak_2014",
    "outbreak_2019",
    "outbreak_2025",
]


def mean_absolute_error(observed, predicted):
    return float(np.mean(np.abs(np.asarray(observed) - np.asarray(predicted))))


def root_mean_squared_error(observed, predicted):
    error = np.asarray(observed) - np.asarray(predicted)
    return float(np.sqrt(np.mean(error**2)))


def load_data():
    data = pd.read_csv(DATA_PATH)
    if data.columns[0].lower().startswith("unnamed") or data.columns[0] not in {
        "year",
        "cases",
    }:
        data = data.drop(columns=data.columns[0])

    data = data[["year", "cases"]].dropna().copy()
    data["year"] = data["year"].astype(int)
    data["cases"] = data["cases"].astype(float)
    data = data[data["year"] <= FORECAST_COUNT_YEAR].sort_values("year")
    history = data[data["year"] < FORECAST_COUNT_YEAR].copy()
    history["cumulative_cases"] = history["cases"].cumsum()
    history = history.rename(columns={"year": "count_year"})
    return history


def create_features(df, base_year):
    df = df.copy()
    years = df["count_year"]
    df["time"] = years - base_year
    df["outbreak_2014"] = (years >= 2014).astype(int)
    df["outbreak_2019"] = (years >= 2019).astype(int)
    df["outbreak_2025"] = (years >= 2025).astype(int)
    return df


def fit_model(model_key, train_df, base_year):
    features = create_features(train_df, base_year)
    y_train = features["cumulative_cases"].astype(float)
    x_train = sm.add_constant(
        features[EXOG_COLS].astype(float),
        has_constant="add",
    )
    if model_key == "poisson":
        model = sm.GLM(y_train, x_train, family=sm.families.Poisson())
        result = model.fit()
    elif model_key == "quasi_poisson":
        model = sm.GLM(y_train, x_train, family=sm.families.Poisson())
        result = model.fit(scale="X2")
    elif model_key == "negative_binomial":
        model = sm.NegativeBinomial(y_train, x_train, loglike_method="nb2")
        try:
            result = model.fit(method="nm", maxiter=10000, disp=False)
        except np.linalg.LinAlgError:
            # Some validation folds are nearly singular with step indicators.
            # Use GLM negative binomial for prediction-only validation fallback.
            glm_model = sm.GLM(
                y_train,
                x_train,
                family=sm.families.NegativeBinomial(alpha=1.0),
            )
            result = glm_model.fit()
    else:
        raise ValueError(f"Unknown model_key: {model_key}")
    return result, x_train.columns


def predict_cumulative(result, x_columns, base_year, count_year):
    pred_df = pd.DataFrame({"count_year": [count_year]})
    pred_features = create_features(pred_df, base_year)
    x_pred = sm.add_constant(
        pred_features[EXOG_COLS].astype(float),
        has_constant="add",
    )
    x_pred = x_pred.reindex(columns=x_columns, fill_value=0.0)
    prediction = result.predict(x_pred)
    return float(np.asarray(prediction, dtype=float)[0])


def annual_prediction_from_cumulative(history, predicted_cumulative, count_year):
    prior_cumulative = float(
        history.loc[history["count_year"] < count_year, "cases"].sum()
    )
    return max(0.0, predicted_cumulative - prior_cumulative)


def summarize_validation(results):
    return pd.DataFrame(
        [
            {
                "validation": "rolling_one_year_ahead",
                "n_scored": int(len(results)),
                "mae": mean_absolute_error(
                    results["observed_cases"], results["predicted_cases"]
                ),
                "rmse": root_mean_squared_error(
                    results["observed_cases"], results["predicted_cases"]
                ),
            }
        ]
    )


def update_model_summary(slug, validation_summary):
    summary_path = ROOT / f"{slug}_model_summary.csv"
    if not summary_path.exists():
        return
    model_summary = pd.read_csv(summary_path)
    stale_validation_cols = [
        col
        for col in model_summary.columns
        if (
            col.startswith("rolling_one_year_ahead_")
            or col.startswith("leave_one_out_missing_response_")
        )
    ]
    model_summary = model_summary.drop(columns=stale_validation_cols)
    for _, row in validation_summary.iterrows():
        prefix = row["validation"]
        model_summary[f"{prefix}_n_scored"] = row["n_scored"]
        model_summary[f"{prefix}_mae"] = row["mae"]
        model_summary[f"{prefix}_rmse"] = row["rmse"]
    model_summary.to_csv(summary_path, index=False)


def append_latex_validation_rows(slug, validation_summary):
    tex_path = ROOT / f"{slug}_model_summary.tex"
    if not tex_path.exists():
        return
    text = tex_path.read_text(encoding="ascii")
    text = "\n".join(
        line
        for line in text.splitlines()
        if not (
            "Rolling validation" in line
            or "Leave-one-out validation" in line
            or "outbreak-year validation" in line
        )
    )

    lookup = {
        row["validation"]: row for _, row in validation_summary.iterrows()
    }
    rows = []
    for key, label in [
        ("rolling_one_year_ahead", "Rolling validation"),
    ]:
        row = lookup.get(key)
        if row is None:
            continue
        rows.extend(
            [
                f"{label} n & {int(row['n_scored'])} \\\\",
                f"{label} MAE & {row['mae']:,.2f} \\\\",
                f"{label} RMSE & {row['rmse']:,.2f} \\\\",
            ]
        )

    marker = "\\bottomrule"
    replacement = "\n".join(rows + [marker])
    text = text.replace(marker, replacement)
    tex_path.write_text(text, encoding="ascii")


def validation_results_for_model(model_key):
    history = load_data()
    train = history[history["count_year"] >= TRAIN_START_YEAR].copy()
    base_year = int(train["count_year"].min())
    rows = []

    for test_year in range(ROLLING_START_YEAR, FORECAST_COUNT_YEAR):
        fold_train = train[train["count_year"] < test_year].copy()
        result, x_columns = fit_model(model_key, fold_train, base_year)
        predicted_cumulative = predict_cumulative(
            result, x_columns, base_year, test_year
        )
        predicted_cases = annual_prediction_from_cumulative(
            history, predicted_cumulative, test_year
        )
        observed_cases = float(
            train.loc[train["count_year"] == test_year, "cases"].iloc[0]
        )
        rows.append(
            {
                "validation": "rolling_one_year_ahead",
                "count_year": test_year,
                "observed_cases": observed_cases,
                "predicted_cases": predicted_cases,
                "absolute_error": abs(observed_cases - predicted_cases),
                "test_set_start_year": ROLLING_START_YEAR,
            }
        )

    return pd.DataFrame(rows)


def main():
    settings = {
        "poisson": "poisson_cumulative",
        "quasi_poisson": "quasi_poisson_cumulative",
        "negative_binomial": "negative_binomial_cumulative",
    }
    for model_key, slug in settings.items():
        results = validation_results_for_model(model_key)
        summary = summarize_validation(results)
        results.to_csv(ROOT / f"{slug}_validation_results.csv", index=False)
        summary.to_csv(ROOT / f"{slug}_validation_summary.csv", index=False)
        update_model_summary(slug, summary)
        append_latex_validation_rows(slug, summary)
        print(f"\n{slug} validation summary:")
        print(summary)


if __name__ == "__main__":
    main()
