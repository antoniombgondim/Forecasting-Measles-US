from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data-table (1).csv"
TRAIN_START_YEAR = 1996
FORECAST_COUNT_YEAR = 2026
PARTIAL_2026_DATE = pd.Timestamp("2026-05-01")
FULL_YEAR_2026_FORECAST_DATE = pd.Timestamp("2026-12-31")
OUTBREAK_YEARS = {2014,2025}
CALIBRATION_START_YEAR = 2016
N_CI_SIMULATIONS = 20_000
RANDOM_SEED = 20260608
AR_ORDER_CANDIDATES = range(0, 5)
ARMA_P_CANDIDATES = range(0, 5)
ARMA_Q_CANDIDATES = range(0, 5)
AR_P_VALUE_ALPHA = 0.05
ARMA_P_VALUE_ALPHA = 0.05
EXOG_COLS = ["time", "major_outbreak_year"]
SPECIFICATION_NOTE = (
    "The revised ARIMAX grid-search workflow compares both ARIMAX(k,0,0) and "
    "ARIMAX(n,0,m) specifications, retaining d=0, an explicit linear time "
    "term, and a one-year major-outbreak pulse. The primary reported "
    "forecast uses the selected AR-only model to avoid weakly identified MA "
    "terms, while the broader ARMA search is reported as a sensitivity check."
)
VALIDATION_NOTE = (
    "Because the 2026 forecast is a single held-out observation, it is "
    "supplemented here with rolling one-year-ahead validation for 2016-2025, "
    "leave-one-out validation across the training years, and explicit "
    "outbreak-year holdout summaries for 2019 and 2025."
)


class FixedOrderArimax:
    def __init__(self, results, order):
        self.results = results
        self.order = order

    @property
    def aic(self):
        return float(self.results.aic)

    def fitted_log(self, start, end, exog):
        pred = self.results.get_prediction(
            start=start,
            end=end,
            exog=exog.iloc[start : end + 1],
        )
        return np.asarray(pred.predicted_mean, dtype=float)

    def forecast_log(self, steps, exog, alpha=0.05):
        forecast = self.results.get_forecast(steps=steps, exog=exog)
        return (
            np.asarray(forecast.predicted_mean, dtype=float),
            np.asarray(forecast.conf_int(alpha=alpha), dtype=float),
        )


def load_cumulative_data(path):
    data = pd.read_csv(path)
    if data.columns[0].lower().startswith("unnamed") or data.columns[0] not in {
        "year",
        "cases",
    }:
        data = data.drop(columns=data.columns[0])

    data = data[["year", "cases"]].dropna()
    data["year"] = data["year"].astype(int)
    data["cases"] = data["cases"].astype(float)
    data = data[data["year"] <= FORECAST_COUNT_YEAR].sort_values("year")
    data["cumulative_cases"] = data["cases"].cumsum()
    data["date"] = pd.to_datetime(
        {"year": data["year"] + 1, "month": 1, "day": 1}
    )
    data.loc[data["year"] == FORECAST_COUNT_YEAR, "date"] = PARTIAL_2026_DATE
    return (
        data.rename(columns={"year": "count_year"})
        .set_index("date")
        .rename_axis("date")
    )


def create_features(df, base_year):
    df = df.copy()
    years = df["count_year"].astype(int)
    df["time"] = years - base_year
    df["major_outbreak_year"] = years.isin(OUTBREAK_YEARS).astype(int)
    return df


def fit_arimax(y_log, exog, order):
    model = SARIMAX(
        y_log,
        exog=exog,
        order=order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
        missing="drop",
    )
    return FixedOrderArimax(model.fit(disp=False), order)


def residual_diagnostics(results, index):
    residuals = pd.Series(
        np.asarray(results.resid, dtype=float),
        index=index[-len(np.asarray(results.resid)) :],
        name="arimax_log_scale_residual",
    ).replace([np.inf, -np.inf], np.nan).dropna()
    max_lag = min(10, max(1, len(residuals) // 2 - 1))
    diagnostic_lags = sorted(
        {lag for lag in [1, 5, 10, max_lag] if 0 < lag <= max_lag}
    )
    ljung_box = acorr_ljungbox(residuals, lags=diagnostic_lags, return_df=True)
    ljung_box.index.name = "lag"
    ljung_box_result = (
        "no residual autocorrelation detected at alpha=0.05"
        if (ljung_box["lb_pvalue"] > 0.05).all()
        else "residual autocorrelation detected at alpha=0.05"
    )
    return residuals, max_lag, diagnostic_lags, ljung_box, ljung_box_result


def ar_p_value_summary(results):
    pvalues = pd.Series(results.pvalues, index=results.param_names)
    ar_pvalues = pvalues[pvalues.index.str.startswith("ar.L")]
    if ar_pvalues.empty:
        return np.nan, np.nan, np.nan, False
    return (
        float(ar_pvalues.min()),
        float(ar_pvalues.max()),
        "; ".join(f"{name}={value:.4g}" for name, value in ar_pvalues.items()),
        bool((ar_pvalues < AR_P_VALUE_ALPHA).all()),
    )


def arma_p_value_summary(results):
    pvalues = pd.Series(results.pvalues, index=results.param_names)
    arma_mask = pvalues.index.str.startswith("ar.L") | pvalues.index.str.startswith(
        "ma.L"
    )
    arma_pvalues = pvalues[arma_mask]
    if arma_pvalues.empty:
        return np.nan, np.nan, np.nan, False
    return (
        float(arma_pvalues.min()),
        float(arma_pvalues.max()),
        "; ".join(f"{name}={value:.4g}" for name, value in arma_pvalues.items()),
        bool((arma_pvalues < ARMA_P_VALUE_ALPHA).all()),
    )


def select_ar_order(y_log, exog, index):
    rows = []
    fitted_models = {}
    for k in AR_ORDER_CANDIDATES:
        order = (k, 0, 0)
        try:
            model = fit_arimax(y_log, exog, order)
            _, _, _, ljung_box, ljung_box_result = residual_diagnostics(
                model.results, index
            )
            min_ar_p, max_ar_p, ar_pvalues, all_ar_p_below_alpha = (
                ar_p_value_summary(model.results)
            )
            row = {
                "order": str(order),
                "k": k,
                "aic": model.aic,
                "min_ar_p_value": min_ar_p,
                "max_ar_p_value": max_ar_p,
                "ar_p_values": ar_pvalues,
                "all_ar_p_values_below_0_05": all_ar_p_below_alpha,
                "ljung_box_min_p_value": float(ljung_box["lb_pvalue"].min()),
                "ljung_box_result": ljung_box_result,
                "converged": bool(model.results.mle_retvals.get("converged", True)),
                "selection_error": "",
            }
            fitted_models[k] = model
        except Exception as exc:
            row = {
                "order": str(order),
                "k": k,
                "aic": np.nan,
                "min_ar_p_value": np.nan,
                "max_ar_p_value": np.nan,
                "ar_p_values": np.nan,
                "all_ar_p_values_below_0_05": False,
                "ljung_box_min_p_value": np.nan,
                "ljung_box_result": "not fitted",
                "converged": False,
                "selection_error": str(exc),
            }
        rows.append(row)

    candidates = pd.DataFrame(rows)
    fitted = candidates[np.isfinite(candidates["aic"])].copy()
    if fitted.empty:
        raise RuntimeError("No ARIMAX(k,0,0) candidate model could be fitted.")

    preferred = fitted[
        (fitted["k"] > 0)
        & fitted["all_ar_p_values_below_0_05"]
        & (fitted["ljung_box_min_p_value"] > 0.05)
    ]
    selection_rule = (
        "lowest AIC among k>0 candidates with all AR p-values < 0.05 "
        "and Ljung-Box min p-value > 0.05"
    )
    if preferred.empty:
        preferred = fitted[fitted["ljung_box_min_p_value"] > 0.05]
        selection_rule = (
            "fallback: lowest AIC among candidates with Ljung-Box min p-value > 0.05"
        )
    if preferred.empty:
        preferred = fitted
        selection_rule = "fallback: lowest AIC among fitted candidates"

    selected_row = preferred.sort_values(["aic", "k"]).iloc[0]
    selected_k = int(selected_row["k"])
    candidates["selected"] = candidates["k"] == selected_k
    candidates["selection_rule"] = selection_rule
    return fitted_models[selected_k], candidates, selection_rule


def select_arma_order(y_log, exog, index):
    rows = []
    fitted_models = {}
    for p in ARMA_P_CANDIDATES:
        for q in ARMA_Q_CANDIDATES:
            order = (p, 0, q)
            try:
                model = fit_arimax(y_log, exog, order)
                _, _, _, ljung_box, ljung_box_result = residual_diagnostics(
                    model.results, index
                )
                (
                    min_arma_p,
                    max_arma_p,
                    arma_pvalues,
                    all_arma_p_below_alpha,
                ) = arma_p_value_summary(model.results)
                row = {
                    "order": str(order),
                    "n": p,
                    "m": q,
                    "aic": model.aic,
                    "min_arma_p_value": min_arma_p,
                    "max_arma_p_value": max_arma_p,
                    "arma_p_values": arma_pvalues,
                    "all_arma_p_values_below_0_05": all_arma_p_below_alpha,
                    "ljung_box_min_p_value": float(ljung_box["lb_pvalue"].min()),
                    "ljung_box_result": ljung_box_result,
                    "converged": bool(model.results.mle_retvals.get("converged", True)),
                    "selection_error": "",
                }
                fitted_models[(p, q)] = model
            except Exception as exc:
                row = {
                    "order": str(order),
                    "n": p,
                    "m": q,
                    "aic": np.nan,
                    "min_arma_p_value": np.nan,
                    "max_arma_p_value": np.nan,
                    "arma_p_values": np.nan,
                    "all_arma_p_values_below_0_05": False,
                    "ljung_box_min_p_value": np.nan,
                    "ljung_box_result": "not fitted",
                    "converged": False,
                    "selection_error": str(exc),
                }
            rows.append(row)

    candidates = pd.DataFrame(rows)
    fitted = candidates[np.isfinite(candidates["aic"])].copy()
    if fitted.empty:
        raise RuntimeError("No ARIMAX(n,0,m) candidate model could be fitted.")

    preferred = fitted[
        ((fitted["n"] > 0) | (fitted["m"] > 0))
        & fitted["all_arma_p_values_below_0_05"]
        & (fitted["ljung_box_min_p_value"] > 0.05)
    ]
    selection_rule = (
        "lowest AIC among non-white-noise ARMA candidates with all AR/MA "
        "p-values < 0.05 and Ljung-Box min p-value > 0.05"
    )
    if preferred.empty:
        preferred = fitted[fitted["ljung_box_min_p_value"] > 0.05]
        selection_rule = (
            "fallback: lowest AIC among candidates with Ljung-Box min p-value > 0.05"
        )
    if preferred.empty:
        preferred = fitted
        selection_rule = "fallback: lowest AIC among fitted candidates"

    selected_row = preferred.sort_values(["aic", "n", "m"]).iloc[0]
    selected_key = (int(selected_row["n"]), int(selected_row["m"]))
    candidates["selected"] = (candidates["n"] == selected_key[0]) & (
        candidates["m"] == selected_key[1]
    )
    candidates["selection_rule"] = selection_rule
    return fitted_models[selected_key], candidates, selection_rule


def case_forecast_from_log(log_forecast):
    return np.maximum(np.expm1(np.asarray(log_forecast, dtype=float)), 0.0)


def arma_burn_in(order):
    return max(int(order[0]), int(order[2]))


def score_errors(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    errors = observed[valid] - predicted[valid]
    return {
        "n_scored": int(valid.sum()),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }


def rolling_one_year_validation(train_features, base_year, order):
    rows = []
    for test_year in range(CALIBRATION_START_YEAR, FORECAST_COUNT_YEAR):
        fold_train = train_features[train_features["count_year"] < test_year]
        if len(fold_train) < 8:
            continue
        fold_exog = fold_train[EXOG_COLS].astype(float)
        fold_y_log = np.log1p(fold_train["cases"].astype(float))
        fold_model = fit_arimax(fold_y_log, fold_exog, order)

        test_frame = create_features(
            pd.DataFrame({"count_year": [test_year]}), base_year
        )
        forecast_log, _ = fold_model.forecast_log(
            steps=1, exog=test_frame[EXOG_COLS].astype(float)
        )
        predicted = float(case_forecast_from_log(forecast_log)[0])
        observed = float(
            train_features.loc[
                train_features["count_year"] == test_year, "cases"
            ].iloc[0]
        )
        rows.append(
            {
                "validation": "rolling_one_year_ahead",
                "count_year": int(test_year),
                "observed_cases": observed,
                "predicted_cases": predicted,
                "absolute_error": abs(observed - predicted),
                "is_outbreak_year": test_year in OUTBREAK_YEARS,
            }
        )
    return pd.DataFrame(rows)


def leave_one_out_validation(train_features, order):
    rows = []
    y_log_full = np.log1p(train_features["cases"].astype(float))
    x_full = train_features[EXOG_COLS].astype(float)
    for test_year in train_features["count_year"].astype(int):
        y_fold = y_log_full.copy()
        fold_index = train_features.index[train_features["count_year"] == test_year][0]
        y_fold.loc[fold_index] = np.nan
        fold_model = fit_arimax(y_fold, x_full, order)
        loc = train_features.index.get_loc(fold_index)
        predicted_log = fold_model.fitted_log(start=loc, end=loc, exog=x_full)[0]
        predicted = float(case_forecast_from_log([predicted_log])[0])
        observed = float(train_features.loc[fold_index, "cases"])
        rows.append(
            {
                "validation": "leave_one_out_missing_response",
                "count_year": int(test_year),
                "observed_cases": observed,
                "predicted_cases": predicted,
                "absolute_error": abs(observed - predicted),
                "is_outbreak_year": test_year in OUTBREAK_YEARS,
            }
        )
    return pd.DataFrame(rows)


def validation_summary(validation_results):
    grouped = validation_results.groupby("validation", dropna=False)
    rows = []
    for name, frame in grouped:
        metrics = score_errors(frame["observed_cases"], frame["predicted_cases"])
        rows.append({"validation": name, **metrics})

    outbreak_frame = validation_results[
        validation_results["count_year"].isin([2019, 2025])
    ].copy()
    for name, frame in outbreak_frame.groupby("validation", dropna=False):
        metrics = score_errors(frame["observed_cases"], frame["predicted_cases"])
        rows.append({"validation": f"{name}_outbreak_2019_2025", **metrics})

    return pd.DataFrame(rows)


def parameter_uncertainty(results):
    params = pd.Series(results.params, index=results.param_names)
    conf = pd.DataFrame(
        np.asarray(results.conf_int(alpha=0.05)),
        index=results.param_names,
        columns=["ci_lower_95_log_scale", "ci_upper_95_log_scale"],
    )
    summary = pd.DataFrame(
        {
            "coef_log_scale": params,
            "std_error": pd.Series(results.bse, index=results.param_names),
            "z_value": pd.Series(results.zvalues, index=results.param_names),
            "p_value": pd.Series(results.pvalues, index=results.param_names),
        }
    ).join(conf)
    summary["multiplicative_effect_on_one_plus_cases"] = np.exp(
        summary["coef_log_scale"]
    )
    summary["multiplicative_ci_lower_95"] = np.exp(
        summary["ci_lower_95_log_scale"]
    )
    summary["multiplicative_ci_upper_95"] = np.exp(
        summary["ci_upper_95_log_scale"]
    )
    return summary


def main():
    all_data = load_cumulative_data(DATA_PATH)
    train = all_data[
        (all_data["count_year"] >= TRAIN_START_YEAR)
        & (all_data["count_year"] < FORECAST_COUNT_YEAR)
    ].copy()
    partial_2026 = all_data[all_data["count_year"] == FORECAST_COUNT_YEAR]
    base_year = int(train["count_year"].min())

    train_features = create_features(train, base_year)
    x_train = train_features[EXOG_COLS].astype(float)
    annual_cases = train_features["cases"].astype(float)
    annual_cases_log = np.log1p(annual_cases)
    y_cumulative = train_features["cumulative_cases"].astype(float)

    model, ar_order_selection, selection_rule = select_ar_order(
        annual_cases_log, x_train, train_features.index
    )
    arma_model, arma_order_selection, arma_selection_rule = select_arma_order(
        annual_cases_log, x_train, train_features.index
    )
    selected_order_summary = pd.concat(
        [
            ar_order_selection.loc[ar_order_selection["selected"]].assign(
                model_family="best_ar_k",
                order_selection_rule=selection_rule,
            ),
            arma_order_selection.loc[arma_order_selection["selected"]].assign(
                model_family="best_arma_n_m",
                order_selection_rule=arma_selection_rule,
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    fitted_cases = pd.Series(
        case_forecast_from_log(
            model.fitted_log(start=0, end=len(train_features) - 1, exog=x_train)
        ),
        index=train_features.index,
        name="arimax_fitted_annual_cases",
    )
    burn_in = arma_burn_in(model.order)
    if burn_in > 0:
        fitted_cases.iloc[:burn_in] = annual_cases.iloc[:burn_in]

    cumulative_before_training = float(y_cumulative.iloc[0] - annual_cases.iloc[0])
    fitted_cumulative = cumulative_before_training + fitted_cases.cumsum()
    residual_cases = np.asarray(annual_cases - fitted_cases, dtype=float)
    residual_cases = residual_cases - residual_cases.mean()
    rng = np.random.default_rng(RANDOM_SEED)
    simulated_residuals = rng.choice(
        residual_cases, size=(N_CI_SIMULATIONS, len(fitted_cases)), replace=True
    )
    simulated_cases = np.maximum(
        np.asarray(fitted_cases, dtype=float)[None, :] + simulated_residuals, 0.0
    )
    simulated_cumulative = cumulative_before_training + np.cumsum(
        simulated_cases, axis=1
    )
    fitted_ci_lower, fitted_ci_upper = np.quantile(
        simulated_cumulative, [0.025, 0.975], axis=0
    )

    future = create_features(
        pd.DataFrame(
            {"count_year": [FORECAST_COUNT_YEAR]},
            index=[FULL_YEAR_2026_FORECAST_DATE],
        ),
        base_year,
    )
    forecast_log, forecast_ci_log = model.forecast_log(
        steps=1, exog=future[EXOG_COLS].astype(float)
    )
    raw_forecast_cases = float(case_forecast_from_log(forecast_log)[0])
    parametric_case_ci = case_forecast_from_log(forecast_ci_log)

    rolling_results = rolling_one_year_validation(
        train_features, base_year, model.order
    )
    loo_results = leave_one_out_validation(train_features, model.order)
    validation_results = pd.concat([rolling_results, loo_results], ignore_index=True)
    validation_metrics = validation_summary(validation_results)

    calibration_errors = rolling_results["absolute_error"].to_numpy(dtype=float)
    calibration_error_95 = float(np.quantile(calibration_errors, 0.95, method="higher"))

    cumulative_through_2025 = float(
        all_data.loc[all_data["count_year"] < FORECAST_COUNT_YEAR, "cases"].sum()
    )
    partial_2026_cumulative = (
        float(partial_2026["cumulative_cases"].iloc[0])
        if not partial_2026.empty
        else np.nan
    )
    raw_forecast_cumulative = cumulative_through_2025 + raw_forecast_cases
    forecast_cumulative = raw_forecast_cumulative
    if np.isfinite(partial_2026_cumulative):
        forecast_cumulative = max(forecast_cumulative, partial_2026_cumulative)
    forecast_cases = forecast_cumulative - cumulative_through_2025
    forecast_ci_cases = np.array(
        [
            max(0.0, raw_forecast_cases - calibration_error_95),
            raw_forecast_cases + calibration_error_95,
        ]
    )
    forecast_ci = cumulative_through_2025 + forecast_ci_cases
    if np.isfinite(partial_2026_cumulative):
        forecast_ci[0] = max(forecast_ci[0], partial_2026_cumulative)
        forecast_ci[1] = max(
            forecast_ci[1],
            forecast_cumulative + calibration_error_95,
        )

    fitted_values = pd.DataFrame(
        {
            "count_year": train_features["count_year"],
            "observed_cumulative_cases": y_cumulative,
            "observed_annual_cases": annual_cases,
            "arimax_fitted_annual_cases": fitted_cases,
            "arimax_fitted_cumulative_cases": fitted_cumulative,
            "ci_lower_95": fitted_ci_lower,
            "ci_upper_95": fitted_ci_upper,
        },
        index=train_features.index,
    )

    residuals, max_lag, diagnostic_lags, ljung_box, ljung_box_result = (
        residual_diagnostics(model.results, train_features.index)
    )

    fit_metrics = score_errors(
        fitted_values["observed_cumulative_cases"].iloc[burn_in:],
        fitted_values["arimax_fitted_cumulative_cases"].iloc[burn_in:],
    )
    model_summary = pd.DataFrame(
        [
            {
                "model": "Revised ARIMAX grid-search annual increments to cumulative",
                "exogenous_variables": ", ".join(EXOG_COLS),
                "forecast_count_year": FORECAST_COUNT_YEAR,
                "partial_2026_observed_date": PARTIAL_2026_DATE.date().isoformat(),
                "partial_2026_observed_cumulative_cases": partial_2026_cumulative,
                "full_year_2026_forecast_date": (
                    FULL_YEAR_2026_FORECAST_DATE.date().isoformat()
                ),
                "raw_forecast_cumulative_cases": raw_forecast_cumulative,
                "forecast_cumulative_cases": forecast_cumulative,
                "raw_forecast_2026_cases_implied": raw_forecast_cases,
                "forecast_2026_cases_implied": forecast_cases,
                "historical_interval_method": (
                    "case-scale residual simulation of cumulative paths"
                ),
                "forecast_interval_method": (
                    "rolling absolute-error calibration, 2016-2025"
                ),
                "calibration_error_95_cases": calibration_error_95,
                "parametric_ci_lower_95_cases": float(parametric_case_ci[0, 0]),
                "parametric_ci_upper_95_cases": float(parametric_case_ci[0, 1]),
                "ci_lower_95": float(forecast_ci[0]),
                "ci_upper_95": float(forecast_ci[1]),
                "aic": model.aic,
                "order": str(model.order),
                "best_ar_order": str(model.order),
                "best_arma_order": str(arma_model.order),
                "conditional_burn_in_years": burn_in,
                "ar_order_selection_rule": selection_rule,
                "arma_order_selection_rule": arma_selection_rule,
                "specification_note": SPECIFICATION_NOTE,
                "validation_note": VALIDATION_NOTE,
                "diagnostic_lags": "; ".join(map(str, diagnostic_lags)),
                "ljung_box_min_p_value": float(ljung_box["lb_pvalue"].min()),
                "ljung_box_result": ljung_box_result,
                **fit_metrics,
            }
        ]
    )

    fitted_values.to_csv(ROOT / "revised_autoarimax_fitted_values.csv")
    ar_order_selection.to_csv(
        ROOT / "revised_autoarimax_ar_order_selection.csv", index=False
    )
    arma_order_selection.to_csv(
        ROOT / "revised_autoarimax_arma_order_selection.csv", index=False
    )
    selected_order_summary.to_csv(
        ROOT / "revised_autoarimax_selected_order_summary.csv", index=False
    )
    parameter_uncertainty(model.results).to_csv(
        ROOT / "revised_autoarimax_parameter_uncertainty.csv",
        index_label="parameter",
    )
    ljung_box.to_csv(ROOT / "revised_autoarimax_ljung_box.csv")
    model_summary.to_csv(ROOT / "revised_autoarimax_model_summary.csv", index=False)
    validation_results.to_csv(
        ROOT / "revised_autoarimax_validation_results.csv", index=False
    )
    validation_metrics.to_csv(
        ROOT / "revised_autoarimax_validation_summary.csv", index=False
    )

    historical_complete = all_data[all_data["count_year"] < FORECAST_COUNT_YEAR]
    model_curve = pd.concat(
        [
            fitted_values["arimax_fitted_cumulative_cases"],
            pd.Series(
                [partial_2026_cumulative, forecast_cumulative],
                index=[PARTIAL_2026_DATE, FULL_YEAR_2026_FORECAST_DATE],
            ),
        ]
    ).sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        historical_complete.index,
        historical_complete["cumulative_cases"],
        marker="o",
        color="black",
        label="Observed cumulative cases",
    )
    ax.fill_between(
        fitted_values.index,
        fitted_values["ci_lower_95"],
        fitted_values["ci_upper_95"],
        color="tab:green",
        alpha=0.12,
        label="Historical simulated 95% prediction band",
    )
    ax.fill_between(
        [PARTIAL_2026_DATE, FULL_YEAR_2026_FORECAST_DATE],
        [partial_2026_cumulative, forecast_ci[0]],
        [partial_2026_cumulative, forecast_ci[1]],
        color="tab:green",
        alpha=0.16,
        label="Calibrated 95% prediction interval",
    )
    ax.plot(
        model_curve.index,
        model_curve,
        color="tab:green",
        linestyle="--",
        linewidth=2.3,
        label="Revised ARIMAX cumulative curve",
    )
    ax.scatter(
        [PARTIAL_2026_DATE],
        [partial_2026_cumulative],
        color="tab:red",
        marker="x",
        s=110,
        linewidths=2.5,
        label="Observed cumulative through May 1, 2026",
    )
    ax.set_xlim(fitted_values.index.min(), FULL_YEAR_2026_FORECAST_DATE)
    ax.set_ylim(bottom=75000)
    ax.set_title("Cumulative measles cases: revised ARIMAX grid-search fit and forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative cases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "revised_autoarimax_cumulative_forecast.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(residuals, lags=max_lag, ax=axes[0])
    plot_pacf(residuals, lags=max_lag, ax=axes[1], method="ywm")
    axes[0].set_title("Revised ARIMAX residual ACF")
    axes[1].set_title("Revised ARIMAX residual PACF")
    fig.tight_layout()
    fig.savefig(ROOT / "revised_autoarimax_residual_acf_pacf.png", dpi=300)
    plt.close(fig)

    print(model_summary.to_string(index=False))
    print()
    print(validation_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
