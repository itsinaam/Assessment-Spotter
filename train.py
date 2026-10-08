from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from score import save_december_chart, validate_december, validate_predictions


NUMERIC_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "day_index",
    "month",
    "day_of_week",
]
CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment"]
TARGET = "posted_rate"


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    dates = pd.to_datetime(result["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Input contains invalid dates")
    result["day_index"] = (dates - pd.Timestamp("2025-01-01")).dt.days
    result["month"] = dates.dt.month
    result["day_of_week"] = dates.dt.dayofweek
    return result


def category_levels(frame: pd.DataFrame) -> dict[str, list[str]]:
    levels = {}
    for column in CATEGORICAL_FEATURES:
        values = frame[column].fillna("__MISSING__").astype(str).unique().tolist()
        levels[column] = values + ["__UNKNOWN__"]
    return levels


def make_features(
    frame: pd.DataFrame, levels: dict[str, list[str]]
) -> pd.DataFrame:
    dated = add_date_features(frame)
    features = pd.DataFrame(index=dated.index)
    for column in NUMERIC_FEATURES:
        features[column] = dated[column] if column in dated else np.nan
    for column in CATEGORICAL_FEATURES:
        values = dated[column].fillna("__MISSING__").astype(str)
        values = values.where(values.isin(levels[column]), "__UNKNOWN__")
        features[column] = pd.Categorical(values, categories=levels[column])
    return features


def make_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=180,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=2.0,
        early_stopping=True,
        random_state=42,
        categorical_features="from_dtype",
    )


def train_and_evaluate(
    training: pd.DataFrame,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    dated = add_date_features(training)
    holdout_month = dated["date"].max()[:7]
    holdout_mask = dated["date"].str.startswith(holdout_month)
    fit_rows = dated.loc[~holdout_mask]
    holdout_rows = dated.loc[holdout_mask]
    if fit_rows.empty or holdout_rows.empty:
        raise ValueError("Could not create a chronological final-month holdout")

    levels = category_levels(fit_rows)
    model = make_model()
    model.fit(
        make_features(fit_rows, levels),
        fit_rows[TARGET].to_numpy() / fit_rows["distance"].to_numpy(),
    )
    predicted = model.predict(make_features(holdout_rows, levels))
    predicted = np.maximum(predicted, 0.01) * holdout_rows["distance"].to_numpy()
    actual = holdout_rows[TARGET].to_numpy()
    metrics = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "holdout_month": holdout_month,
        "holdout_rows": int(len(holdout_rows)),
        "training_rows": int(len(fit_rows)),
    }
    return metrics, dated


def write_report(
    metrics: dict[str, float | int | str], chart_path: Path, output: Path
) -> None:
    with PdfPages(output) as pdf:
        figure = plt.figure(figsize=(8.27, 11.69))
        axis = figure.add_axes((0.11, 0.08, 0.8, 0.84))
        axis.axis("off")
        report_text = (
            "FREIGHT RATE PREDICTION\n"
            "Model validation and December forecast\n\n"
            "VALIDATION DESIGN\n"
            f"Chronological split: {metrics['training_rows']:,} rows before "
            f"{metrics['holdout_month']} for fitting; {metrics['holdout_rows']:,} "
            f"rows from {metrics['holdout_month']} held out. This simulates forecasting "
            "future loads and avoids random-split leakage across time.\n\n"
            "MODEL AND DATA HANDLING\n"
            "HistGradientBoostingRegressor predicts posted rate per mile; predictions are "
            "multiplied by distance. Features include endpoints, coordinates, equipment, "
            "weight, market/quote signals, and calendar features. Numeric missing values "
            "are handled natively; unseen cities map to an explicit unknown category. "
            "Load IDs are excluded from model features.\n\n"
            "The fixed December chart inputs omit market/quote signals, and labeled data "
            "ends in October. The forecast is therefore stable across December dates; no "
            "unsupported daily or seasonal trend is inferred.\n\n"
            "OCTOBER HOLDOUT RESULTS\n"
            f"MAE: ${metrics['mae']:,.2f}\n"
            f"RMSE: ${metrics['rmse']:,.2f}\n"
            f"R²: {metrics['r2']:.4f}\n\n"
            "These metrics describe the October holdout only. Validation labels are not "
            "provided, so hidden-set performance is not measurable locally."
        )
        axis.text(
            0,
            1,
            report_text,
            transform=axis.transAxes,
            fontsize=12,
            linespacing=1.55,
            va="top",
            wrap=True,
        )
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        if chart_path.is_file():
            chart = plt.imread(chart_path)
            figure, axis = plt.subplots(figsize=(11.69, 8.27))
            axis.imshow(chart)
            axis.axis("off")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train the freight-rate model and write assessment outputs."
    )
    parser.add_argument("--train", type=Path, default=root / "train-test.csv")
    parser.add_argument("--validation", type=Path, default=root / "validation.csv")
    parser.add_argument(
        "--template", type=Path, default=root / "validation-predictions-template.csv"
    )
    parser.add_argument(
        "--december-input", type=Path, default=root / "december-chart-inputs.csv"
    )
    parser.add_argument("--predictions", type=Path, default=root / "validation_predictions.csv")
    parser.add_argument(
        "--december-output", type=Path, default=root / "december_chart_predictions.csv"
    )
    parser.add_argument("--report", type=Path, default=root / "validation_report.pdf")
    parser.add_argument("--output-dir", type=Path, default=root / "scorer_results")
    args = parser.parse_args()

    labeled = pd.read_csv(args.train)
    required = {"date", "distance", TARGET, *CATEGORICAL_FEATURES}
    missing = required - set(labeled.columns)
    if missing:
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")
    if (labeled["distance"] <= 0).any():
        raise ValueError("Training data contains non-positive distances")

    metrics, dated = train_and_evaluate(labeled)
    levels = category_levels(dated)
    final_model = make_model()
    final_model.fit(
        make_features(dated, levels),
        dated[TARGET].to_numpy() / dated["distance"].to_numpy(),
    )

    validation = pd.read_csv(args.validation)
    template = pd.read_csv(args.template)
    if validation["load_id"].duplicated().any() or template["load_id"].duplicated().any():
        raise ValueError("Validation inputs contain duplicate load_id values")
    if set(validation["load_id"]) != set(template["load_id"]):
        raise ValueError("Validation rows and prediction template have different load_id values")
    if (validation["distance"] <= 0).any():
        raise ValueError("Validation data contains non-positive distances")
    validation_rates = np.maximum(
        final_model.predict(make_features(validation, levels)), 0.01
    ) * validation["distance"].to_numpy()
    rate_by_id = pd.Series(validation_rates, index=validation["load_id"]).to_dict()
    predictions = template[["load_id"]].copy()
    predictions["predicted_rate"] = predictions["load_id"].map(rate_by_id)
    predictions.to_csv(args.predictions, index=False)
    validate_predictions(predictions)

    december = pd.read_csv(args.december_input)
    december["predicted_rate"] = np.maximum(
        final_model.predict(make_features(december, levels)), 0.01
    ) * december["distance"].to_numpy()
    december.to_csv(args.december_output, index=False)
    checked_december = validate_december(december)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = args.output_dir / "candidate_december.png"
    save_december_chart(checked_december, chart_path)
    write_report(metrics, chart_path, args.report)

    print(
        f"October holdout: MAE ${metrics['mae']:,.2f}; "
        f"RMSE ${metrics['rmse']:,.2f}; R2 {metrics['r2']:.4f}"
    )
    print(f"Wrote {len(predictions):,} validation predictions: {args.predictions}")
    print(f"Wrote {len(december):,} December predictions: {args.december_output}")
    print(f"Wrote report: {args.report}")
    print(f"Wrote chart: {chart_path}")


if __name__ == "__main__":
    main()