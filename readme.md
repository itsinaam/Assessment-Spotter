# Freight Rate Prediction Challenge

See `freight-rate-ml-assessment.pdf` for the assessment instructions.

## Run

The CSV files are in the repository root. Use Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
python train.py
python score.py --predictions validation_predictions.csv --december-predictions december_chart_predictions.csv
```

`train.py` uses a chronological October holdout, reports MAE/RMSE/R², then refits on all labeled data. It saves `validation_predictions.csv`, `december_chart_predictions.csv`, `validation_report.pdf`, and `scorer_results/candidate_december.png`. The model predicts rate per mile and scales by distance; numeric missing values and unseen city categories are handled by the model.

The scorer checks that the validation file contains exactly `load_id,predicted_rate` for all 12,000 loads, verifies the December inputs, and regenerates the chart. Final hidden-label validation metrics are calculated by Spotter after submission.

## Submit

- Accessible GitHub repository with the code, dependencies, and run instructions
- `validation_predictions.csv`
- `validation_report.pdf`
- A 2–3 minute Loom walkthrough