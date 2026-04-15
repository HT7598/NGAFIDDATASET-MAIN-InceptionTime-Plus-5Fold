# NGAFID Before/After Reproduction

This project reproduces the binary maintenance-event detection task described in `renwu.txt`: classify flights as maintenance `before` vs `after` using the NGAFID benchmark subset (`2days`).

## Environment
Create and activate a project-local virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset
This repository expects the benchmark subset to already exist locally in:

- `2days/flight_data.pkl`
- `2days/flight_header.csv`
- `2days/stats.csv`

The loader has been adjusted to prefer local datasets and skip downloading when these files already exist.

## Run one formal fold
This is the recommended first run to verify the script in a clean environment.

```bash
python train_before_after_cv.py --only-fold 0 --epochs 10 --steps-per-epoch 100
```

Outputs will be written to:

- `outputs/before_after_cv/fold_0/history.csv`
- `outputs/before_after_cv/fold_0/predictions.csv`
- `outputs/before_after_cv/fold_0/metrics.json`
- `outputs/before_after_cv/fold_0/best_before_after.weights.h5`

## Run 5-fold cross validation

```bash
python train_before_after_cv.py --epochs 10 --steps-per-epoch 100
```

Cross-validation outputs:

- `outputs/before_after_cv/cv_results.csv`
- `outputs/before_after_cv/summary.json`
- `outputs/before_after_cv/fold_*/...`

## Metrics
The script reports and saves three metrics for each fold:

- Accuracy
- F1
- ROC-AUC

These were selected to satisfy the task requirement in `renwu.txt` and to provide a more reliable evaluation than accuracy alone.

## Notes
- The script is extracted from the TensorFlow notebook and no longer depends on Colab- or TPU-specific cells.
- The current script uses the Inception-style baseline model for the `before_after` task.
- For reproducibility, fold-level predictions and metrics are saved to disk.
