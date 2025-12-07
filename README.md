# cs230-project

## Weather Event Database and Delivery Route Impact Assessment

This project extracts weather events from NOAA's storm events database, stores them in SQLite relational databases (SQL-based, table-structured), and applies a deterministic weighting algorithm to assess and label delivery routes based on their historical weather event impact.

## Dataset

All inputs are stored as SQLite files under data/:
* california_events.db: historical weather events (24k entries)
* delivery_routes.db: generated routes without labels
* route_events.db: route–event associations (not included due to size)
* routes_scores.db: final labeled dataset used for model training

Each route has:
* a comma-separated list of counties
* a computed impact score
* a binary label based on a fixed threshold (0.25)

This labeled dataset is directly used by all models.

## Running Models

Install dependencies:
```pip install -r requirements.txt```

Then run the model scripts:

```python scripts/train_logreg_baseline.py
python scripts/train_random_forest.py
python scripts/train_mlp_models.py
```

Each script trains a model and writes outputs to ```results/```, including evaluation metrics, PR curves, confusion matrices, and saved probability arrays.

To generate combined comparison plots:
```python scripts/plot_model_curve.py```

## Repository Structure
```data/               SQLite datasets
scripts/            Python training and evaluation scripts
src/                data extraction and scoring utilities
notebooks/          exploratory development notebooks
results/            saved figures, metrics, and model outputs
models/             trained neural network models
examples/           sample route and event records
```

## Scripts Provided
Located under ```scripts/```:
* train_logreg_baseline.py
* train_random_forest.py
* train_mlp_models.py
* plot_model_curve.py

These are the .py equivalents of the development notebooks.

## Notebooks
The corresponding exploratory notebooks are located in notebooks/:
* logreg_baseline.ipynb
* random_forest.ipynb
* deep_model_mlp.ipynb
* error_analysis.ipynb

These were used during development; the .py scripts reproduce final results.