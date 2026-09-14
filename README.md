# CTR Prediction and Ad Scoring System

An academic machine-learning project for training, comparing, and serving click-through-rate models on a 10-million-row sample of the Criteo Display Advertising dataset.

[Live Streamlit app](https://ctrpredictor.streamlit.app/) · [Deployment notes](STREAMLIT_DEPLOYMENT.md)

## Results

| Model | Test AUC | Log loss |
|---|---:|---:|
| XGBoost | **0.9067** | **0.3105** |
| LightGBM | 0.9024 | 0.3180 |
| Ensemble | 0.9067 | 0.3106 |

- **Data:** 10 million Criteo records, split into 7 million training, 1 million validation, and 2 million test rows.
- **Features:** 150 model features derived from 13 integer and 26 categorical input fields.
- **Ranking analysis:** the recorded experiment reports 265.6% CTR lift in the top prediction decile.
- **Serving:** the repository includes a Flask API and Streamlit interface; single-prediction latency was below 0.5 seconds in project tests.

These are offline academic results, not online A/B-test or commercial-impact measurements. Latency varies with hardware, model loading, and request concurrency.

## What is included

- Feature engineering for sparse numerical and categorical advertising data.
- XGBoost and LightGBM model wrappers plus Optuna-based tuning support.
- Ad scoring and ranking utilities.
- Flask endpoints for health checks, prediction, scoring, and batch workflows.
- Streamlit dashboard for model inspection and what-if scoring.

## Repository structure

```text
.
├── app/
│   ├── feature_engineer.py       # transforms 39 raw fields into model features
│   ├── ctr_model.py              # model loading and prediction
│   ├── ad_scorer.py              # score calculation
│   └── ad_ranking_system.py      # ranking workflow
├── app.py                        # Flask API
├── streamlit_app.py              # interactive dashboard
├── download_models.py            # retrieves packaged model artifacts
├── test_api.py                   # live API smoke checks
└── requirements.txt
```

## Quick start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Venkateswara-Sahu/CTR_Predictor_and_Scorer.git
cd CTR_Predictor_and_Scorer
python -m venv .venv
```

Activate the environment, then install dependencies and retrieve the model artifacts:

```bash
pip install -r requirements.txt
python download_models.py
```

Run either interface:

```bash
python app.py
streamlit run streamlit_app.py
```

The API listens on `http://localhost:5000`; Streamlit normally uses `http://localhost:8501`.

To exercise the live API, start `python app.py` in one terminal and run this in another:

```bash
python test_api.py
```

## Evaluation notes

The metrics above are the values displayed by the project application and recorded during the Term 7 Predictive Analysis project (Sep–Nov 2025). The large training dataset and model artifacts are not stored directly in Git. Reproducing the exact scores requires the same Criteo sample, split, preprocessing configuration, and random seeds used for the recorded experiment.

## Limitations

- The repository is a portfolio and academic implementation; it has not served production advertising traffic.
- Offline AUC and top-decile lift do not demonstrate revenue lift or improved user experience.
- Model artifacts are downloaded separately and should be checked before deployment in another environment.
- The API smoke script expects a locally running Flask service and is not an isolated unit-test suite.

## Attribution

Dataset: [Criteo Display Advertising Challenge](https://ailab.criteo.com/ressources/). Developed for the Predictive Analysis course at Lovely Professional University.
