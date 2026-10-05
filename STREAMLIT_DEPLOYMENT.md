# Streamlit deployment

Deploy this v2 branch with Python 3.12, `requirements.txt`, and entry point
`streamlit_app.py`. The model/transformer/manifest under `app/models` and the
matching `evidence/benchmark.json` must ship together. No release download is
required. A manifest/report mismatch or a missing/checksum-invalid selected model
stops the app with a setup error; it never falls back to historical assets.

Validate locally with `python -m pytest` and `streamlit run streamlit_app.py` before
merging. Public deployment remains on its existing revision until this branch is
merged and Streamlit completes a successful rebuild; a local run is not deployment.

The app presents artifact-backed bounded-study results, raw JSON predictions,
CSV batches and candidate CTR ranking. Anonymous Criteo inputs have no established
semantic quality labels. No conversion, revenue, causal lift or SLA claim is made.
