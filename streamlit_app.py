"""CTR demo with artifact-backed evidence and one shared inference pipeline."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from app.ad_ranking_system import AdRankingSystem
from app.ctr_model import file_hash

st.set_page_config(page_title='CTR | Prediction & Evidence', page_icon=':material/ads_click:', layout='wide')
ROOT = Path(__file__).parent
BUNDLE = ROOT / 'app/models'


@st.cache_resource
def load_system(bundle_digest):
    return AdRankingSystem(BUNDLE)


@st.cache_data
def load_report(report_digest):
    return json.loads((ROOT / 'evidence/benchmark.json').read_text(encoding='utf-8'))


st.title('Click probability. Clear evidence.')
st.write('Predict clicks from anonymous Criteo features and compare candidates by predicted probability.')
try:
    system = load_system(file_hash(BUNDLE / 'manifest.json'))
    report = load_report(file_hash(ROOT / 'evidence/benchmark.json'))
    if report['manifest_sha256'] != file_hash(BUNDLE / 'manifest.json'):
        raise ValueError('Model bundle and evaluation report do not match')
except (OSError, ValueError, KeyError) as error:
    st.error(f'Compatible model/evidence bundle required: {error}')
    st.stop()

with st.sidebar:
    st.header('CTR Predictor')
    st.caption('v2 · shared training and serving pipeline')
    st.write(f"Model: **{report['selection']}**")
    st.write(f"Test sample: **{report['splits']['test']['retained_rows']:,} rows**")
    st.caption('Anonymous fields have no verified placement, device or quality meaning.')
    st.link_button('Source & evaluation protocol', 'https://github.com/Venkateswara-Sahu/CTR_Predictor_and_Scorer')

selected = report['metrics']['selected']
with st.container(horizontal=True):
    st.metric('Test ROC AUC', f"{selected['roc_auc']:.4f}", border=True)
    st.metric('Test log loss', f"{selected['log_loss']:.4f}", border=True)
    st.metric('Test Brier score', f"{selected['brier']:.4f}", border=True)
    st.metric('Top-decile enrichment', f"{selected['top_decile']['lift']:.2f}×", border=True)
st.caption('AUC measures ranking discrimination. Top-decile enrichment compares observed click rates within this test sample; it is not a causal business improvement.')
predict_tab, batch_tab, evidence_tab = st.tabs(['Predict a row', 'Batch & ranking', 'Evaluation evidence'])

with predict_tab:
    st.subheader('Try a raw feature row')
    st.write('Use I1–I13 for numbers or null; C1–C26 for strings or null. Omitted fields are treated as missing.')
    st.caption('The default is a synthetic input example. Partial and unfamiliar rows carry no performance guarantee.')
    with st.form('single_prediction'):
        text = st.text_area('Raw features (JSON)', value=json.dumps({'I1': 4, 'I2': 0, 'C1': 'demo-category', 'C2': None}, indent=2), height=190)
        submit = st.form_submit_button('Predict click probability', type='primary')
    if submit:
        try:
            values = json.loads(text)
            if not isinstance(values, dict) or not values:
                raise ValueError('Provide one nonempty raw-feature object')
            result = system.predict_single_ad(values)
            st.metric('Predicted click probability', f"{result['predicted_ctr']:.2%}")
            st.caption('This is a model estimate, not an observed click or guaranteed outcome.')
        except (ValueError, TypeError, OverflowError) as error:
            st.error(str(error))

with batch_tab:
    st.subheader('Compare candidate rows')
    st.write('Upload a CSV containing raw feature columns, or paste a JSON list. Maximum 1,000 rows; categories must remain strings.')
    upload = st.file_uploader('Raw feature CSV', type=['csv'])
    text_batch = st.text_area('Raw feature rows (JSON)', value=json.dumps([{'I1': 4, 'C1': 'demo-a'}, {'I1': 9, 'C1': 'demo-b'}], indent=2), height=150)
    if st.button('Predict & rank candidates', type='primary'):
        try:
            if upload is not None:
                rows = pd.read_csv(upload, dtype={f'C{i}': 'string' for i in range(1, 27)})
            else:
                values = json.loads(text_batch)
                if not isinstance(values, list) or not all(isinstance(v, dict) and v for v in values):
                    raise ValueError('Provide a list of nonempty feature objects')
                rows = pd.DataFrame(values)
            if not 1 <= len(rows) <= 1000:
                raise ValueError('Provide 1..1000 rows')
            ranked = system.rank_ads(rows)
            st.dataframe(ranked, hide_index=True, column_config={'predicted_ctr': st.column_config.NumberColumn('Click probability', format='%.4f')})
            st.download_button('Download predictions', ranked.to_csv(index=False), 'ctr_predictions.csv', 'text/csv')
            st.caption('ad_id is the zero-based input row. final_score equals predicted_ctr. This orders supplied candidates; no live ranking outcome was measured.')
        except (ValueError, TypeError, OverflowError, pd.errors.ParserError) as error:
            st.error(str(error))

with evidence_tab:
    st.subheader('A separate, protected test sample')
    st.write('Preprocessing fits on training rows. Training target encodings exclude their own folds. Validation selects the model and threshold; the final test window is beyond the historical 10-million-row prefix.')
    st.dataframe(pd.DataFrame([{ 'Split': name, 'Source rows': values['source_window_rows'], 'Retained rows': values['retained_rows'], 'Duplicates removed': values['removed_duplicate_features'], 'Click rate': values['click_rate']} for name, values in report['splits'].items()]), hide_index=True)
    st.subheader('Baselines and selected model')
    st.dataframe(pd.DataFrame([{'Model': name, 'ROC AUC': value['roc_auc'], 'Log loss': value['log_loss'], 'Average precision': value['average_precision'], 'Brier score': value['brier']} for name, value in report['metrics'].items()]), hide_index=True)
    st.subheader('Uncertainty & calibration')
    st.dataframe(pd.DataFrame([{'Metric': name, '95% lower': interval[0], '95% upper': interval[1]} for name, interval in report['selected_ci95'].items()]), hide_index=True)
    calibration = pd.DataFrame(report['calibration_bins']).rename(columns={'mean_prediction': 'Mean predicted probability', 'observed_click_rate': 'Observed click rate'})
    st.scatter_chart(calibration, x='Mean predicted probability', y='Observed click rate', x_label='Mean predicted click probability', y_label='Observed click rate')
    with st.expander('Local inference measurement'):
        st.dataframe(pd.DataFrame(report['local_warm_latency']).T, hide_index=False)
        st.caption('30 warmed repetitions on the recorded local hardware. Preprocessing included; loading, network and concurrent users excluded.')
    st.subheader('Scope of the evidence')
    for limitation in report['limitations']:
        st.write(f'• {limitation}')
    st.download_button('Download full evaluation report', json.dumps(report, indent=2), 'ctr_benchmark.json', 'application/json')
