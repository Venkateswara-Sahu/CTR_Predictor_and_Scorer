from pathlib import Path
import json
from streamlit.testing.v1 import AppTest


def test_dashboard_uses_saved_results_and_predicts():
    app = AppTest.from_file('streamlit_app.py', default_timeout=30).run()
    assert not app.exception
    app.text_area[0].set_value(json.dumps({'I1': 10**400})).run()
    app.button[0].click().run()
    assert len(app.error) == 1
    assert not app.exception
    report = json.loads(Path('evidence/benchmark.json').read_text())
    assert app.metric[0].value == f"{report['metrics']['selected']['roc_auc']:.4f}"
    app.text_area[0].set_value(json.dumps({'I1': 4, 'C1': 'demo-category'})).run()
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[-1].label == 'Predicted click probability'
    app.text_area[0].set_value('{bad json').run()
    app.button[0].click().run()
    assert len(app.error) == 1
    assert not app.exception


def test_dashboard_batch_ranking():
    app = AppTest.from_file('streamlit_app.py', default_timeout=30).run()
    app.button[1].click().run()
    assert not app.exception
    ranking = app.dataframe[0].value
    assert len(ranking) == 2
    assert (ranking.predicted_ctr == ranking.final_score).all()
