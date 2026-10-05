import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from app.feature_engineer import FeatureEngineer


def sample():
    return pd.DataFrame({'I1': [None, 0, 3, 99, 5, 7, 9, 11, 13, 17],
                         'C1': ['a', 'b', 'a', None, 'c', 'd', 'e', 'f', 'g', 'h']})


def test_missing_and_batch_invariance(tmp_path):
    frame = sample()
    engineer = FeatureEngineer().fit(frame, np.arange(10) % 2)
    full = engineer.transform(frame)
    assert full.shape == (10, 123)
    assert full.loc[0, 'I1_missing'] == 1
    assert full.loc[1, 'I1_missing'] == 0
    pd.testing.assert_frame_equal(full.iloc[[0]].reset_index(drop=True),
                                  engineer.transform(frame.iloc[[0]]).reset_index(drop=True))
    pd.testing.assert_frame_equal(full, engineer.transform(frame.iloc[::-1]).iloc[::-1])
    path = tmp_path / 'transformer.json'
    engineer.save(path)
    pd.testing.assert_frame_equal(full, FeatureEngineer.load(path).transform(frame))
    code = "from app.feature_engineer import FeatureEngineer; import pandas as pd; import json; import sys; print(FeatureEngineer.load(sys.argv[1]).transform(pd.DataFrame([{'C1':'a'}])).to_json())"
    outputs = [subprocess.check_output([sys.executable, '-c', code, str(path)],
                                      env={**os.environ, 'PYTHONHASHSEED': str(seed)})
               for seed in (1, 2)]
    assert outputs[0] == outputs[1]


def test_cross_fit_excludes_own_fold_labels():
    x = sample()
    y = np.arange(10) % 2
    a = FeatureEngineer().fit_transform(x, y)
    changed = y.copy()
    changed[0] = 1 - changed[0]
    b = FeatureEngineer().fit_transform(x, changed)
    assert a.loc[0, 'C1_target'] == b.loc[0, 'C1_target']


def test_unknown_and_bad_inputs(tmp_path):
    engineer = FeatureEngineer().fit(sample(), np.arange(10) % 2)
    out = engineer.transform(pd.DataFrame([{'C1': 'never_seen', 'I1': None}]))
    assert np.isfinite(out.to_numpy()).all()
    assert out.loc[0, 'C1_frequency'] == 0
    assert out.loc[0, 'C1_target'] == engineer.state['prior']
    for frame in (pd.DataFrame([{'C1': 1}]), pd.DataFrame([{'C1': 1}, {'C1': None}])):
        with pytest.raises(ValueError, match='string'):
            engineer.transform(frame)
    for frame in (pd.DataFrame(), pd.DataFrame([{'I1': 'bad'}]),
                  pd.DataFrame([{'I1': float('inf')}]), pd.DataFrame([{'click': 1}])):
        with pytest.raises(ValueError):
            engineer.transform(frame)
    path = tmp_path / 'old.json'
    path.write_text(json.dumps({'schema_version': 1}))
    with pytest.raises(ValueError):
        FeatureEngineer.load(path)
