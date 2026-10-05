from pathlib import Path
from training.data import extract_windows, load_split


def line(label, category):
    return f'{label}\t' + '\t'.join(['1'] * 13 + [category] * 26) + '\n'


def test_streaming_windows_and_label_independent_dedup(tmp_path):
    source = tmp_path / 'train.txt'
    source.write_text(line(0, 'a') + line(1, 'a') + line(1, 'b') + line(0, 'c'))
    paths = extract_windows(source, tmp_path / 'out', {'train': (0, 2), 'test': (2, 4)})
    seen = set()
    train, metadata = load_split(paths['train'], seen, 0)
    assert len(train) == 1
    assert metadata['removed_duplicate_features'] == 1
    assert train['click'].tolist() == [0]
    test, metadata = load_split(paths['test'], seen, 2)
    assert test['source_row'].tolist() == [2, 3]
    assert len(metadata['sha256']) == 64
