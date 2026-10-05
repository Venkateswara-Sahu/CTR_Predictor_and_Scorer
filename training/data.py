import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from app.feature_engineer import RAW_FEATURES

WINDOWS = {'train': (0, 600000), 'validation': (1000000, 1150000),
           'test': (10000001, 10250001)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_windows(source, output, windows=WINDOWS):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {key: output / f'{key}.tsv' for key in windows}
    handles = {key: path.open('wb') for key, path in paths.items()}
    counts = dict.fromkeys(windows, 0)
    try:
        with open(source, 'rb') as stream:
            stop = max(end for start, end in windows.values())
            for row, line in enumerate(stream):
                if row >= stop:
                    break
                for key, (start, end) in windows.items():
                    if start <= row < end:
                        handles[key].write(line)
                        counts[key] += 1
                if row and row % 1000000 == 0:
                    print(f'Scanned {row:,} source rows', flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    if any(counts[key] != end - start for key, (start, end) in windows.items()):
        raise ValueError('Source is too short for frozen windows')
    return paths


def load_split(path, seen, offset):
    rows, ids = [], []
    removed, count = 0, 0
    with open(path, 'rb') as stream:
        for i, line in enumerate(stream):
            count += 1
            values = line.rstrip(b'\r\n').split(b'\t')
            if len(values) != 40 or values[0] not in (b'0', b'1'):
                raise ValueError(f'Malformed source row {offset + i}')
            fingerprint = hashlib.sha256(b'\t'.join(values[1:])).digest()
            if fingerprint in seen:
                removed += 1
                continue
            seen.add(fingerprint)
            rows.append([v.decode('ascii') if v else None for v in values])
            ids.append(offset + i)
    frame = pd.DataFrame(rows, columns=['click'] + RAW_FEATURES)
    frame['click'] = frame['click'].astype(np.int8)
    for c in RAW_FEATURES[:13]:
        frame[c] = pd.to_numeric(frame[c])
    frame['source_row'] = ids
    metadata = {'source_window_rows': count, 'retained_rows': len(frame),
                'removed_duplicate_features': removed, 'positive_rows': int(frame.click.sum()),
                'click_rate': float(frame.click.mean()), 'sha256': sha256(path),
                'missing_fraction': frame[RAW_FEATURES].isna().mean().to_dict()}
    return frame, metadata
