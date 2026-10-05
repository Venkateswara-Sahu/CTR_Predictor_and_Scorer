"""Optional smoke check against a running v2 local API: python test_api.py."""
import argparse
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError


def call(base, path, payload=None):
    request = Request(base + path, data=None if payload is None else json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='http://127.0.0.1:5000')
    args = parser.parse_args()
    status, health = call(args.base, '/health')
    assert status == 200 and health['schema_version'] == 2
    rows = [{'I1': 4, 'C1': 'synthetic-a'}, {'I1': 9, 'C1': 'synthetic-b'}]
    status, batch = call(args.base, '/predict_ctr', {'features': rows})
    assert status == 200 and batch['count'] == 2
    status, single = call(args.base, '/predict_single', {'features': rows[0]})
    assert status == 200 and abs(single['predicted_ctr'] - batch['predictions'][0]) < 1e-12
    status, ranked = call(args.base, '/rank_ads', {'ads': rows, 'top_k': 2})
    assert status == 200 and ranked['returned_ads'] == 2
    assert all(row['final_score'] == row['predicted_ctr'] for row in ranked['ranked_ads'])
    assert call(args.base, '/score_ad', {'features': rows[0]})[0] == 410
    assert call(args.base, '/predict_ctr', {'features': {'I1': 'bad'}})[0] == 400
    print('V2 health, single/batch, ranking, retired-quality and validation smoke checks passed.')


if __name__ == '__main__':
    main()
