"""Verify checked-in v2 assets; old v1.0 auto-download is intentionally retired."""
from app.ctr_model import CTRPredictor


def download_models():
    CTRPredictor()
    return True

if __name__ == '__main__':
    download_models()
    print('Compatible v2 model bundle verified')
