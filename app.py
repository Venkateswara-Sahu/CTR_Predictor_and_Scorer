"""Run the v2 CTR API with python app.py."""
import os
from app.api import create_app

if __name__ == '__main__':
    app = create_app(os.environ.get('CTR_MODEL_PATH'))
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
