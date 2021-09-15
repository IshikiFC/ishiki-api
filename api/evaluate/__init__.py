from logging import getLogger

from flask import request, jsonify

from api import app
from api.evaluate.grf import evaluate as evaluate_grf

LOGGER = getLogger(__name__)


@app.route('/evaluate/grf', methods=['GET', 'POST'])
def evaluate_grf_api():
    obs = request.get_json()
    LOGGER.info(f'evaluate: {obs}')
    try:
        return jsonify(evaluate_grf(obs))
    except ValueError as e:
        LOGGER.warning(f'failed to evaluate: {obs}, {e}')
        return jsonify({'message': str(e)}), 400
