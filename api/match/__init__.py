from logging import getLogger

from flask import jsonify, request

from api import app
from api.match.match import get_match

LOGGER = getLogger(__name__)


@app.route('/match', methods=['GET'])
def get_match_api():
    kwargs = {
        'name': request.args.get('name', None),
        'num_steps': int(request.args.get('step', -1)),
        'cache': request.args.get('cache', True, type=bool)
    }
    try:
        return jsonify(get_match(**kwargs))
    except Exception as e:
        LOGGER.exception(f'failed to get match: {kwargs}')
        return jsonify({'message': str(e)}), 400
