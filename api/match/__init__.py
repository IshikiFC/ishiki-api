from logging import getLogger

from flask import jsonify, request

from api import app
from api.match.match import get_match

LOGGER = getLogger(__name__)


@app.route('/match', methods=['GET'])
def get_match_api():
    kwargs = {
        'num_steps': int(request.args.get('s', -1))
    }
    return jsonify(get_match(**kwargs))
