import json

from api import app
from api.evaluate.grf import evaluate as evaluate_grf


@app.route('/evaluate/grf', methods=['GET', 'POST'])
def evaluate_grf_api():
    obs = {
        'ball': [0, 0, 0],
        'left_team': [[0, 0] for _ in range(11)],
        'right_team': [[0, 0] for _ in range(11)],
        'left_team_designated_player': 0
    }
    return json.dumps(evaluate_grf(obs))
