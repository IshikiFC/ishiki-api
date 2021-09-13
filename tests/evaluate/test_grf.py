from api.evaluate.grf import build_smm_stacked, evaluate

obs_empty = {
    'ball': [0, 0, 0],
    'left_team': [[0, 0] for _ in range(11)],
    'right_team': [[0, 0] for _ in range(11)],
    'left_team_designated_player': 0
}


def test_evaluate():
    result = evaluate(obs_empty)
    assert result


def test_build_stacked_smm():
    smm_stacked = build_smm_stacked(obs_empty)
    assert smm_stacked.shape == (1, 72, 96, 16)
