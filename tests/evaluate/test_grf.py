import json

import pytest

from api.evaluate.grf import build_smm_stacked, evaluate, validate

obs_empty = {
    'ball': [0, 0, 0],
    'left_team': [[0, 0] for _ in range(11)],
    'right_team': [[0, 0] for _ in range(11)],
    'active': 0
}


def test_evaluate():
    response = evaluate(obs_empty)
    assert 'action' in response
    assert 'value' in response
    assert json.dumps(response)


def test_build_stacked_smm():
    smm_stacked = build_smm_stacked(obs_empty)
    assert smm_stacked.shape == (1, 72, 96, 16)


def test_validate():
    with pytest.raises(ValueError) as e:
        validate({})
    assert 'invalid observation' in str(e)
