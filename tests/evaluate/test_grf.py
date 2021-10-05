import json
import pytest

from api.evaluate.grf import evaluate, validate

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


def test_validate():
    with pytest.raises(ValueError) as e:
        validate({})
    assert 'invalid observation' in str(e)
