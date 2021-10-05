import json

from api.match.match import get_match


def test_get_match():
    num_steps = 10
    match = get_match(num_steps=num_steps)
    assert len(match) == num_steps
    assert json.dumps(match)
    for frame in match:
        for field in ['ball', 'left_team', 'right_team', 'action', 'value']:
            assert field in frame
