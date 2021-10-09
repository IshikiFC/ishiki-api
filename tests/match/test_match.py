import json

from api.match.match import get_match


def test_get_match():
    num_steps = 10
    match = get_match(num_steps=num_steps)
    assert len(match) == num_steps
    assert json.dumps(match)
    for frame in match:
        observation = frame['observation']
        for field in ['ball', 'left_team', 'right_team', 'action']:
            assert field in observation
        evaluation = frame['evaluation']
        for agent in ['grf', 'tamakeri']:
            assert agent in evaluation
            for field in ['action', 'value']:
                assert field in evaluation[agent]
            assert evaluation[agent]['action'][0][0].lower() == 'idle'
