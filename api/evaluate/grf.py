from logging import getLogger

from api.internal.grf import GrfAgent

LOGGER = getLogger(__name__)

agent = GrfAgent()


def evaluate(obs):
    agent.reset()
    agent.step([obs])
    return {
        'action': agent.get_action_probs(),
        'value': float(agent.get_value())
    }


def validate(obs):
    try:
        assert obs is not None, 'must pass JSON observation in request body'
        for key in ['ball', 'left_team', 'right_team', 'active']:
            assert key in obs, f'must include \'{key}\''
        assert len(obs['ball']) == 3, '\'ball\' must be 3D'
        assert len(obs['left_team']) == 11, '\'left_team\' must be 11D'
        assert len(obs['right_team']) == 11, '\'right_team\' must be 11D'
    except Exception as e:
        raise ValueError(f'invalid observation: {str(e)}') from e
