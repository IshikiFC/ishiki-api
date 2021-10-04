from logging import getLogger

import tensorflow.compat.v1 as tf

from api.internal.grf import player
from gfootball.env import football_action_set
from gfootball.env import observation_preprocessing
from gfootball.env.players.ppo2_cnn import ObservationStacker

LOGGER = getLogger(__name__)

policy = player._policy


def evaluate(obs):
    """
    Evaluate observation with Google Research Football trained PPO checkpoint
    https://github.com/google-research/football#trained-checkpoints
    """

    validate(obs)
    smm_stacked = build_smm_stacked(obs)
    return evaluate_internal(smm_stacked)


def evaluate_internal(smm_stacked):
    action_probs, value_estimate = policy._evaluate([tf.nn.softmax(policy.pd.logits), policy.vf], smm_stacked)
    action_probs = action_probs[0]
    value_estimate = value_estimate[0]

    actions = football_action_set.action_set_dict['default']
    action_prob_dict = dict([
        (str(action), round(float(prob), 2))
        for action, prob in zip(actions, action_probs)
    ])

    return {
        'action': action_prob_dict,
        'value': float(value_estimate)
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


def build_smm_stacked(obs):
    smm = observation_preprocessing.generate_smm([obs])
    observation_stacker = ObservationStacker(4)
    return observation_stacker.get(smm)
