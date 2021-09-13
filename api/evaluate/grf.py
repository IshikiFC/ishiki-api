import pkg_resources
import tensorflow.compat.v1 as tf
from gfootball.env import observation_preprocessing
from gfootball.env.players.ppo2_cnn import Player, ObservationStacker
from logging import getLogger

LOGGER = getLogger(__name__)


def build_model_path():
    return pkg_resources.resource_filename('api', 'models/11_vs_11_easy_stochastic_v2')


def build_player():
    player_config = {
        'index': 0,
        'left_players': 1,
        'right_players': 0,
        'policy': 'gfootball_impala_cnn',
        'checkpoint': build_model_path()
    }
    env_config = {}
    return Player(player_config, env_config)


# build global player once for evaluate method
player = build_player()
policy = player._policy


def evaluate(obs):
    """
    Evaluate observation with Google Research Football trained PPO checkpoint
    https://github.com/google-research/football#trained-checkpoints
    """

    smm_stacked = build_smm_stacked(obs)
    return policy._evaluate(tf.nn.softmax(policy.pd.logits), smm_stacked)[0].tolist()


def build_smm_stacked(obs):
    obs['active'] = obs['left_team_designated_player']
    smm = observation_preprocessing.generate_smm([obs])
    observation_stacker = ObservationStacker(4)
    return observation_stacker.get(smm)
