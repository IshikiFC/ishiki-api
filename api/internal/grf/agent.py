import numpy as np
import pkg_resources
import tensorflow.compat.v1 as tf

from gfootball.env import observation_preprocessing, football_action_set
from gfootball.env.players.ppo2_cnn import Player, ObservationStacker


def build_model_path():
    return pkg_resources.resource_filename('api', 'resources/models/11_vs_11_easy_stochastic_v2')


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


player = build_player()  # build global player once to avoid loading tf variables multiple times


class GrfAgent:
    def __init__(self):
        self.policy = player._policy
        self.observation_stacker = ObservationStacker(4)
        self.actions = football_action_set.action_set_dict['default']

        self.probs = np.ones(1)
        self.value = 0
        self.prev_action = 0

    def step(self, obs):
        smm = observation_preprocessing.generate_smm([obs])
        action, probs, value = self.policy._evaluate(
            [self.policy.action, tf.nn.softmax(self.policy.pd.logits), self.policy.vf],
            self.observation_stacker.get(smm)
        )
        self.probs = probs[0]
        self.value = value[0]
        self.prev_action = action
        return self.prev_action

    def get_action(self, to_name=False):
        return self.actions[self.prev_action] if to_name else self.prev_action

    def get_action_probs(self, to_name=False):
        action_probs = dict()
        for idx, prob in enumerate(self.probs):
            key = self.actions[idx] if to_name else idx
            action_probs[key] = prob
        return action_probs

    def get_value(self):
        return self.value
