from collections import OrderedDict

import numpy as np
import pkg_resources
import tensorflow.compat.v1 as tf

from gfootball.env import observation_preprocessing, football_action_set
from gfootball.env.players.ppo2_cnn import Player, ObservationStacker


def build_model_path():
    return pkg_resources.resource_filename('api', 'resources/models/11_vs_11_easy_stochastic_v2')


def build_player():
    global _player
    if _player:  # already initialized
        return _player
    player_config = {
        'index': 0,
        'left_players': 1,
        'right_players': 0,
        'policy': 'gfootball_impala_cnn',
        'checkpoint': build_model_path()
    }
    env_config = {}
    _player = Player(player_config, env_config)
    return _player


_player = None  # cache player to avoid loading tf variables multiple times


class GrfAgent:
    name = 'grf'

    def __init__(self):
        player = build_player()
        self.policy = player._policy
        self.observation_stacker = ObservationStacker(4)
        self.actions = football_action_set.action_set_dict['default']
        self._initialize()

    def _initialize(self):
        self.probs = np.ones(1)
        self.value = 0
        self.action = 0

    def reset(self):
        self.observation_stacker.reset()
        self._initialize()

    def step(self, obs):
        smm = observation_preprocessing.generate_smm(obs)
        action, probs, value = self.policy._evaluate(
            [self.policy.action, tf.nn.softmax(self.policy.pd.logits), self.policy.vf],
            self.observation_stacker.get(smm)
        )
        self.probs = probs[0]
        self.value = value[0]
        self.action = action[0]
        return self.action

    def get_action(self, to_name=False):
        return str(self.actions[self.action]) if to_name else self.action

    def get_action_probs(self, to_name=False, to_list=False):
        probs_dict = OrderedDict()
        for idx, prob in enumerate(self.probs):
            key = str(self.actions[idx]) if to_name else idx
            probs_dict[key] = float(prob)
        if to_list:
            return list([e for e in probs_dict.items()])
        else:
            return probs_dict

    def get_value(self):
        return float(self.value)
