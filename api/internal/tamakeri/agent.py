from collections import OrderedDict

import numpy as np
import pkg_resources
import stringcase

from api.internal.tamakeri.env import Environment, to_tamakeri_obs, Action, KICK_ACTIONS
from api.internal.tamakeri.model import load_model


def build_model_path():
    return pkg_resources.resource_filename('api', 'resources/models/1679.pth')


def to_actions(action_idx):
    """
    convert tamakeri action index (0-52) to list of raw actions
    see Environment.special_to_actions() for details
    """

    assert 0 <= action_idx < Environment.ACTION_LEN + 1
    if 0 <= action_idx < 19:
        return [Action(action_idx)]
    elif action_idx == 19:
        return [Action.Idle]
    else:
        for kick_action, kick_idx in KICK_ACTIONS.items():
            if kick_idx <= action_idx < kick_idx + 8:
                return [kick_action, Action(action_idx - kick_idx + 1)]


def to_action_name(action_idx):
    name_upper_camel = ''.join([a.name for a in to_actions(action_idx)])
    return stringcase.snakecase(stringcase.camelcase(name_upper_camel))


def to_base_action(action_idx):
    return to_actions(action_idx)[0]


class TamakeriAgent:
    name = 'tamakeri'

    def __init__(self):
        self.env = Environment()
        self.model = load_model(self.env.net()(self.env), build_model_path())
        self.action_names = [to_action_name(idx) for idx in range(Environment.ACTION_LEN + 1)]
        self.base_actions = [to_base_action(idx) for idx in range(Environment.ACTION_LEN + 1)]

        self._initialize()

    def _initialize(self):
        self.probs = np.ones(1)
        self.value = 0
        self.action = 0
        self.reserved_action = None

    def reset(self):
        self.env.reset()
        self._initialize()

    def step(self, obs):
        info = [{'observation': to_tamakeri_obs(obs), 'action': [self.action]}, None]
        self.env.play_info(info)
        logits, value, _, _ = self.model.inference(self.env.observation(0), None)
        self.probs = np.exp(logits) / sum(np.exp(logits))
        self.value = value[0]

        actions = self.env.legal_actions(0)
        action = max(actions, key=lambda x: self.probs[x])
        if self.reserved_action is not None:
            self.action, self.reserved_action = self.reserved_action, None
        else:
            self.action, self.reserved_action = self.env.special_to_actions(action)
        return self.action

    def get_action(self, to_name=False):
        return to_action_name(self.action) if to_name else self.action

    def get_action_probs(self, to_name=False, to_base=False, to_list=False):
        probs_dict = dict()
        for idx_raw, prob in enumerate(self.probs):
            idx = self.base_actions[idx_raw].value if to_base else idx_raw
            probs_dict[idx] = probs_dict.get(idx, 0) + float(prob)

        probs_dict_ordered = OrderedDict()
        for idx in sorted(probs_dict.keys()):
            key = self.action_names[idx] if to_name else idx
            probs_dict_ordered[key] = probs_dict[idx]

        if to_list:
            return list([e for e in probs_dict_ordered.items()])
        else:
            return probs_dict_ordered

    def get_value(self):
        return float(self.value)
