import numpy as np
import pkg_resources

from api.internal.tamakeri.env import Environment, to_tamakeri_obs, Action, KICK_ACTIONS
from api.internal.tamakeri.model import load_model


def build_model_path():
    return pkg_resources.resource_filename('api', 'resources/models/1679.pth')


def to_action_name(action_idx):
    """
    convert tamakeri action index (0-52) to action name
    see Environment.special_to_actions() for details
    """

    assert 0 <= action_idx < Environment.ACTION_LEN + 1
    if 0 <= action_idx < 19:
        return Action(action_idx).name
    elif action_idx == 19:
        return 'None'
    else:
        for kick, kick_idx in KICK_ACTIONS.items():
            if kick_idx <= action_idx < kick_idx + 8:
                return kick.name + Action(action_idx - kick_idx + 1).name


class TamakeriAgent:
    def __init__(self):
        self.env = Environment()
        self.model = load_model(self.env.net()(self.env), build_model_path())

        self.prev_action = 0
        self.reserved_action = None
        self.probs = np.ones(1)
        self.value = 0

    def step(self, obs):
        info = [{'observation': to_tamakeri_obs(obs), 'action': [self.prev_action]}, None]
        self.env.play_info(info)
        logits, value, _, _ = self.model.inference(self.env.observation(0), None)
        self.probs = np.exp(logits) / sum(np.exp(logits))
        self.value = self.value[0]

        actions = self.env.legal_actions(0)
        action = max(actions, key=lambda x: self.logit[x])

        if self.reserved_action is not None:
            self.prev_action, self.reserved_action = self.reserved_action, None
        else:
            self.prev_action, self.reserved_action = self.env.special_to_actions(action)
        return self.prev_action

    def get_action(self, to_name=False):
        return to_action_name(self.prev_action) if to_name else self.prev_action

    def get_action_probs(self, to_name=False):
        action_probs = dict()
        for idx, prob in enumerate(self.probs):
            key = to_action_name(idx) if to_name else idx
            action_probs[key] = prob
        return action_probs

    def get_value(self):
        return self.value
