import six.moves.cPickle

from gfootball.env import script_helpers


def write_dump(dump, fp):
    with open(fp, 'wb') as f:
        for step in dump:
            six.moves.cPickle.dump(step, f)


def read_dump(fp):
    return script_helpers.ScriptHelpers().load_dump(fp)


def convert_observation(observation_raw):
    observation_raw['active'] = int(observation_raw['left_team_designated_player'])  # for grf
    observation_raw['sticky_actions'] = observation_raw['left_agent_sticky_actions'][0]  # for tamakeri
    return [observation_raw]
