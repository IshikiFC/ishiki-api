import pkg_resources

from api.internal.grf import GrfAgent
from gfootball.env import script_helpers


def _read_match(fp):
    grf_agent = GrfAgent()
    dump_raw = script_helpers.ScriptHelpers().load_dump(fp)

    match = []
    for frame_raw in dump_raw[:300]:
        obs_raw = frame_raw['observation']
        obs = _convert_observation(obs_raw)
        grf_agent.step(obs)

        frame = dict()
        for obs_key in ['ball', 'left_team', 'right_team']:
            frame[obs_key] = obs_raw[obs_key].tolist()
        frame['action'] = grf_agent.get_action_probs()
        frame['value'] = grf_agent.get_value()
        match.append(frame)

    return match


def _convert_observation(observation_raw):
    observation_raw['active'] = int(observation_raw['left_team_designated_player'])
    return [observation_raw]


_match = _read_match(
    pkg_resources.resource_filename('api', 'resources/matches/episode_done_20210918-064519149416.dump')
)


def get_match(num_steps=-1):
    return _match[:num_steps] if num_steps > 0 else _match
